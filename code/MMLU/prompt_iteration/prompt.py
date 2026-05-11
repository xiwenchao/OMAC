import os
import time
import uuid

import numpy as np
import json
import argparse
import datetime
import logging
from prompt_iteration.chat_service import ChatService
from utils import generate_answer
from prompt_lib import *
# from LLM_Neuron import parse_ranks
# from pro_utils import load_json, save_json

MIN_INTERVAL = 0

def load_candidate_agents():
    with open('agent_config.json', 'r') as f:
        config = json.load(f)
    return config['candidate_agents']

ROLES = load_candidate_agents()

class Prompt():
    def __init__(self, evol_model, dataset_name):
        self.model = evol_model
        self.data_set = dataset_name
        self.optimized_prompts = dict()  # role: [role_name, optimized_prompt]

    def initialize(self, role, initialize_num):
        self.prompts = []
        self.role_names = []
        self.new_prompts = []
        self.roles_list = []
        self.prompt2id = dict()
        self.id2prompt = dict()
        self.id2score = dict()
        self.role = role
        # self.base_roles = ROLES if 'construct-role' not in self.optimized_prompts else ROLES + [self.optimized_prompts['construct-role'][0]]
        
        if role == 'rank':
            self.rank_format_prompt = RANK_FORMAT
            self.example_prompt = RANK_DEFAULT if role not in self.optimized_prompts else self.optimized_prompts[role][1]
            initialize_prompt = RANK_INIT.format(initialize_num-1, self.example_prompt) # + "\n\n" + self.rank_format_prompt
        if role in ROLES:
            self.example_prompt = ROLE_MAP[role] if role not in self.optimized_prompts else self.optimized_prompts[role][1]
            initialize_prompt = ROLE_INIT.format(initialize_num-1, self.role, self.example_prompt, self.example_prompt)
        if role == 'construct-role':
            self.example_prompt = ROLE_MAP["Assistant"] if role not in self.optimized_prompts else self.optimized_prompts[role][1]
            initialize_prompt = CONSTRUCT_ROLE_INIT.format(initialize_num-1, ROLES, "The role is: " + "Assistant" + "\n" + "The prompt is: " + self.example_prompt)
        if role == 'pre-rank':
            self.prerank_format_prompt = PRE_RANK_FORMAT
            self.example_prompt = PRE_RANK_DEFAULT if role not in self.optimized_prompts else self.optimized_prompts[role][1]
            initialize_prompt = PRE_RANK_INIT.format(initialize_num-1, self.example_prompt)
        if role == 'structure':
            self.structure_format_prompt = STRUCTURE_FORMAT
            self.example_prompt = STRUCTURE_DEFAULT if role not in self.optimized_prompts else self.optimized_prompts[role][1]
            initialize_prompt = STRUCTURE_INIT.format(initialize_num-1, self.example_prompt)

        # logging.info('example prompt {}'.format(self.example_prompt))
        self.new_prompts.append(self.example_prompt)
        self.prompt2id[self.example_prompt] = 0
        self.id2prompt[0] = self.example_prompt
        # self.roles_list.append(self.base_roles)
        if role == 'construct-role':
            cs_role = "Assistant" if role not in self.optimized_prompts else self.optimized_prompts[role][0]
            self.role_names.append(cs_role)
        else:
            self.role_names.append(role)

        service = ChatService(initialize_prompt)
        for i in range(initialize_num-1):
            start_time = time.time()
            try:
                if self.role == 'construct-role':
                    question = """Directly output the role name of the {}th prompt without any prefix or explanation.""".format(i+1)
                    role_name = service.ask(question, self.model).rstrip('\n')
                    self.role_names.append(role_name)
                    time.sleep(1)
                    # self.roles_list.append(self.base_roles + [role_name])
                else:
                    role_name = self.role
                    self.role_names.append(role_name)
                question = """Directly output the {}th prompt without any prefix or explanation.""".format(i+1) # + "\n\n" + self.rank_format_prompt
                answer_1 = service.ask(question, self.model).rstrip('\n')  # type: str
                # print('content is:', initialize_prompt)
                # print('question is:', question)
                # print('anwer is:', init_prompt)
                rand_uuid = str(uuid.uuid4())
                init_prompt = "{{{}}}: ".format(rand_uuid) + answer_1
                logging.info('initialize prompt {}'.format(i+1))
                logging.info('prompt role name: {}'.format(role_name))
                logging.info('prompt content: {}'.format(init_prompt))
                logging.info('end output prompt')
                self.new_prompts.append(init_prompt)
                self.prompt2id[init_prompt] = i+1
                self.id2prompt[i+1] = init_prompt

                '''
                if self.role == 'pre-rank':
                    system_pt = construct_prerank_message(self.base_roles, self.optimized_prompts, answer_1)
                    service_2 = ChatService(system_pt)
                    question = """Directly output your choices of agent roles.""" + "\n" + self.prerank_format_prompt
                    try:
                        completion = service_2.ask(question, self.model).rstrip('\n')
                    except Exception as e:
                        print(e)
                    tops = parse_ranks(completion, max_num=len(self.base_roles), random_num=4)
                    logging.info('pre-rank answer: {}'.format(completion))
                    logging.info('tops: {}'.format(tops))
                    pre_rank_roles = [self.base_roles[i] for i in tops]
                    self.roles_list.append(pre_rank_roles)
                elif self.role != 'construct-role':
                    self.roles_list.append(self.base_roles)
                '''
            except Exception as e:
                print(e)

            interval = time.time() - start_time
            if interval <= MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - interval)
    
    def renew_prompts(self):
        self.prompts += self.new_prompts
        self.new_prompts = []

    def FD_parent_selection(self, FD_score_threshold):
        if len(self.prompts) < 2 * FD_score_threshold:
            score_threshold = int(len(self.prompts) / 2)
        else:
            score_threshold = FD_score_threshold
        # get two prompt ids, the first one is randomly choosen from the first score_threshold prompts with highest scores, the second one is randomly choosen from the first score_threshold prompts with the lowest score_threshold scores. The information is stored in the prompt.id2score dictionary
        sorted_id2score = sorted(self.id2score.items(), key=lambda x: x[1], reverse=True)
        print("sorted_id2score", sorted_id2score)
        positive_position = 0  # score_threshold
        negative_position = len(self.prompts) - 1 - np.random.randint(score_threshold)
        positive_prompt_id = sorted_id2score[positive_position][0]
        negative_prompt_id = sorted_id2score[negative_position][0]
        return [positive_prompt_id, negative_prompt_id]


    def FD_mutation(self, prompt_pair):
        positive_prompt_id, negative_prompt_id = prompt_pair

        if self.role == 'rank':
            FD_prompt = RANK_FD
        if self.role in ROLES:
            FD_prompt = ROLE_FD.format(self.role, self.example_prompt)
        if self.role == 'construct-role':
            FD_prompt = CONSTRUCT_ROLE_FD.format(self.role_names[positive_prompt_id], self.role_names[negative_prompt_id])
        if self.role == 'pre-rank':
            FD_prompt = PRE_RANK_FD
        if self.role == 'structure':
            FD_prompt = STRUCTURE_FD
        
        parent_examples = '\n\nThe positive parent prompt is: "{}"'.format(self.id2prompt[positive_prompt_id]) + '\n\nThe negative parent prompt is: "{}"'.format(self.id2prompt[negative_prompt_id])
        FD_prompt += parent_examples
        prompt_question = """Directly output the content of the child prompt without any prefix or explanation."""
        # content = [{"role": "system", "content": FD_prompt}, {"role": "user", "content": prompt_question}]
        service = ChatService(FD_prompt)

        try:
            if self.role == 'construct-role':
                role_name = service.ask("""Directly output the role name of the child prompt without any prefix or explanation.""", self.model).rstrip('\n')
                self.role_names.append(role_name)
                enhanced = service.ask(prompt_question, self.model).rstrip('\n')
                # self.roles_list.append(self.base_roles + [role_name])
            else:
                role_name = self.role
                self.role_names.append(role_name)
                enhanced = service.ask(prompt_question, self.model).rstrip('\n')
            logging.info('FD mutation prompt {}'.format(len(self.prompts)))
            logging.info('prompt role name: {}'.format(role_name))
            logging.info('prompt content: {}'.format(enhanced))
            logging.info('end output prompt')
            self.new_prompts.append(enhanced)
            index = len(self.prompts)
            self.prompt2id[enhanced] = index
            self.id2prompt[index] = enhanced

            '''
            if self.role == 'pre-rank':
                system_pt = construct_prerank_message(self.base_roles, self.optimized_prompts, enhanced)
                service_2 = ChatService(system_pt)
                question = """Directly output your choices of agent roles.""" + "\n" + self.prerank_format_prompt
                try:
                    completion = service_2.ask(question, self.model).rstrip('\n')
                except Exception as e:
                    print(e)
                tops = parse_ranks(completion, max_num=len(self.base_roles), random_num=4)
                logging.info('pre-rank answer: {}'.format(completion))
                logging.info('tops: {}'.format(tops))
                pre_rank_roles = [self.base_roles[i] for i in tops]
                self.roles_list.append(pre_rank_roles)
            elif self.role != 'construct-role':
                self.roles_list.append(self.base_roles)
            '''
        except Exception as e:
            print(e)
