import os
import time
import uuid

import numpy as np
import json
import argparse
import datetime
import logging
from prompt_iteration.chat_service import ChatService
# from utils import generate_answer
from prompt_lib import *
# from pro_utils import load_json, save_json

MIN_INTERVAL = 0


class Prompt():
    def __init__(self, evol_model, dataset_name):
        self.model = evol_model
        self.data_set = dataset_name
        self.optimized_prompts = dict()  # role: [role_name, optimized_prompt]

    def initialize(self, role, initialize_num, roles):
        self.prompts = []
        self.new_prompts = []
        self.roles_list = []
        self.prompt2id = dict()
        self.id2prompt = dict()
        self.id2score = dict()
        self.role = role
        self.base_roles = roles # if 'construct' not in self.optimized_prompts else roles + ['construct']
        
        if role == 'rank':
            self.rank_format_prompt = RANK_FORMAT
            self.example_prompt = [RANK_DEFAULT] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [RANK_INIT.format(initialize_num-1, self.example_prompt)] # + "\n\n" + self.rank_format_prompt
        if role in ["System"]:
            self.example_prompt = [SYSTEM_PROMPT] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [SYSTEM_INIT.format(initialize_num-1, self.example_prompt)]
        if role in ["Examples"]:
            self.example_prompt = [EXAMPLES_ORIG] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [EXAMPLES_INIT.format(initialize_num-1, self.example_prompt)]
        if role in ["Examples_cot"]:
            self.example_prompt = [EXAMPLES_COT] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [EXAMPLES_INIT.format(initialize_num-1, self.example_prompt)]
        if role == 'construct-role':
            self.example_prompt = [SYSTEM_PROMPT, EXAMPLES_ORIG] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [SYSTEM_INIT.format(initialize_num-1, self.example_prompt[0]), EXAMPLES_INIT.format(initialize_num-1, self.example_prompt[1])]
        if role == 'pre-rank':
            self.prerank_format_prompt = PRE_RANK_FORMAT
            self.example_prompt = [PRE_RANK_DEFAULT] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [PRE_RANK_INIT.format(initialize_num-1, self.example_prompt)]
        if role == 'structure':
            self.structure_format_prompt = STRUCTURE_FORMAT
            self.example_prompt = [STRUCTURE_DEFAULT] if role not in self.optimized_prompts else self.optimized_prompts[role]
            initialize_prompts = [STRUCTURE_INIT.format(initialize_num-1, self.example_prompt)]

        self.new_prompts.append(self.example_prompt)
        self.prompt2id['|sys_exa|'.join(self.example_prompt)] = 0
        self.id2prompt[0] = self.example_prompt

        init_prompts = []
        for j, initialize_prompt in enumerate(initialize_prompts):
            service = ChatService(initialize_prompt)
            for i in range(initialize_num-1):
                start_time = time.time()
                try:
                    rand_uuid = str(uuid.uuid4())
                    if self.role in ['System', 'rank', 'pre-rank', 'structure'] or (self.role == 'construct-role' and j == 0):
                        question = """Directly output the {}th prompt without any prefix or explanation.""".format(i+1)
                        init_prompt = service.ask(question, self.model).rstrip('\n')
                        init_prompt = "{{{}}}: ".format(rand_uuid) + init_prompt
                        time.sleep(1)
                        init_prompts.append(init_prompt)
                    if self.role in ['Examples', 'Examples_cot'] or (self.role == 'construct-role' and j == 1):
                        question = """Directly output the {}th example set without any prefix or explanation.""".format(i+1)
                        init_prompt = service.ask(question, self.model).rstrip('\n')  # type: str
                        init_prompt = "{{{}}}: ".format(rand_uuid) + init_prompt
                        init_prompts.append(init_prompt)
                except Exception as e:
                    print(e)

                interval = time.time() - start_time
                if interval <= MIN_INTERVAL:
                    time.sleep(MIN_INTERVAL - interval)

                '''
                if self.role == 'pre-rank':
                    system_pt = construct_prerank_message(self.base_roles, self.optimized_prompts, init_prompt.rstrip('\n'))
                    service_2 = ChatService(system_pt)
                    question = """Directly output your choice of the number of agents.""" + "\n" + self.prerank_format_prompt
                    try:
                        completion = service_2.ask(question, self.model).rstrip('\n')
                    except Exception as e:
                        print(e)
                    top = parse_ranks(completion, max_num=4, random_num=3)
                    logging.info('pre-rank answer: {}'.format(completion))
                    logging.info('tops: {}'.format(top))
                    pre_rank_roles = self.base_roles[:top] if len(self.base_roles) <= 4 else self.base_roles[:top] + self.base_roles[4:]
                    self.roles_list.append(pre_rank_roles)
                elif self.role == 'construct':
                    self.roles_list.append(self.base_roles + ['construct'])
                else:
                    self.roles_list.append(self.base_roles)
                '''
        
        for i in range(initialize_num-1):
            if role == 'construct-role':
                system_index = i
                examples_index = i + initialize_num - 1
                if examples_index >= len(init_prompts):
                    continue
                init_prompt = [init_prompts[system_index], init_prompts[examples_index]]
            else:
                if i >= len(init_prompts):
                    continue
                init_prompt = [init_prompts[i]]
            logging.info('initialize prompt {}'.format(i+1))
            # logging.info('prompt role name: {}'.format(role_name))
            logging.info('prompt content: {}'.format(init_prompt[0]))
            logging.info('end output prompt')
            self.new_prompts.append(init_prompt)
            self.prompt2id['|sys_exa|'.join(init_prompt)] = i+1
            self.id2prompt[i+1] = init_prompt
    
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
        positive_position = np.random.randint(score_threshold)
        negative_position = len(self.prompts) - 1 - np.random.randint(score_threshold)
        positive_prompt_id = sorted_id2score[positive_position][0]
        negative_prompt_id = sorted_id2score[negative_position][0]
        return [positive_prompt_id, negative_prompt_id]


    def FD_mutation(self, prompt_pair):
        positive_prompt_id, negative_prompt_id = prompt_pair

        target_name = ['prompt']
        if self.role == 'rank':
            FD_prompt = [RANK_FD]
        if self.role in ["System"]:
            FD_prompt = [SYSTEM_FD]
        if self.role in ['Examples', 'Examples_cot']:
            FD_prompt = [EXAMPLES_FD]
            target_name = ['example set']
        if self.role == 'construct-role':
            FD_prompt = [SYSTEM_FD, EXAMPLES_FD]
            target_name = ['prompt', 'example set']
        if self.role == 'pre-rank':
            FD_prompt = [PRE_RANK_FD]
        if self.role == 'structure':
            FD_prompt = [STRUCTURE_FD]
        
        anwer_pt = []
        for j, fd_pt in enumerate(FD_prompt):
            parent_examples = f'\n\nThe positive parent {target_name[j]} is: "{self.id2prompt[positive_prompt_id][j]}"' + f'\n\nThe negative parent {target_name[j]} is: "{self.id2prompt[negative_prompt_id][j]}"'
            fd_pt = fd_pt + parent_examples
            prompt_question = f"""Directly output the content of the child {target_name[j]} without any prefix or explanation."""
            # content = [{"role": "system", "content": FD_prompt}, {"role": "user", "content": prompt_question}]
            service = ChatService(fd_pt)

            try:
                enhanced = service.ask(prompt_question, self.model).rstrip('\n')
                anwer_pt.append(enhanced)
            except Exception as e:
                print(e)
    
        if self.role == 'construct-role':
            final_pt = [anwer_pt[0], anwer_pt[1]]
        else:
            final_pt = [anwer_pt[0]]
        
        logging.info('FD mutation prompt {}'.format(len(self.prompts)))
        logging.info('prompt role name: {}'.format(self.role))
        logging.info('prompt content: {}'.format(final_pt[-1]))
        logging.info('end output prompt')
        self.new_prompts.append(final_pt)
        index = len(self.prompts)
        self.prompt2id['|sys_exa|'.join(final_pt)] = index
        self.id2prompt[index] = final_pt

        '''
        if self.role == 'pre-rank':
            system_pt = construct_prerank_message(self.base_roles, self.optimized_prompts, enhanced)
            service_2 = ChatService(system_pt)
            question = """Directly output your choice of the number of agents.""" + "\n" + self.prerank_format_prompt
            try:
                completion = service_2.ask(question, self.model).rstrip('\n')
            except Exception as e:
                print(e)
            tops = parse_ranks(completion, max_num=len(self.base_roles), random_num=4)
            logging.info('pre-rank answer: {}'.format(completion))
            logging.info('tops: {}'.format(tops))
            pre_rank_roles = self.base_roles[:tops] if len(self.base_roles <= 4) else self.base_roles[:tops] + self.base_roles[4:]
            self.roles_list.append(pre_rank_roles)
        elif self.role == 'construct-role':
            self.roles_list.append(self.base_roles + ['construct'])
        else:
            self.roles_list.append(self.base_roles)
        '''
