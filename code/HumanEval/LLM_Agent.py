import math
import random
import json
from utils import parse_code_completion, generate_answer, parse_judge_attitude
from prompt_lib import *


class LLMAgent:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_code_completion, qtype="single_choice", prompt_info=None, optimized_prompts=None):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None
        self.prompt = ""

        self.resp_cost = 1
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

        if mtype == "gpt-3.5-turbo":
            self.model = "gpt-3.5-turbo-1106"  # Shijun: 0301 original version
        elif mtype == "gpt-4":
            self.model = "gpt4"
        else:
            raise NotImplementedError("Error init model type")
        
        self.role_map = ROLE_MAP
        self.role_map_init = ROLE_MAP_INIT
        self.judge_map = JUDGE_MAP
        prompt_role, role_name, optimizable_prompt = prompt_info
        
        self.change_role(prompt_role, role_name, optimizable_prompt)
        if len(optimized_prompts) > 0:
            for rl in optimized_prompts:
                if rl != prompt_role or rl == "construct-role":
                    rl_name, rl_pompot = optimized_prompts[rl]
                    self.change_role(rl, rl_name, rl_pompot)

        if role in self.role_map:
            self.agent_description = ROLE_MAP[self.role]
            self.prompt = self.role_map[self.role]
        elif role in self.judge_map:  
            self.agent_description = JUDGE_MAP[self.role]
            self.prompt = self.judge_map[self.role]
        else:
            self.agent_description = optimizable_prompt
            self.prompt = optimizable_prompt
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

        with open('agent_config.json', 'r') as f:
            config = json.load(f)
        self.type = None
        for k, v in config['type_to_agents'].items():
            if self.role in v:
                self.type = k
                break

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer
    
    def get_unit_tests(self):
        return self.unit_tests

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

    def activate(self, question):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # print("formers: ", formers)
        # shuffle
        random.shuffle(formers)
        formers = [mess[0] for mess in formers]
        # print("shuffled: ", shuffled_idxs)

        if self.role == "Passer":
            self.reply = formers
            self.answer = self.ans_parser(self.reply, question, self.role, formers)
        elif self.role == "Ranker" and len(formers) <= 2:
            self.reply = "[1, 2]"
            self.answer = self.ans_parser(self.reply, question, self.role, formers)
        elif self.role in self.judge_map:
            contexts.append(construct_judge_message(formers, question, self.qtype, self.role))
            # print(contexts)
            self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
            self.answer = self.ans_parser(self.reply, question, self.role, formers)
        else:
            contexts.append(construct_message(formers, question, self.qtype))
            # print(contexts)
            self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
            self.answer = self.ans_parser(self.reply, question)

        if self.role == "Tester":
            self.answer, self.unit_tests = self.answer
        
    def get_context(self):
        if self.qtype == "code_completion":
            if self.role in list(self.role_map.keys()) + ['construct-role']:
                if len(self.from_edges) == 0:
                    sys_prompt = self.role_map_init[self.role]
                else:
                    sys_prompt = self.prompt
                formers = [(edge.a1, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
            else:
                sys_prompt = self.prompt
                formers = [(edge.a1.answer, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        else:
            raise NotImplementedError("Error init question type")
        contexts = [{"role": "system", "content": sys_prompt}]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []
        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts
    
    def change_role(self, prompt_role, role_name, optimizable_prompt):
        if prompt_role in ["PythonAssistant", "AlgorithmDeveloper", "ComputerScientist", "Programmer", "CodingArtist", "SoftwareArchitect", "construct-role"]:
            self.role_map[role_name] = optimizable_prompt
        if prompt_role in ["construct-role"]:
            self.role_map_init[role_name] = optimizable_prompt + CONSTRUCT_ROLE_INIT_APPENDIX
        if prompt_role in ["Passer", "Tester", "Reflector", "Ranker"]:
            self.judge_map[role_name] = optimizable_prompt
        return None


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0
