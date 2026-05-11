import json
import random
import re

from utils import parse_single_choice, generate_answer
from prompt_lib import (
    ROLE_MAP,
    ROLE_MAP_MATH,
    SYSTEM_PROMPT_MMLU,
    SYSTEM_PROMPT_MATH,
    RANK_DEFAULT,
    construct_message,
    construct_ranking_message,
)


class LLMAgent:
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_single_choice,
                 qtype="single_choice", prompt_info=None, optimized_prompts=None):
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
        self.prompt_tokens = 0
        self.completion_tokens = 0

        if mtype == "gpt-3.5-turbo":
            self.model = "gpt-3.5-turbo-1106"
        elif mtype == "gpt-4":
            self.model = "gpt4"
        else:
            raise NotImplementedError("Error init model type")

        self.role_map = ROLE_MAP
        self.math_role_map = ROLE_MAP_MATH
        prompt_role, role_name, optimizable_prompt = prompt_info
        self.change_role(prompt_role, role_name, optimizable_prompt)
        for optimized_role, optimized_value in optimized_prompts.items():
            if optimized_role != prompt_role or optimized_role == "construct-role":
                optimized_name, optimized_prompt = optimized_value
                self.change_role(optimized_role, optimized_name, optimized_prompt)

        if qtype == "math_exp":
            self.agent_description = self.math_role_map.get(role, optimizable_prompt)
            self.prompt = self.math_role_map.get(role, optimizable_prompt)
        else:
            self.agent_description = self.role_map.get(role, optimizable_prompt)
            self.prompt = self.role_map.get(role, optimizable_prompt)

        with open('agent_config.json', 'r') as f:
            config = json.load(f)
        self.type = None
        for agent_type, agents in config['type_to_agents'].items():
            if self.role in agents:
                self.type = agent_type
                break

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def activate(self, question):
        self.question = question
        self.active = True
        contexts, formers = self.get_context()
        random.shuffle(formers)
        former_replies = [former[0] for former in formers]
        contexts.append(construct_message(former_replies, question, self.qtype))
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
        self.answer = self.ans_parser(self.reply)

    def get_context(self):
        if self.qtype == "single_choice":
            sys_prompt = self.prompt + "\n" + SYSTEM_PROMPT_MMLU
        elif self.qtype == "math_exp":
            sys_prompt = self.prompt + "\n" + SYSTEM_PROMPT_MATH
        else:
            raise NotImplementedError("Error init question type")
        contexts = [{"role": "system", "content": sys_prompt}]
        formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges)
                   if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers

    def get_conversation(self):
        if not self.active:
            return []
        contexts, formers = self.get_context()
        contexts.append(construct_message([former[0] for former in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts

    def change_role(self, prompt_role, role_name, optimizable_prompt):
        if prompt_role in self.role_map or prompt_role == 'construct-role':
            self.role_map[role_name] = optimizable_prompt
            self.math_role_map[role_name] = optimizable_prompt


class LLMEdge:
    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0


def parse_ranks(completion, max_num=7, random_num=6):
    content = completion
    pattern = r'\[([1-9])(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?\]'
    matches = [tuple(filter(None, match)) for match in re.findall(pattern, content)]

    try:
        match = matches[-1]
        tops = [int(match[i]) - 1 for i in range(len(match))]

        def clip(x):
            if x < 0:
                return 0
            if x > max_num - 1:
                return max_num - 1
            return x

        tops = [clip(x) for x in tops]
    except Exception:
        tops = random.sample(list(range(max_num)), min(random_num, max_num))
    return tops


def listwise_ranker_2(responses, question, qtype, model="gpt-3.5-turbo",
                      rank_prompt=RANK_DEFAULT, max_agents=7):
    if model == "gpt-3.5-turbo":
        model = "gpt-3.5-turbo-1106"
    elif model == "gpt-4":
        model = "gpt4"
    else:
        raise NotImplementedError("Error init model type")
    if rank_prompt is None:
        rank_prompt = RANK_DEFAULT
    message = construct_ranking_message(responses, question, qtype, rank_prompt, max_agents)
    completion, prompt_tokens, completion_tokens = generate_answer([message], model)
    return parse_ranks(completion, max_num=len(responses)), prompt_tokens, completion_tokens
