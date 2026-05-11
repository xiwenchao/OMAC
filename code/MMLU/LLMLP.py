import json
import math
import random

from LLM_Agent import LLMAgent, LLMEdge, listwise_ranker_2, parse_ranks
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer
from prompt_lib import *
from prompt_iteration.chat_service import ChatService


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}


class LLMLP:
    def __init__(self, default_model_name, agents=4, agent_roles=None,
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo",
                 prompt_info=None, optimized_prompts=None):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        self.prompt_info = prompt_info
        self.optimized_prompts = optimized_prompts or {}

        prompt_role, role_name, opt_prompt = prompt_info
        if prompt_role == "rank":
            self.rank_prompt = opt_prompt + RANK_FORMAT
        elif "rank" in self.optimized_prompts:
            self.rank_prompt = self.optimized_prompts["rank"][1] + RANK_FORMAT
        else:
            self.rank_prompt = RANK_DEFAULT + RANK_FORMAT

        agent_roles = agent_roles or []
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer
        else:
            raise NotImplementedError("Error init qtype")

        with open('agent_config.json', 'r') as f:
            config = json.load(f)
        self.type_to_agents = config['type_to_agents']
        self.layer_type_list = config['layer_type_list']
        self.layer_max_agents = config['layer_max_agents']
        self.default_agent_type = self.layer_type_list[0]
        if prompt_role == 'construct-role':
            self.type_to_agents[self.default_agent_type].append(role_name)
        if 'construct-role' in self.optimized_prompts:
            self.type_to_agents[self.default_agent_type].append(self.optimized_prompts['construct-role'][0])

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges, self.num_agents = [], [], []
        previous_layer = []
        for layer_type in self.layer_type_list[:self.rounds]:
            current_layer = []
            for role in agent_roles:
                if role in self.type_to_agents[layer_type]:
                    node = LLMAgent(role, self.mtype, self.ans_parser, self.qtype,
                                    self.prompt_info, self.optimized_prompts)
                    self.nodes.append(node)
                    current_layer.append(node)
                    if previous_layer:
                        self.connect_agents(previous_layer, node)
            self.num_agents.append(len(current_layer))
            previous_layer = current_layer

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")

    def layer_start(self, layer_idx):
        return sum(self.num_agents[:layer_idx])

    def layer_end(self, layer_idx):
        return sum(self.num_agents[:layer_idx + 1])

    def get_structure_prompt(self):
        if self.prompt_info[0] == 'structure':
            return self.prompt_info[2]
        if 'structure' in self.optimized_prompts:
            return self.optimized_prompts['structure'][1]
        return None

    def connect_agents(self, previous_agents, target_agent):
        structure_prompt = self.get_structure_prompt()
        if structure_prompt:
            candidate_roles = [agent.role for agent in previous_agents]
            selected_roles = self.get_structure(target_agent.role, candidate_roles, structure_prompt, self.optimized_prompts)
        else:
            selected_roles = [agent.role for agent in previous_agents]
        for previous_agent in previous_agents:
            if previous_agent.role in selected_roles:
                self.edges.append(LLMEdge(previous_agent, target_agent))

    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2 / 3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def rank_agents(self, source_layer_idx, max_agents, question, resp_cnt,
                    total_prompt_tokens, total_completion_tokens):
        idxs = list(range(self.layer_start(source_layer_idx), self.layer_end(source_layer_idx)))
        replies = [self.nodes[idx].get_reply() for idx in idxs]
        indices = list(range(len(replies)))
        random.shuffle(indices)
        shuffled_replies = [replies[idx] for idx in indices]
        tops, prompt_tokens, completion_tokens = self.activation(
            shuffled_replies, question, self.qtype, self.mtype, self.rank_prompt, max_agents
        )
        if len(tops) > max_agents:
            tops = tops[:max_agents]
        idx_mask = [indices[top] for top in tops if top < len(indices)]
        resp_cnt += self.activation_cost
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        return idx_mask, resp_cnt, total_prompt_tokens, total_completion_tokens

    def forward(self, question):
        def get_completions():
            completions = [[] for _ in range(len(self.agent_roles))]
            for layer_idx in range(self.rounds):
                for pos, node_idx in enumerate(range(self.layer_start(layer_idx), self.layer_end(layer_idx))):
                    if self.nodes[node_idx].active:
                        completions[pos].append(self.nodes[node_idx].get_reply())
                    else:
                        completions[pos].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2

        last_active = []
        for layer_idx in range(self.rounds):
            layer_size = self.num_agents[layer_idx]
            loop_indices = list(range(self.layer_start(layer_idx), self.layer_end(layer_idx)))
            random.shuffle(loop_indices)

            max_agents = self.layer_max_agents[layer_idx]
            if layer_idx >= 2 and max_agents > 0 and max_agents < layer_size:
                idx_mask, resp_cnt, total_prompt_tokens, total_completion_tokens = self.rank_agents(
                    layer_idx - 1, max_agents, question, resp_cnt, total_prompt_tokens, total_completion_tokens
                )
            else:
                idx_mask = list(range(layer_size))
                if max_agents > 0 and max_agents < layer_size:
                    idx_mask = random.sample(idx_mask, max_agents)

            active_indices = []
            layer_start = self.layer_start(layer_idx)
            for pos, node_idx in enumerate(loop_indices):
                if node_idx - layer_start not in idx_mask:
                    continue
                self.nodes[node_idx].activate(question)
                resp_cnt += 1
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                active_indices.append(node_idx)
                last_active = active_indices
                if len(active_indices) > math.floor(2 / 3 * len(idx_mask)):
                    reached, reply = self.check_consensus(active_indices, idx_mask)
                    if reached:
                        return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        return most_frequent([self.nodes[idx].get_answer() for idx in last_active], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def backward(self, result):
        flag_last = False
        for layer_idx in range(self.rounds - 1, -1, -1):
            layer_indices = list(range(self.layer_start(layer_idx), self.layer_end(layer_idx)))
            if not flag_last:
                if len([idx for idx in layer_indices if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue

                candidates = [
                    idx for idx in layer_indices
                    if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result)
                ]
                ave_w = 1 / len(candidates) if candidates else 0
                for idx in layer_indices:
                    self.nodes[idx].importance = ave_w if idx in candidates else 0
            else:
                for idx in layer_indices:
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]

    def get_structure(self, current_agent, candidate_agents, structure_prompt, optimized_prompts):
        if len(candidate_agents) == 0:
            return []
        system_pt = construct_structure_message(current_agent, candidate_agents, structure_prompt, optimized_prompts)
        service_2 = ChatService(system_pt)
        question = """Directly output your choices of candidate agents.""" + "\n" + STRUCTURE_FORMAT
        try:
            completion = service_2.ask(question, self.mtype).rstrip('\n')
        except Exception as e:
            print(e)
            completion = ""
        tops = parse_ranks(completion, max_num=len(candidate_agents), random_num=min(2, len(candidate_agents)))
        if len(tops) < min(2, len(candidate_agents)):
            remaining = list(set(range(len(candidate_agents))) - set(tops))
            if remaining:
                tops.append(random.choice(remaining))
        return [candidate_agents[i] for i in tops]
