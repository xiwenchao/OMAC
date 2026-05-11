import ast
import math
import random
import json
from LLM_Agent import LLMAgent, LLMEdge
from utils import check_function_result, parse_single_choice, parse_code_completion, most_frequent, parse_judge_attitude, listwise_ranker_2, parse_ranks
from sacrebleu import sentence_bleu
from prompt_lib import *
from prompt_iteration.chat_service import ChatService


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented
CODE_THRESHOLD = 0.9

Examples_dict = {"Tester": EXAMPLE_TESTER, "Reflector": EXAMPLE_REFLECTOR, "Debugger": EXAMPLE_DEBUGGER}

class CoLLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[], judges=4, judge_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo", prompt_info=None, optimized_prompts=None):
        self.default_model_name = default_model_name
        self.agents = agents
        self.judges = judges
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        prompt_role, role_name, opt_prompt = prompt_info

        if prompt_role in ["Tester", "Reflector", "Debugger"]:
            opt_prompt = opt_prompt + "Here is the example.\n" + Examples_dict[prompt_role]
        for rl in ["Tester", "Reflector", "Debugger"]:
            if rl in optimized_prompts:
                optimized_prompts[rl] = (optimized_prompts[rl][0], optimized_prompts[rl][1] + "Here is the example.\n" + Examples_dict[rl])

        self.prompt_info = prompt_role, role_name, opt_prompt

        if prompt_role == "rank":
            self.rank_prompt = opt_prompt + RANK_FORMAT
        elif "rank" in optimized_prompts:
            self.rank_prompt = optimized_prompts["rank"][1] + RANK_FORMAT
        else:
            self.rank_prompt = RANK_DEFAULT + RANK_FORMAT

        self.optimized_prompts = optimized_prompts
        
        assert len(agent_roles) == agents and agents > 0
        assert len(judge_roles) == judges and judges > 0
        self.agent_roles = agent_roles
        self.judge_roles = judge_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "code_completion":
            self.cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= CODE_THRESHOLD * 100
            self.ans_parser = parse_code_completion
        else:
            raise NotImplementedError("Error init qtype")

        with open('agent_config.json', 'r') as f:
            config = json.load(f)
        self.type_to_agents = config['type_to_agents']
        self.layer_type_list = config['layer_type_list']
        self.layer_max_agents = config['layer_max_agents']
        if prompt_role == 'construct-role':
            self.type_to_agents['Coder'].append(role_name)
        if 'construct-role' in self.optimized_prompts:
            self.type_to_agents['Coder'].append(self.optimized_prompts['construct-role'][0])
    
        self.init_nn(self.activation, self.agent_roles, self.judge_roles)

    def init_nn(self, activation, agent_roles, judge_roles):
        self.nodes, self.edges, self.num_agents = [], [], [0]
        for idx in range(self.agents):
            if agent_roles[idx] in self.type_to_agents[self.layer_type_list[0]]:
                self.nodes.append(LLMAgent(agent_roles[idx], self.mtype, self.ans_parser, self.qtype, self.prompt_info, self.optimized_prompts))
                self.num_agents[-1] += 1
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            current_judges = 0
            current_roles = 0
            for idx in range(self.judges):
                if judge_roles[idx] in self.type_to_agents[self.layer_type_list[2*rid-1]]:
                    current_judges += 1
                    self.nodes.append(LLMAgent(judge_roles[idx], self.mtype, parse_judge_attitude, self.qtype, self.prompt_info, self.optimized_prompts))
                    self.connect_agents(agents_last_round, self.nodes[-1])
            agents_last_round = self.nodes[-current_judges:]
            self.num_agents.append(current_judges)

            for idx in range(self.agents):
                if agent_roles[idx] in self.type_to_agents[self.layer_type_list[2*rid]]:
                    current_roles += 1
                    self.nodes.append(LLMAgent(agent_roles[idx], self.mtype, self.ans_parser, self.qtype, self.prompt_info, self.optimized_prompts))
                    self.connect_agents(agents_last_round, self.nodes[-1])
            agents_last_round = self.nodes[-current_roles:]
            self.num_agents.append(current_roles)

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")

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

    def cut_def_question(self, func_code, question, entry_point):
        def parse_imports(src_code):
            res = []
            for line in src_code.split("\n"):
                if "import" in line:
                    res.append(line)
            res = ["    " + line.strip() for line in res]
            return res
        import_lines = parse_imports(func_code)

        def extract_functions_with_body(source_code):
            # Parse the source code to an AST
            tree = ast.parse(source_code)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if the function is nested inside another function
                    # We can determine this by checking the ancestors of the node
                    parents = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    nesting_level = sum(1 for parent in parents if
                                        parent.lineno <= node.lineno and parent.end_lineno >= node.end_lineno)
                    
                    if nesting_level == 1:  # Only top-level functions
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        function_body = source_code.splitlines()[start_line:end_line]
                        functions.append("\n".join(function_body))
                    
            return functions
        try:
            funcs = extract_functions_with_body(func_code)
        except:
            funcs = [func_code]

        def extract_func_def(src_code):
            for line in src_code.split("\n"):
                if "def" in line and entry_point in line:
                    return line
            return ""
        que_func = extract_func_def(question)

        for fiid, func_ins_code in enumerate(funcs):
            if question in func_ins_code:
                func_ins_code = func_ins_code.split(question)[-1]
            elif question.strip() in func_ins_code:
                func_ins_code = func_ins_code.split(question.strip())[-1]
            elif que_func in func_ins_code:
                # remove the line before def
                res_lines = func_ins_code.split("\n")
                func_ins_code = ""
                in_func = False
                for line in res_lines:
                    if in_func:
                        func_ins_code += line + "\n"
                    if "def" in line:
                        in_func = True
            else:
                continue

            other_funcs = []
            for other_func in funcs[:fiid] + funcs[fiid+1:]:
                other_func = other_func.split("\n")
                other_func = other_func[:1] + import_lines + other_func[1:]
                other_func = "\n".join(other_func)
                other_funcs.append(other_func)
                        
            return "\n".join(import_lines) + "\n" + func_ins_code + "\n" + "\n".join(other_funcs)
        
        res_lines = func_code.split("\n")
        func_code = ""
        in_func = False
        for line in res_lines:
            if in_func:
                func_code += line + "\n"
            if "def" in line:
                in_func = True
        
        return "\n".join(import_lines) + "\n" + func_code

    def check_consensus(self, idxs, idx_mask, question, entry_point):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        python_codes = []
        backup = []
        for cand in candidates:
            result = check_function_result(cand)
            if result["passed"]:
                python_codes.append(cand)
            else:
                backup.append(cand)
        
        if len(python_codes) == 0:
            return False, None

        pred_answers = []
        for python_code in python_codes:
            python_code = self.cut_def_question(python_code, question, entry_point)
            pred_answers.append(python_code)

        consensus_answer, ca_cnt = most_frequent(pred_answers, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            # print("Consensus answer: {}".format(consensus_answer))
            return True, consensus_answer
        return False, None

    def get_final_result(self, idxs, question, entry_point):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        python_codes = []
        backup = []
        for cand in candidates:
            result = check_function_result(cand)
            if result["passed"]:
                python_codes.append(cand)
            else:
                backup.append(cand)
        
        if len(python_codes) == 0:
            for rid in range(self.rounds-2, -1, -1):
                for idx in range(self.agents):
                    if self.nodes[rid*(self.agents+self.judges)+idx].active:
                        result = check_function_result(self.nodes[rid*(self.agents+self.judges)+idx].get_answer())
                        if result["passed"]:
                            python_codes.append(self.nodes[rid*(self.agents+self.judges)+idx].get_answer())
                
                if len(python_codes) > 0:
                    break
        
        if len(python_codes) == 0:
            python_codes = backup
        
        final_res = random.choice(python_codes)
        final_res = self.cut_def_question(final_res, question, entry_point)

        # print("Final answer: {}".format(final_res))
        return final_res

    def all_tests_and_get_final_result(self, question, unit_tests, entry_point):
        candidates = [self.nodes[idx].get_answer() for idx in range(self.agents) if isinstance(self.nodes[idx], LLMAgent) and self.nodes[idx].active]
        candidates = [self.cut_def_question(cand, question, entry_point) for cand in candidates]
        python_codes = []
        for cand in candidates:
            passed_tests = []
            failed_tests = []
            for test in unit_tests:
                result = check_function_result(question + "\n" + cand + "\n" + test)
                if result["passed"]:
                    passed_tests.append(test)
                else:
                    failed_tests.append(test)
            python_codes.append((cand, len(passed_tests), passed_tests, failed_tests))
        
        # Sort the codes based on the number of passed tests in descending order
        sorted_codes = sorted(python_codes, key=lambda x: x[1], reverse=True)
        
        # Get the maximum number of passed tests
        max_passed_tests = sorted_codes[0][1]
        
        # Filter the codes that have the maximum number of passed tests
        top_codes = [code for code in sorted_codes if code[1] == max_passed_tests]
        if len(top_codes) < 5:
            top_codes = [code for code in sorted_codes if code[1] == max_passed_tests or code[1] == max_passed_tests - 1]
        
        # Randomly select one of the top codes
        selected_code = random.choice(top_codes)
        
        return selected_code[0]  # Return the code part of the tuple

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def rank_agents(self, round_idx, max_agents, question, resp_cnt, total_prompt_tokens, total_completion_tokens):
        idxs = list(range(sum(self.num_agents[:round_idx]), sum(self.num_agents[:round_idx+1])))
        replies = [self.nodes[idx].get_answer() for idx in idxs]
        indices = list(range(len(replies)))
        random.shuffle(indices)
        shuffled_replies = [replies[idx] for idx in indices]
        tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype, self.rank_prompt, max_agents)
        if len(tops) > max_agents:
            tops = tops[:max_agents]
        idx_mask = list(map(lambda x: idxs[indices[x]] % (self.agents+self.judges), tops))
        resp_cnt += self.activation_cost
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        return idx_mask, resp_cnt, total_prompt_tokens, total_completion_tokens

    def forward(self, question, entry_point):
        def get_completions():
            # get completions
            completions = [[] for _ in range(self.agents+self.judges)]
            for rid in range(self.rounds):
                for idx in range(sum(self.num_agents[:rid*2]), sum(self.num_agents[:rid*2+2])):  # sum(self.num_agents[idx:idx+2])
                    if idx < sum(self.num_agents[:rid*2+1]) and self.nodes[idx].active:
                        completions[idx % (self.agents+self.judges)].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % (self.agents+self.judges)].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        unit_tests = []
        self.set_allnodes_deactivated()
        assert self.rounds > 2
        # question = format_question(question, self.qtype)
        
        # shuffle the order of agents
        loop_indices = list(range(self.num_agents[0]))
        random.shuffle(loop_indices)

        if self.layer_max_agents[0] > 0 and self.layer_max_agents[0] < self.num_agents[0]:
            loop_indices = random.sample(loop_indices, self.layer_max_agents[0])

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            # print(0, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
        loop_indices = list(range(sum(self.num_agents[:1]), sum(self.num_agents[:2])))
        random.shuffle(loop_indices)
        if self.layer_max_agents[1] > 0 and self.layer_max_agents[1] < self.num_agents[1]:
            loop_indices = random.sample(loop_indices, self.layer_max_agents[1])

        for idx, node_idx in enumerate(loop_indices):
            # print(0.5, idx)
            self.nodes[node_idx].activate(question)
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            resp_cnt += self.nodes[node_idx].resp_cost

            if self.nodes[node_idx].role == "Tester":
                unit_tests.extend(self.nodes[node_idx].get_unit_tests())

        loop_indices = list(range(sum(self.num_agents[:2]), sum(self.num_agents[:3])))
        random.shuffle(loop_indices)

        idx_mask = list(range(self.num_agents[0]))
        if self.layer_max_agents[2] > 0 and self.layer_max_agents[2] < self.num_agents[2]:
            idx_mask, resp_cnt, total_prompt_tokens, total_completion_tokens = self.rank_agents(
                0, self.layer_max_agents[2], question, resp_cnt, total_prompt_tokens, total_completion_tokens
                )
        activated_indices = []
        layer_start = sum(self.num_agents[:2])
        for idx, node_idx in enumerate(loop_indices):
            if node_idx - layer_start in idx_mask:
                # print(rid, idx)
                self.nodes[node_idx].activate(question)
                resp_cnt += 1
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                activated_indices.append(node_idx)
                if len(activated_indices) >= math.floor(2/3 * len(idx_mask)):
                    reached, reply = self.check_consensus(activated_indices, idx_mask, question, entry_point)
                    if reached:
                        # return reply, resp_cnt, get_completions(), unit_tests
                        return self.all_tests_and_get_final_result(question, unit_tests, entry_point), resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens, unit_tests

        for rid in range(2, self.rounds):
            loop_indices = list(range(sum(self.num_agents[:2*rid-1]), sum(self.num_agents[:2*rid])))
            if self.layer_max_agents[2*rid-1] > 0 and self.layer_max_agents[2*rid-1] < self.num_agents[2*rid-1]:
                loop_indices = random.sample(loop_indices, self.layer_max_agents[2*rid-1])

            for idx, node_idx in enumerate(loop_indices):
                # print(rid-0.5, idx)
                self.nodes[node_idx].activate(question)
                total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                total_completion_tokens += self.nodes[node_idx].completion_tokens
                resp_cnt += self.nodes[node_idx].resp_cost

                if self.nodes[node_idx].role == "Tester":
                    unit_tests.extend(self.nodes[node_idx].get_unit_tests())

            idx_mask = list(range(self.num_agents[0]))
            if self.layer_max_agents[2*rid] > 0 and self.layer_max_agents[2*rid] < self.num_agents[2*rid]:
                idx_mask, resp_cnt, total_prompt_tokens, total_completion_tokens = self.rank_agents(
                    2*rid-2, self.layer_max_agents[2*rid], question, resp_cnt, total_prompt_tokens, total_completion_tokens
                    )
            loop_indices = list(range(sum(self.num_agents[:2*rid]), sum(self.num_agents[:2*rid+1])))
            random.shuffle(loop_indices)
            idxs = []
            layer_start = sum(self.num_agents[:2*rid])
            for idx, node_idx in enumerate(loop_indices):
                if node_idx - layer_start in idx_mask:
                    # print(rid, idx)
                    if node_idx >= len(self.nodes):
                        print('len_nodes: ', len(self.nodes))
                        print('num_agents: ', self.num_agents)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask, question, entry_point)
                        if reached:
                            # return reply, resp_cnt, get_completions(), unit_tests
                            return self.all_tests_and_get_final_result(question, unit_tests, entry_point), resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens, unit_tests
        
        completions = get_completions()
        # return self.get_final_result(idxs, question), resp_cnt, completions, total_prompt_tokens, total_completion_tokens, unit_tests
        return self.all_tests_and_get_final_result(question, unit_tests, entry_point), resp_cnt, completions, total_prompt_tokens, total_completion_tokens, unit_tests

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
