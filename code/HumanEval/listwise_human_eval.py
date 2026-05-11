import json
import os
import logging
import random
import sys
from CoLLMLP import CoLLMLP
from utils import *
from prompt_lib import *
from prompt_iteration.chat_service import ChatService


openai.api_key = os.getenv("OPENAI_API_KEY", openai.api_key)
# openai.api_base =
# openai.api_type =
# openai.api_version =

ACTIVATION = "listwise"
TYPE = "code_completion"

SUBSET = 40

def set_rd_seed(seed):
    random.seed(seed)

def listwise(PART, EXP_NAME, MODEL, DIR_NAME, ROLES, JUDGES, PROMPT_INFO, optimized_prompts):
    set_rd_seed(0)
    assert len(ROLES) > 0
    assert len(JUDGES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    llmlp = CoLLMLP(MODEL, len(ROLES), ROLES, len(JUDGES), JUDGES, 3, ACTIVATION, TYPE, MODEL, PROMPT_INFO, optimized_prompts)
    qa_pairs = get_human_eval_qa_pairs()

    with open(DIR_NAME+'/'+EXP_NAME+'.json', 'w') as f:
        f.write("")
    with open(DIR_NAME+'/'+EXP_NAME+'.tests', 'w') as f:
        f.write("")

    results, resp_cnts, importances = [], 0, []
    total_prompt_tokens, total_completion_tokens = 0, 0

    print('Start forward and backward')
    for task_id, que, entry_point in qa_pairs:
        qid = int(task_id.split("/")[-1])
        if qid < (PART)*SUBSET or qid >= (PART+1)*SUBSET:   # Shijun: change for a shoter experimental time
            continue
        
        if PROMPT_INFO[0] == 'pre-rank':
            system_pt = construct_prerank_message(ROLES, optimized_prompts, PROMPT_INFO[2], que)
            service_2 = ChatService(system_pt)
            # logging.info('pre-rank question: {}'.format(PROMPT_INFO[2]))
            question = """Directly output your choices of agents.""" + "\n" + PRE_RANK_FORMAT
            try:
                completion = service_2.ask(question, MODEL).rstrip('\n')
            except Exception as e:
                print(e)
            tops = parse_ranks(completion, max_num=len(ROLES), random_num=4)
            # logging.info('pre-rank answer: {}'.format(completion))
            # logging.info('tops: {}'.format(tops))
            pre_rank_roles = [ROLES[i] for i in tops]
            llmlp.agent_roles = pre_rank_roles
            llmlp.agents = len(pre_rank_roles)
            llmlp.init_nn(0, llmlp.agent_roles, llmlp.judge_roles)

        res, resp_cnt, completions, prompt_tokens, completion_tokens, tests = llmlp.forward(que, entry_point)

        results.append({"task_id": task_id, "completion": res})
        resp_cnts += resp_cnt
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        with open(DIR_NAME+'/'+EXP_NAME+'.json', 'a') as f:
            f.write(json.dumps(completions) + '\n')
        with open(DIR_NAME+'/'+EXP_NAME+'.tests', 'a') as f:
            f.write(json.dumps(tests) + '\n')
    print('End forward')

    print(results)
    print(resp_cnts)
    print(total_prompt_tokens, total_completion_tokens)

    print('Start write txt and jsonl')
    with open(DIR_NAME+'/'+EXP_NAME+'.txt', 'w') as f:
        # f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
        # f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + " " + str(total_completion_tokens) + '\n')
    
    write_jsonl(DIR_NAME+'/'+EXP_NAME+'.jsonl', results, True)
    return None
        
        
                
