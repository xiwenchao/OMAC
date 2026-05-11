import ast
import json
import os
import openai
import random
import uuid
from LLMLP import LLMLP
from utils import *
from LLM_Agent import parse_ranks

import glob
import subprocess
import logging
from prompt_iteration.chat_service import ChatService
from prompt_lib import *


# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

ACTIVATION = "listwise"
TYPE = "single_choice"

def set_rd_seed(seed):
    random.seed(seed)

def listwise(EXP_NAME, MODEL, DIR_NAME, stage, roles, prompt_info, optimized_prompts):  # (EXP_NAME, MODEL, DIR_NAME_TRAIN, 'train', roles, prompt_info, optimized_prompts)

    set_rd_seed(0)
    assert len(roles) > 0

    # Get all CSV files in the directory
    if stage == 'train':
        directory = "../../data/mmlu/data/downsampled/train"
    elif stage == 'test':
        directory = "../../data/mmlu/data/downsampled/test"
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    llmlp = LLMLP(MODEL, len(roles), roles, 3, "listwise", "single_choice", MODEL, prompt_info, optimized_prompts)
    for csv_file in csv_files:
        # Extract filename without extension
        filename = os.path.basename(csv_file)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Define paths for results and logs
        result_file = f"{DIR_NAME}/{EXP_NAME}/{filename_without_ext}.txt"
        # log_file = f"{DIR_NAME}/{EXP_NAME}.log"
        
        # Check if result file exists and has exactly 4 lines
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                if len(f.readlines()) == 4:
                    continue  # Skip this file

        qa_pairs = get_mmlu_qa_pairs(csv_file)

        with open(DIR_NAME+'/'+EXP_NAME+'/'+filename_without_ext+'.json', 'w') as f:
            f.write("")

        accs, resp_cnts, importances = [], 0, []
        completion_list = []
        total_prompt_tokens, total_completion_tokens = 0, 0

        for que, ans in qa_pairs:
            llmlp.zero_grad()
            llmlp.set_allnodes_deactivated()
            if prompt_info[0] == 'pre-rank':
                system_pt = construct_prerank_message(roles, optimized_prompts, prompt_info[2], que)
                service_2 = ChatService(system_pt)
                # print('pre-rank question: {}'.format(prompt_info[2]))
                rand_uuid = str(uuid.uuid4())
                question = "{{{}}}: ".format(rand_uuid) + """Directly output your choices of agents.""" + "\n" + PRE_RANK_FORMAT
                try:
                    completion = service_2.ask(question, MODEL).rstrip('\n')
                except Exception as e:
                    print(e)
                tops = parse_ranks(completion, max_num=len(roles), random_num=7)
                # print('pre-rank answer: {}'.format(completion))
                # print('tops: {}'.format(tops))
                pre_rank_roles = [roles[i] for i in tops]
                llmlp.agent_roles = pre_rank_roles
                llmlp.agents = len(pre_rank_roles)
                llmlp.init_nn(0, llmlp.agent_roles)

            res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(que)

            imp_score = 0
            completion_list.append(completions)
            accs.append(ans == res)
            resp_cnts += resp_cnt
            importances.append(imp_score)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            with open(DIR_NAME+'/'+EXP_NAME+'/'+filename_without_ext+'.json', 'a') as f:
                f.write(json.dumps(completions) + '\n')

        # print(accs)
        # print(resp_cnts)
        # print(importances)

        with open(DIR_NAME+'/'+EXP_NAME+'/'+filename_without_ext+'.txt', 'w') as f:
            f.write(str(accs) + ' ' + str(sum(accs)/len(qa_pairs)) + '\n')
            f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
            f.write(json.dumps(importances) + '\n')
            f.write(json.dumps([pos/len(qa_pairs) for pos in importances]) + '\n')
            f.write(str(total_prompt_tokens) + '\n')
            f.write(str(total_completion_tokens) + '\n')
        # break
