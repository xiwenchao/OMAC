import subprocess
from subprocess import PIPE
from prompt_iteration.prompt import Prompt
from listwise_mmlu import listwise
from calc_score import calculate_score

import os
import time

import numpy as np
import json
import argparse
import datetime
import gc
from copy import deepcopy



MODEL="gpt-3.5-turbo"
# MODEL=gpt-4

def load_candidate_agents():
    with open('agent_config.json', 'r') as f:
        config = json.load(f)
    return config['candidate_agents']

ROLES = load_candidate_agents()


def testify(prompt, prompt_role, prompt_id, DIR_NAME_TEST, iter):

    print('---------------testing----------------')
    opt_prompt = prompt.id2prompt[prompt_id]
    role_name = prompt.role_names[prompt_id]
    optimized_prompts = deepcopy(prompt.optimized_prompts)
    # roles = deepcopy(prompt.roles_list[prompt_id])
    print('--------------- Agent Collaboration ----------------')

    roles = deepcopy(ROLES)
    if prompt_role == 'construct-role':
        roles = roles + [role_name]
    if 'construct-role' in optimized_prompts:
        roles = roles + [optimized_prompts['construct-role'][0]]

    EXP_NAME="llmlp_mmlu_{}_{}_pid_{}".format(prompt_role, iter, prompt_id)
    current_dir = os.path.join(DIR_NAME_TEST, EXP_NAME)
    os.makedirs(current_dir, exist_ok=True)

    prompt_info = [prompt_role, role_name, opt_prompt]
    listwise(EXP_NAME, MODEL, DIR_NAME_TEST, 'test', roles, prompt_info, optimized_prompts)
        
    print('---------------Evaluate Prompts ----------------')

    result_dir = os.path.join(DIR_NAME_TEST, f'llmlp_mmlu_{prompt_role}_iter_{iter}_{prompt_id}.csv')
    score = calculate_score(DIR_NAME_TEST, EXP_NAME, result_dir)
    with open(os.path.join(DIR_NAME_TEST, f'llmlp_mmlu_{prompt_role}_iter_{iter}_{prompt_id}_test.txt'), 'w') as f:
        f.write(f"The final average accuracy is: {score}")
        print(f"The final average accuracy is: {score}")
    return None


def agent_collaboration(prompt, prompt_role, DIR_NAME_TRAIN, iter):
    
    for pt in prompt.new_prompts:
        prompt_id = prompt.prompt2id[pt]
        opt_prompt = pt
        role_name = prompt.role_names[prompt_id]
        optimized_prompts = deepcopy(prompt.optimized_prompts)
        # roles = deepcopy(prompt.roles_list[prompt_id])
        print('--------------- Agent Collaboration ----------------')

        roles = deepcopy(ROLES)
        if prompt_role == 'construct-role':
            roles = roles + [role_name]
        if 'construct-role' in optimized_prompts:
            roles = roles + [optimized_prompts['construct-role'][0]]

        EXP_NAME="llmlp_mmlu_{}_{}_pid_{}".format(prompt_role, iter, prompt_id)
        current_dir = os.path.join(DIR_NAME_TRAIN, EXP_NAME)
        os.makedirs(current_dir, exist_ok=True)

        '''
        p = subprocess.run(['python', 'llmlp_listwise_human_eval.py',
                        str(part), EXP_NAME, MODEL, DIR_NAME_VAL, ROLES, JUDGES, prompt_role, opt_prompt], stdout=PIPE, stderr=PIPE, text=True, check=True)
        print(p.stdout)
        print("subprocess stderr:", p.stderr)
        # gc.collect()
        '''
        prompt_info = [prompt_role, role_name, opt_prompt]
        listwise(EXP_NAME, MODEL, DIR_NAME_TRAIN, 'train', roles, prompt_info, optimized_prompts)

        print('--------------- Evaluate Prompts ----------------')

        result_dir = os.path.join(DIR_NAME_TRAIN, f'llmlp_mmlu_{prompt_role}_iter_{iter}_{prompt_id}.csv')
        score = calculate_score(DIR_NAME_TRAIN, EXP_NAME, result_dir)
        prompt.id2score[prompt_id] = score
    return None
