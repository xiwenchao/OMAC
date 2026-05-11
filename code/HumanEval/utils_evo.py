import subprocess
from subprocess import PIPE
from prompt_iteration.prompt import Prompt
from listwise_human_eval import listwise

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

def load_collaboration_roles():
    with open('agent_config.json', 'r') as f:
        config = json.load(f)
    type_to_agents = config['type_to_agents']
    return type_to_agents['Coder'], type_to_agents['Verifier']

ROLES, JUDGES = load_collaboration_roles()

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

    for part in range(2):
        EXP_NAME="llmlp_human_eval_{}_{}_{}_pid_{}".format(part, prompt_role, iter, prompt_id)
        prompt_info = [prompt_role, role_name, opt_prompt]
        listwise(part, EXP_NAME, MODEL, DIR_NAME_TEST, roles, JUDGES, prompt_info, optimized_prompts)
        
    print('---------------Evaluate Prompts ----------------')
    for part in range(2):
        EXP_NAME="llmlp_human_eval_{}_{}_{}_pid_{}".format(part, prompt_role, iter, prompt_id)
        subprocess.run(f'cat {DIR_NAME_TEST}/{EXP_NAME}.jsonl >> {DIR_NAME_TEST}/llmlp_human_eval_{prompt_role}_{iter}_{prompt_id}.jsonl', shell=True)
    subprocess.run(f'evaluate_functional_correctness {DIR_NAME_TEST}/llmlp_human_eval_{prompt_role}_{iter}_{prompt_id}.jsonl > {DIR_NAME_TEST}/llmlp_human_eval_{prompt_role}_iter_{iter}_{prompt_id}_test.txt', shell=True)
    gc.collect()
    return None


def agent_collaboration(prompt, prompt_role, DIR_NAME_VAL, iter):
    
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

        for part in range(2, 4):
            EXP_NAME="llmlp_human_eval_{}_{}_{}_pid_{}".format(part, prompt_role, iter, prompt_id)
            prompt_info = [prompt_role, role_name, opt_prompt]
            listwise(part, EXP_NAME, MODEL, DIR_NAME_VAL, roles, JUDGES, prompt_info, optimized_prompts)

        print('--------------- Evaluate Prompts ----------------')
        for part in range(2, 4):
            EXP_NAME="llmlp_human_eval_{}_{}_{}_pid_{}".format(part, prompt_role, iter, prompt_id)
            subprocess.run(f'cat {DIR_NAME_VAL}/{EXP_NAME}.jsonl >> {DIR_NAME_VAL}/llmlp_human_eval_{prompt_role}_{iter}_{prompt_id}.jsonl', shell=True)
        subprocess.run(f'evaluate_functional_correctness {DIR_NAME_VAL}/llmlp_human_eval_{prompt_role}_{iter}_{prompt_id}.jsonl > {DIR_NAME_VAL}/llmlp_human_eval_{prompt_role}_iter_{iter}_{prompt_id}.txt', shell=True)

        result_dir = os.path.join(DIR_NAME_VAL, f'llmlp_human_eval_{prompt_role}_iter_{iter}_{prompt_id}.txt')
        score = calculate_score(result_dir)
        prompt.id2score[prompt_id] = score
    return None


def calculate_score(result_dir):
    with open(result_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'pass@1' in line:
                score = line.split(':')[1].strip().replace('}', '')
                return float(score)
    return None
