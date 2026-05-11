import subprocess
from subprocess import PIPE
from prompt_iteration.prompt import Prompt
from listwise_math import listwise

import os
import time

import numpy as np
import json
import argparse
import datetime
import gc
import logging
from copy import deepcopy




batch_size = 512
lr = '5e-4'
export_num = 2
specific_export_num = 4
history_augment = False
prompt_augment = True



# MODEL="gpt-3.5-turbo"
# MODEL=gpt-4
# ENGINE = "chatgpt0301"


# ROLES="['PythonAssistant', 'AlgorithmDeveloper', 'ComputerScientist', 'Programmer']"
# JUDGES="['Passer', 'Tester', 'Reflector', 'Ranker']"
# ROLES= ['PythonAssistant', 'AlgorithmDeveloper', 'ComputerScientist', 'Programmer']


def testify(prompt, prompt_role, prompt_id, DIR_NAME_TEST, iter):

    print('---------------testing----------------')
    opt_prompt = deepcopy(prompt.id2prompt[prompt_id])
    optimized_prompts = deepcopy(prompt.optimized_prompts)
    # roles = prompt.roles_list[prompt_id]
    print('--------------- Agent Collaboration ----------------')

    roles = prompt.base_roles

    if prompt_role == 'construct-role':
        roles = roles + ['construct-role']
    if 'construct-role' in optimized_prompts:
        roles = roles + ['construct-role']

    EXP_NAME="{}_iter_{}_pid_{}".format(prompt_role, iter, prompt_id)

    prompt_info = [prompt_role, opt_prompt]
    
    score = listwise(EXP_NAME, DIR_NAME_TEST, "test", prompt.model, roles, prompt_info, optimized_prompts)

    with open(os.path.join(DIR_NAME_TEST, f'{prompt_role}_iter_{iter}_pid_{prompt_id}_result.txt'), 'w') as f: 
        f.write("Accuracy: " + str(score))
    
    # gc.collect()
    return score


def agent_collaboration(prompt, prompt_role, DIR_NAME, iter, stage):
    
    for pt in prompt.new_prompts:
        prompt_id = prompt.prompt2id['|sys_exa|'.join(pt)]
        opt_prompt = deepcopy(pt)
        # role_name = prompt.role_names[prompt_id]
        optimized_prompts = deepcopy(prompt.optimized_prompts)
        print('--------------- Agent Collaboration ----------------')

        roles = prompt.base_roles

        if prompt_role == 'construct-role':
            roles = roles + ['construct-role']
        if 'construct-role' in optimized_prompts:
            roles = roles + ['construct-role']

        EXP_NAME="{}_iter_{}_pid_{}".format(prompt_role, iter, prompt_id)

        prompt_info = [prompt_role, opt_prompt]
        score = listwise(EXP_NAME, DIR_NAME, stage, prompt.model, roles, prompt_info, optimized_prompts)

        prompt.id2score[prompt_id] = score

    return None
