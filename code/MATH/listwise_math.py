import math
import re
import pandas as pd
import json
import time
import random
import openai
import sys
import os
from util import _strip_string, extract_math_answer, is_equiv
import backoff
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout
from util import OutOfQuotaException, AccessTerminatedException
from prompt_lib import *
import os
import glob
import uuid
import multiprocessing
import logging
from evol_eval_math import eval_math
from prompt_iteration.chat_service import ChatService



# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

def construct_message(agents, question):
    if len(agents) == 0:
        # unused
        return {"role": "user", "content": "Can you double-check that your answer is correct? Put your final answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."}

    prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question +  "\n\nThese are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[-1]["content"]
        response = "\n\nOne agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and the other agents' solutions step by step. Notice that the former answers might be all wrong.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_ranking_message(agents, question, rank_prompt, max_agents=2):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double-check that your answer is correct? Put your final answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."}

    # please get the text between 'Please solve the problem below.' and the last '\nAnswer:' of question
    question = question.split('Please solve the problem below.')[-1].split('\nAnswer:')[0]
    prefix_string = "Answer the mathematics problem.\n\n" + question +  "\n\nThese are the solutions to the problem from other agents: "

    for aid, agent in enumerate(agents, 1):
        agent_response = agent[-1]["content"]
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n" + rank_prompt + '\n' + RANK_FORMAT.format(max_agents)
    return {"role": "user", "content": prefix_string} #TODO: add role as judge


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def construct_prerank_message(agents, prerank_prompt, subdir, file):

    with open(os.path.join(subdir, file), 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {file}", e)
            raise e
        prob_content = problem_data["problem"]

    prefix_string = "Here is the question:\n" + prob_content

    for aid, agent in enumerate(agents, 1):
        system_prompt = agent[-1]["content"]
        response = "\n\nFunctional description of agent " + str(aid) + ": ```{}```".format(system_prompt)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n" + prerank_prompt # .format(question)
    return prefix_string


def parse_ranks_2(completion, max_num=4, random_num=2):
    if not isinstance(completion, str):
        content = completion["choices"][0]["message"]["content"]
    else:
        content = completion
    # pattern = r'\[([1234]),\s*([1234])\]'
    pattern = r'\[([1-9])(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?(?:,\s*([1-9]))?\]'
    matches = [tuple(filter(None, match)) for match in re.findall(pattern, content)]

    try:
        match = matches[-1]
        tops = [int(match[i])-1 for i in range(len(match))]
        def clip(x):
            if x < 0:
                return 0
            if x > max_num-1:
                return max_num-1
            return x
        tops = [clip(x) for x in tops]
        tops = list(set(tops))
    except:
        print("error in parsing ranks")
        print("content: ", content)
        tops = random.sample(list(range(max_num)), random_num)
    return tops

def load_agent_config():
    with open('agent_config.json', 'r') as f:
        return json.load(f)

def limit_agent_contexts(agent_contexts, store_contexts, max_agents):
    if max_agents <= 0 or max_agents >= len(agent_contexts):
        return agent_contexts, store_contexts
    selected = sorted(random.sample(list(range(len(agent_contexts))), max_agents))
    return [agent_contexts[i] for i in selected], [store_contexts[i] for i in selected]

@backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout), max_tries=20)
def generate_answer(answer_context, MODEL):
    try:
        completion = openai.ChatCompletion.create(
                  model=MODEL,
                  messages=answer_context,
                  temperature=0.2,
                  max_tokens=2048,
                  n=1)
    except RateLimitError as e:
        if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
            raise OutOfQuotaException(openai.api_key)
        elif "Your access was terminated due to violation of our policies" in e.user_message:
            raise AccessTerminatedException(openai.api_key)
        else:
            raise e

    return completion


def parse_question_answer(subdir, file, examples):
    
    def find_math_answer(s):
        assert('boxed' in s)
        # s = s.replace(",", "")
        ans = s.split('boxed')[-1]
        if(ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if(c == '{'):
                    stack += 1
                    a += c
                elif(c == '}'):
                    stack -= 1
                    if(stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a=_strip_string(a)
        return a

    with open(os.path.join(subdir, file), 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {file}", e)
            raise e
        prob_content = problem_data["problem"]
        question = examples + "\n\nPlease solve the problem below.\nProblem: " + prob_content + "\nAnswer:"
        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None

        # answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
        answer = find_math_answer(problem_data['solution'])

        return question, prob_level, prob_type, answer

def parse_ranks(completion):
    content = completion["choices"][0]["message"]["content"]
    pattern = r'\[([1-9]),\s*([1-9])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > 3:
                return 3
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = [0, 1]

    return tops

def check_reach_consensus(agent_contexts):
    pred_solutions = [context[-1]["content"] for context in agent_contexts]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = extract_math_answer(pred_solution)
        if pred_answer:
            pred_answers.append(pred_answer)

    if len(pred_answers) == 0:
        print("No answer found")
        return False
    
    def most_frequent(List):
        counter = 0
        num = List[0]

        for i in List:
            current_frequency = sum(is_equiv(i, item) for item in List)
            if current_frequency > counter:
                counter = current_frequency
                num = i

        return num, counter
    
    consensus_answer, counter = most_frequent(pred_answers)
    if counter > math.floor(2/3 * len(agent_contexts)):
        # print("Consensus answer: {}".format(consensus_answer))
        # logging.info("Consensus answer: {}".format(consensus_answer))
        return True

def exp_math_dir_cot(EXP_NAME, DIR_NAME, SUB_DIR, MIN_FILENAME, MAX_FILENAME, MODEL, roles, prompt_info, optimized_prompts):

    RESPONSES_TOTAL = DIR_NAME+f"/{EXP_NAME}_responses_total.txt"
    TOKENS_TOTAL = DIR_NAME+f"/{EXP_NAME}_tokens_total.txt"

    random.seed(0)
    response_dict = {}
    idx = 0
    total_responses = 0
    total_prompt_tokens, total_completion_tokens = 0, 0

    prompt_role, opt_prompt = prompt_info

    for subdir, dirs, files in os.walk(SUB_DIR):
        for file in files:
            file_num = int(os.path.splitext(file)[0])  # Get the filename without extension and convert to int
            if not (MIN_FILENAME <= file_num <= MAX_FILENAME):
                continue

            agent_contexts = list()
            store_conetxts = list()

            examples = EXAMPLES_ORIG
            rank_prompt = RANK_DEFAULT

            if prompt_role == "Examples":
                examples = opt_prompt[0] if isinstance(opt_prompt, list) else opt_prompt
            elif  "Examples" in optimized_prompts:
                examples = optimized_prompts["Examples"][0]
            if prompt_role == "rank":
                rank_prompt = opt_prompt[0] if isinstance(opt_prompt, list) else opt_prompt
            elif "rank" in optimized_prompts:
                rank_prompt = optimized_prompts["rank"][0]
            question, prob_level, prob_type, answer = parse_question_answer(subdir, file, examples)


            for role in roles:
                system_prompt = SYSTEM_PROMPT
                if prompt_role == "System":
                    system_prompt = opt_prompt[0] if isinstance(opt_prompt, list) else opt_prompt
                elif role in optimized_prompts:
                    if role == "System":
                        system_prompt = optimized_prompts[role][0]
                
                agent_contexts.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": question}])
                store_conetxts.append([{"role": "system", "content": system_prompt}])

            consensus = False
            for i, agent_context in enumerate(agent_contexts):
                # print(idx, 0, i, agent_context, "\n")
                completion, prompt_tokens, completion_tokens = generate_answer(agent_context, MODEL)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                # print(completion, "\n")
                total_responses += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts[:i+1], answer, prob_level, prob_type)
                    consensus = True
                    break

            if consensus:
                continue

            consensus = False
            message = construct_ranking_message(agent_contexts, question, rank_prompt)
            for i, agent_context in enumerate(agent_contexts):
                agent_context.pop()
                agent_context.pop()
                agent_context.append(message)
                # print(idx, 1, i, agent_context, "\n")
                completion, prompt_tokens, completion_tokens = generate_answer(agent_context, MODEL)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                # print(completion, "\n")
                total_responses += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
                    consensus = True
                    break

            if consensus:
                continue

            # TODO: PageRanker
            message = construct_ranking_message(agent_contexts, question, rank_prompt)
            completion, prompt_tokens, completion_tokens = generate_answer([message], MODEL)
            total_responses += 1
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            # print(completion, "\n")
            tops = parse_ranks(completion)
            agent_contexts = [agent_contexts[top] for top in tops]

            if check_reach_consensus(agent_contexts):
                response_dict[question] = (agent_contexts, answer, prob_level, prob_type)
                continue

            message = construct_message(agent_contexts, question)
            for i, agent_context in enumerate(agent_contexts):
                agent_context.pop()
                agent_context.pop()
                agent_context.append(message)
                # print(idx, 2, i, agent_context, "\n")
                completion, prompt_tokens, completion_tokens = generate_answer(agent_context, MODEL)
                total_responses += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                # print(completion, "\n")

            response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
            idx += 1
        

    json.dump(response_dict, open(DIR_NAME+"/{}_{}_{}_{}.json".format(EXP_NAME, os.path.basename(os.path.normpath(SUB_DIR)), MIN_FILENAME, MAX_FILENAME), "w"))
    with open(RESPONSES_TOTAL, "a") as f:
        f.write("{}\n".format(total_responses))
    with open(TOKENS_TOTAL, "a") as f:
        f.write("Prompt tokens: {}, Completion tokens: {}\n".format(total_prompt_tokens, total_completion_tokens))
    
    return None

def exp_math_dir(EXP_NAME, DIR_NAME, SUB_DIR, MIN_FILENAME, MAX_FILENAME, MODEL, roles, prompt_info, optimized_prompts):

    RESPONSES_TOTAL = DIR_NAME+f"/{EXP_NAME}_responses_total.txt"
    MIN_FILENAME, MAX_FILENAME = int(MIN_FILENAME), int(MAX_FILENAME)

    random.seed(0)
    response_dict = {}
    idx = 0
    total_responses = 0

    prompt_role, opt_prompt = prompt_info
    agent_config = load_agent_config()
    layer_max_agents = agent_config.get('layer_max_agents', [len(roles), len(roles), 2])

    for subdir, dirs, files in os.walk(SUB_DIR):
        for file in files:
            file_num = int(os.path.splitext(file)[0])  # Get the filename without extension and convert to int
            if not (MIN_FILENAME <= file_num <= MAX_FILENAME):
                continue

            agent_contexts = list()
            store_conetxts = list()
        
            if prompt_role == "rank":
                rank_prompt = opt_prompt[0]
            elif "rank" in optimized_prompts:
                rank_prompt = optimized_prompts["rank"][0]
            else:
                rank_prompt = RANK_DEFAULT
            
            new_agent = True if prompt_role == "construct-role" else False
            for role in roles:
                if 'System' in optimized_prompts:
                    system_prompt = optimized_prompts['System'][0]
                else:
                    system_prompt = SYSTEM_PROMPT

                if 'Examples' in optimized_prompts:
                    examples = optimized_prompts['Examples'][0]
                elif 'Examples_cot' in optimized_prompts:
                    examples = optimized_prompts['Examples_cot'][0]
                elif 'Examples_cot' in role:
                    examples = EXAMPLES_COT
                else:
                    examples = EXAMPLES_ORIG
                
                if "construct-role" in role and not new_agent:
                    if role in optimized_prompts:
                        system_prompt = optimized_prompts[role][0]
                        examples = optimized_prompts[role][1]

                if prompt_role == "System":
                    system_prompt = opt_prompt[0]
                elif prompt_role in ["Examples", "Examples_cot"]:
                    examples = opt_prompt[0]
                elif prompt_role == "construct-role" and prompt_role in role:
                    system_prompt = opt_prompt[0]
                    examples = opt_prompt[1]
                    new_agent = False
                
                question, prob_level, prob_type, answer = parse_question_answer(subdir, file, examples)
                
                agent_contexts.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": question}])
                store_conetxts.append([{"role": "system", "content": system_prompt}])

            if prompt_role == "pre-rank":
                system_pt = construct_prerank_message(store_conetxts, opt_prompt[0], subdir, file)
                service_2 = ChatService(system_pt)
                # logging.info('pre-rank question: {}'.format(opt_prompt[0]))
                qs = """Directly output your choices of agents.""" + "\n" + PRE_RANK_FORMAT
                try:
                    completion = service_2.ask(qs, MODEL).rstrip('\n')
                except Exception as e:
                    print(e)
                tops = parse_ranks_2(completion, max_num=len(roles), random_num=4)
                # logging.info('pre-rank answer: {}'.format(completion))
                # logging.info('tops: {}'.format(tops))
                
                agent_contexts = [agent_contexts[i] for i in tops]
                store_conetxts = [store_conetxts[i] for i in tops]

            agent_contexts, store_conetxts = limit_agent_contexts(
                agent_contexts, store_conetxts, layer_max_agents[0]
            )

            consensus = False
            for i, agent_context in enumerate(agent_contexts):
                # print(idx, 0, i, agent_context, "\n")
                completion = generate_answer(agent_context, MODEL)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                # print(completion, "\n")
                total_responses += 1

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts[:i+1], answer, prob_level, prob_type)
                    consensus = True
                    break

            if consensus:
                continue

            agent_contexts, store_conetxts = limit_agent_contexts(
                agent_contexts, store_conetxts, layer_max_agents[1]
            )
            agent_structure = {}
            for idx, agent in enumerate(agent_contexts):
                if prompt_role == 'structure' or 'structure' in optimized_prompts:
                    structure_prompt = opt_prompt[0] if prompt_role == 'structure' else optimized_prompts['structure'][0]
                    agent_structure[idx] = get_structure(agent, agent_contexts, structure_prompt)
                else:
                    agent_structure[idx] = [i for i in range(len(agent_contexts))]

            consensus = False
            # message = construct_message(agent_contexts, question)
            for i, agent_context in enumerate(agent_contexts):
                agent_context.pop()
                agent_context.pop()
                current_contexts = [agent_contexts[k] for k in agent_structure[i]]
                message = construct_message(current_contexts, agent_context[-1]["content"])
                agent_context.append(message)
                # print(idx, 1, i, agent_context, "\n")
                completion = generate_answer(agent_context, MODEL)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                # print(completion, "\n")
                total_responses += 1

                if i >= math.floor(2/3 * len(agent_contexts)) and check_reach_consensus(agent_contexts[:i+1]):
                    response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
                    consensus = True
                    break

            if consensus:
                continue

            # TODO: PageRanker
            final_max_agents = layer_max_agents[2] if len(layer_max_agents) > 2 else 2
            message = construct_ranking_message(agent_contexts, question, rank_prompt, final_max_agents)
            completion = generate_answer([message], MODEL)
            total_responses += 1
            # print(completion, "\n")
            tops = parse_ranks_2(
                completion,
                max_num=len(agent_contexts),
                random_num=min(final_max_agents, len(agent_contexts)),
            )
            agent_contexts = [agent_contexts[top] for top in tops]
            store_conetxts = [store_conetxts[top] for top in tops]  # Shijun

            if check_reach_consensus(agent_contexts):
                response_dict[question] = (agent_contexts, answer, prob_level, prob_type)
                continue

            # message = construct_message(agent_contexts, question)
            # debug
            if len(agent_contexts) > len(store_conetxts):
                print("tops: ", tops)
                print("len_store: ", len(store_conetxts))
            for i, agent_context in enumerate(agent_contexts):
                agent_context.pop()
                agent_context.pop()
                message = construct_message(agent_contexts, agent_context[-1]["content"])
                agent_context.append(message)
                # print(idx, 2, i, agent_context, "\n")
                completion = generate_answer(agent_context, MODEL)
                total_responses += 1

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                store_conetxts[i].extend(agent_context[1:])
                # print(completion, "\n")

            response_dict[question] = (store_conetxts, answer, prob_level, prob_type)
            idx += 1

            print("end file: ", file_num)
        print("end subdir: ", subdir)

    json.dump(response_dict, open(DIR_NAME+"/{}_{}_{}_{}.json".format(EXP_NAME, os.path.basename(os.path.normpath(SUB_DIR)), MIN_FILENAME, MAX_FILENAME), "w"))
    with open(RESPONSES_TOTAL, "a") as f:
        f.write("{}\n".format(total_responses))
    
    return None

def run_exp_math_dir_in_background(EXP_NAME, DIR_NAME, subdir, min_file, max_file, MODEL, roles, prompt_info, optimized_prompts):
    process = multiprocessing.Process(target=exp_math_dir, args=(EXP_NAME, DIR_NAME, subdir, min_file, max_file, MODEL, roles, prompt_info, optimized_prompts))
    process.daemon = True
    process.start()
    return process

def listwise(EXP_NAME, DIR_NAME, stage, MODEL, roles, prompt_info, optimized_prompts):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=2)  # .get_context("spawn")
    
    # Set up constants for MODEL and ENGINE.
    # MODEL = "35_0301"
    # ENGINE = "chatgpt0301"
    # MODEL = "4_0613"
    # ENGINE = "gpt4"

    # Specify your directory here.
    directory = "../../data/MATH/test/"

    # List to keep track of background processes.
    processes = []

    # Loop over all items in the specified directory.
    for subdir in sorted(glob.glob(os.path.join(directory, '*'))):
        if os.path.isdir(subdir):
            print(f"Processing {subdir}")
            current_time = time.time()
            
            # Find all JSON files in the subdirectory and strip the .json extension.
            json_files = [os.path.splitext(os.path.basename(f))[0] 
                        for f in glob.glob(os.path.join(subdir, "*.json"))]
            
            # Sort the file names numerically. If conversion fails, fall back to lexicographical sort.
            try:
                json_files.sort(key=lambda x: int(x))
            except ValueError:
                json_files.sort()
            
            total_files = len(json_files)
            # Calculate how many loops are needed (each loop processes up to 100 files).
            loop_num = 10
            loops = (total_files + loop_num-1) // loop_num
            print("Loops: ", loops)
            span_range = []
            if stage == "train":
                span_range = [0, 1]
            elif stage == "test":
                span_range = [1, 2]

            # Process files in batches of 100.
            for i in range(span_range[0], span_range[1]):
                start = i * loop_num
                end = (i + 1) * loop_num - 1
                if end >= total_files:
                    end = total_files - 1

                min_file = json_files[start]
                max_file = json_files[end]

                # Construct the result file name.
                # DIR_NAME+"/{}_{}_{}_{}.json".format(EXP_NAME, os.path.basename(os.path.normpath(SUB_DIR)), MIN_FILENAME, MAX_FILENAME)
                result_file_name = DIR_NAME+"/{}_{}_{}_{}.json".format(EXP_NAME, os.path.basename(os.path.normpath(subdir)), min_file, max_file)

                # Skip this batch if the result file already exists.
                if os.path.exists(result_file_name):
                    continue

                # pool.apply_async(exp_math_dir, args=(EXP_NAME, DIR_NAME, subdir, min_file, max_file, MODEL, roles, prompt_info, optimized_prompts))
                exp_math_dir(EXP_NAME, DIR_NAME, subdir, min_file, max_file, MODEL, roles, prompt_info, optimized_prompts)


                '''
                

                _p = run_exp_math_dir_in_background(EXP_NAME, DIR_NAME, subdir, min_file, max_file, MODEL, roles, prompt_info, optimized_prompts)
                processes.append(_p)

                if len(processes) >= 6:
                    for _p in processes:
                        _p.join()
                    processes = []
                    print("Finished processing 6 batches")

                # Echo the processing details.
                print(f"{subdir} {min_file} {max_file}")
                

                    
                # Launch the processing script in the background.
                p = subprocess.Popen([
                    "python", "llmlp_gen_math_listwise_deeper_markov.py",
                    EXP_NAME, DIR_NAME_VAL, subdir, min_file, max_file, MODEL, ENGINE
                ])
                processes.append(p)
                '''
                
                # This additional echo replicates the extra background echo in the Bash script.
                # print(f"{subdir} {min_file} {max_file}")
            
            print(f"Finished processing {subdir}")
            end_time = time.time()
            elapsed_time = (end_time - current_time) / 60
            print(f"Elapsed time: {elapsed_time} minutes")

    
    current_time = time.time()
    print("Wait for all background jobs to finish.")
    pool.close()
    pool.join()
    del pool
    end_time = time.time()
    elapsed_time = (end_time - current_time) / 60
    print(f"Poll time: {elapsed_time} minutes")
    '''


    if len(processes) > 0:
        for p in processes:
            p.join()
    del processes


    # Wait for all background jobs to finish.
    for p in processes:
        p.wait()
    '''

    print("All done")

    # Finally, run the evaluation script.
    # subprocess.run(["python", "eval_math.py", "DIR_NAME", "None"])
    filter = EXP_NAME
    score = eval_math(DIR_NAME, filter)
    return score

def get_structure(current_agent, candidate_agents, structure_prompt):
    # get the structure of the neural network
    # current_agent: the current agent
    # candidate_agents: the candidate agents
    # return: a list of tuples (agent, weight)
    structure = []
    
    system_pt = construct_structure_message(current_agent, candidate_agents, structure_prompt)
    service_2 = ChatService(system_pt)
    # logging.info('pre-rank question: {}'.format(PROMPT_INFO[2]))
    rand_uuid = str(uuid.uuid4())
    question = "{{{}}}: ".format(rand_uuid) + """Directly output your choices of candidate agents.""" + "\n" + STRUCTURE_FORMAT
    completion = ""
    try:
        max_retries = 5
        retry_count = 0
        backoff_time = 1  # Initial backoff time in seconds
        
        while retry_count < max_retries:
            try:
                completion = service_2.ask(question, "gpt-3.5-turbo").rstrip('\n')
                break  # Successfully got response, exit the retry loop
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} attempts. Last error: {e}")
                    # Set a default completion or re-raise based on your needs
                else:
                    print(f"Attempt {retry_count} failed: {e}. Retrying in {backoff_time}s...")
                    import time
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
    except Exception as e:
        print(f"Unexpected error in retry mechanism: {e}")

    tops = parse_ranks_2(completion, max_num=len(candidate_agents), random_num=min(2, len(candidate_agents)))
    print('structure answer: {}'.format(completion))
    print('tops: {}'.format(tops))
    if len(tops) < min(2, len(candidate_agents)):
        remaining = list(set(range(len(candidate_agents))) - set(tops))
        if remaining:
            tops.append(random.choice(remaining))
    structure = tops
    return structure
