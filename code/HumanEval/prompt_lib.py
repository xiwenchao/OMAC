
TEMPERATURE = 0.8
MAX_TOKENS = 1024


ROLE_MAP = {
    "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).", # from https://github.com/composable-models/llm_multiagent_debate.git
    "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solving real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "Programmer": "You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment).", # from https://github.com/getcursor/eval.git
    "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature)."
}

ROLE_MAP_INIT = {
    "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a series of previous implementations of the same function signature and docstring. You use the previous implementations as a hint and your goal is to write your full implementation again (restate the function signature). Here are some examples.", # from https://github.com/composable-models/llm_multiagent_debate.git
    "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here are some examples.",
    "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solving real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here are some examples.",
    "Programmer": "You are an intelligent programmer. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. You must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment). Here are some examples.", # from https://github.com/getcursor/eval.git
    "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here are some examples.",
    "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here are some examples."
}


EXAMPLE_TESTER = """Example:
function signature:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

unit tests:
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False"""

EXAMPLE_REFLECTOR = """Example:
[previous impl 1]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[previous impl 2]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[reflection 1]:
The implementation will fail when no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.

[reflection 2]:
The implementation has an issue stemming from the while loop `while current_sum + nums[right] <= target:`, which directly accesses `nums[right]` without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to `while right < len(nums) and current_sum + nums[right] <= target:`. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError."""

EXAMPLE_DEBUGGER = """Example:
[previous impl 1]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[previous impl 2]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[bug fix 1]:
Expression `right - left + 1 >= max_length` is buggy. I suggest to fix it as `right - left + 1 > max_length`.

[bug fix 2]:
The while loop `while current_sum + nums[right] <= target:` is a bug. It's better to use `while right < len(nums) and current_sum + nums[right] <= target:`."""

JUDGE_MAP = {
    "Passer": "",
    "Tester": f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring.""",
    "Reflector": f"You are a Python writing assistant. You will be given a series of function implementations of the same function signature. Write a few sentences to explain whether and why the implementations are wrong. These comments will be used as a hint and your goal is to write your thoughts on the n-th previous implementation after [reflection n].",
    "Ranker": "You are a Python writing assistant. You will be given a series of function implementations of the same function signature. You need to choose the best 2 implementations in consideration of correctness, efficiency, and possible corner cases.",
    "Debugger": f"You are a debugger, specialized in finding and fixing bugs in Python code. You will be given a function implementation with a bug in it. Your goal is to identify the bug and provide a corrected implementation. Include comments to explain what was wrong and how it was fixed. Here's the example.",
    "QualityManager": f"You are a quality manager, ensuring that the code meets high standards in terms of readability, efficiency, and accuracy. You will be given a function implementation and you need to provide a code review. Comment on its correctness, efficiency, and readability, and suggest improvements if needed."
}

JUDGE_PREFIX = {
    "Passer": "[syntax check {}]:\n",
    "Tester": "[unit test results {}]:\n",
    "Reflector": "[reflection {}]:\n",
    "Debugger": "[bug fix {}]:\n",
    "QualityManager": "[code review {}]:\n",
}

JUDGE_NAMES = {
    "Passer": "Syntax Checker",
    "Tester": "Unit Tests",
    "Reflector": "Reflector",
    "Ranker": "Ranker",
    "Debugger": "Debugger",
    "QualityManager": "Quality Manager",
}


def construct_ranking_message(responses, question, qtype, rank_prompt, max_agents):
    if qtype == "single_choice":
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

            prefix_string = prefix_string + response

        # prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
        prefix_string = prefix_string + "\n\n" + rank_prompt.format(max_agents)
        return {"role": "user", "content": prefix_string}

    elif qtype == "code_completion":
        prefix_string = "Here are some implementations of the same function and the thoughts of them. The function has a signature and a docstring explaining its functionality.\n\n"
        for aid, agent_response in enumerate(responses, start=1):
            response = "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

            prefix_string = prefix_string + response

        # prefix_string = prefix_string + """[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.""".format(question)
        prefix_string = prefix_string + """[function signature]:\n```python\n{}\n```\n\n""".format(question) + rank_prompt.format(max_agents)
        return {"role": "user", "content": prefix_string}
    else:
        raise NotImplementedError

def construct_message(responses, question, qtype):
    if qtype == "single_choice":
        if len(responses) == 0:
            prefix_string = "Here is the question:\n" + question + "\n\nPut your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."
            return {"role": "user", "content": prefix_string}

        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and the other agents' solutions step by step. Notice that their answers might be all wrong. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D). Along with the answer, give a score from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
        return {"role": "user", "content": prefix_string}

    elif qtype == "code_completion":
        if len(responses) == 0:
            qtemplate = "```python\n{}\n```"
            qtemplate = '''You must complete the python function I give you.
Be sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.
[function impl]:
{}\nOnce more, please follow the template by repeating the original function, then writing the completion.'''.format(qtemplate.format(question))
            return {"role": "user", "content": qtemplate}
        else:
            flag_ranker = False
            ranker = None
            for agent in responses:
                if agent.role == "Ranker":
                    flag_ranker = True
                    ranker = agent
                    break
            pre_impls = list(responses[0].get_answer().keys())
            if flag_ranker:
                ranks = ranker.get_answer()
                pre_impls = [pre_impl for pre_impl in pre_impls if ranks[pre_impl]]
            
            prefix_string = ""
            for impl_id, pre_impl in enumerate(pre_impls, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(impl_id, pre_impl)
                for agent in responses:
                    if agent.role == "Ranker":
                        continue
                    reply = agent.get_answer()[pre_impl]
                    prefix_string += JUDGE_PREFIX[agent.role].format(impl_id) + reply + "\n\n"
            
            def connect_judges(judges):
                # connect the name by ',' and "and"
                judges = [judge for judge in judges if judge.role != "Ranker"]
                if len(judges) == 0:
                    raise ValueError("No judges")
                if len(judges) == 1:
                    return JUDGE_NAMES[judges[0].role]
                if len(judges) == 2:
                    return JUDGE_NAMES[judges[0].role] + " and " + JUDGE_NAMES[judges[1].role]
                return ", ".join([JUDGE_NAMES[judge.role] for judge in judges[:-1]]) + " and " + JUDGE_NAMES[judges[-1].role]

            prefix_string = prefix_string + "You must complete the python function I give you by rectifying previous implementations. Use the other information as a hint.\nBe sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.\n[improved impl]:\n```python\n{}\n```\n\nPlease follow the template by repeating the function signature and complete the new implementation in [improved impl]. If no changes are needed, simply rewrite the implementation in the Python code block.\nAlong with the new implementation, give a score from 1 to 5 to {} in terms of helpfulness. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(question, connect_judges(responses), len(responses)-1)

            return {"role": "user", "content": prefix_string}
    else:
        raise NotImplementedError

def construct_judge_message(responses, question, qtype, role):
    if qtype == "code_completion":
        if role == "Tester":
            prefix_string = "function signature:\n```python\n{}\n```\n\nunit tests:\n".format(question)
            return {"role": "user", "content": prefix_string}
        elif role == "Reflector":
            prefix_string = "Here are previous implementations of the same function. The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Write your reflection on these implementations in consideration of correctness, efficiency, and possible corner cases. Put your reflection of the n-th implementation after [reflection n].\nAlong with the reflections, give a score from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "Debugger":
            prefix_string = "Here are previous implementations of the same function. The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Debug each implementation and write your feedback as a debugger. Put your debug information for the n-th implementation after [bug fix n].\nPlease give a score from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "QualityManager":
            prefix_string = "Here are previous implementations of the same function. The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Write your code review on these implementations in multiple aspects. Put your review of the n-th implementation after [code review n].\nAlso, give a score from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "Ranker":
            prefix_string = "Here are some implementations of the same function. The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)

                prefix_string = prefix_string + response

            prefix_string = prefix_string + "[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.".format(question)
            return {"role": "user", "content": prefix_string}
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


RANK_DEFAULT = "Take correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."

RANK_FORMAT = "Please ensure to specify the choice of solutions strictly in the format of a single list as described above (like [1,2] or [1,2,3] or [1,2,3,4]). The elements of this list should represent the sequence numbers of all selected solutions. Avoid expressing your choice in the format like '[function impl 1] and [function impl 2]'. Also, ensure that anything within '[]' consists solely of sequence numbers of solutions, separated by commas, with no additional text like 'function impl'. Note that the maximum number of solutions you can choose is {}."

RANK_INIT = """Create {} different prompts for an LLM to choose some top solutions that match a previously provided function's docstring best. 
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
You can decide the number of the chosen solutions and the content of the prompt.
The prompt should help to accurately and efficiently select the top solutions that match the function's docstring best.
Note that all the solutions have been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for solution selection.
You may consider adding more detailed and thorough instructions to help the LLM select the top solutions better.
The generated prompt should specify the output format like the given example (also ensure that it is different from the example prompt):
"{}"."""

RANK_FD = """Create and output a child prompt for an LLM to choose some top solutions that match a previously provided function's docstring best.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to select more useful and effective solutions that match the function's docstring.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose top solutions more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

ROLE_INIT = """Generate {} distinct prompts to instruct an LLM to implement functions with Python for some unknown, unspecified function signatures and docstrings. The LLM's role is '{}'. Each prompt should guide the model to accurately and efficiently generate the Python implementation while adhering to the specified role.
Do not output all the prompts at once. Instead, I will provide a sequence number, and you should return only the corresponding prompt one by one. 
Ensure that the prompts are not for a specific function signature and docstring. Instead, they should be general enough to apply to all possible function signatures and docstrings that will be provided during inference.
Do not create any specific instances of the function signature and docstring in the prompt, because they are not provided now.
You should consider adding more detailed, logical, and thorough instructions, which can help the LLM generate the code better.
Ensure that the generated prompts follow the given example format but differ in content and structure from the example itself. The example is as follows:
"{}"."""
# Ensure that the generated prompts follow the given example format but differ in wording and structure from the example itself
# You can decide the content and detailed functional instructions of the prompt.

ROLE_FD = """Generate and output a child prompt for an LLM to write Python code that implements some unknown, unspecified function signatures and docstrings, which will be provided by users during inference. The LLM's role is '{}'.
At the end, a pair of parent prompts is provided: one positive and one negative. The positive parent prompt has been shown to be more effective and efficient in guiding the LLM to generate high-quality code following the given role.
Your task is to carefully compare the two parent prompts, identifying the key reasons why the positive parent prompt performs better. Based on these insights, generate and output a child prompt that further improves upon the positive parent prompt to enhance code quality.
Ensure that the prompts are not for a specific function signature and docstring. Instead, they should be general enough to apply to all possible function signatures and docstrings that will be provided during inference.
Do not create any instances of the function signature and docstring in the prompt, because they are not provided now.
The child prompt should be distinct from both parent prompts. Output only the content of the child prompt."""

JUDGE_INIT = """Generate {} different prompts to instruct an LLM to evaluate Python function implementations based on their signature and docstring. The LLM's role is '{}' to perform the '{}' task. Each prompt should guide the LLM to evaluate Python implementation accurately and efficiently based on this role.
Do not output all the generated prompts at once. Instead, I will provide a sequence number, and you should output only the corresponding prompt. 
You may consider adding more detailed and thorough instructions to help the LLM evaluate the Python implementations.
Ensure that the prompts remain general enough to evaluate any Python function implementations given during inference.
Ensure that the prompts are not for a specific Python function implementation. Instead, they should be general enough to evaluate all possible function implementations that will be provided during inference.
Do not create any instances of the function signature and docstring in the prompt, because they are not provided now.
The generated prompt should instruct the LLM to evaluate the implementations following the given example format while ensuring uniqueness. Each child prompt should be different. The example is as follows: 
"{}"."""

JUDGE_FD = """Create and output a child prompt that instructs an LLM to evaluate Python implementation functions based on their signatures and docstrings. The LLM's role is '{}' and its task is '{}'. I will provide a pair of parent prompts, which are given in the end. Your output should consist solely of the child prompt, adhering to these guidelines:
Create and output a child prompt that instructs an LLM to evaluate implementation functions based on their signatures and docstrings.
The positive parent prompt is known to be more effective in evaluating the implementations for the given role.
Analyze and compare the two parent prompts, identifying why the positive one is superior to the negative one.
Using your analysis, generate a new child prompt that improves upon the positive parent prompt in assessing the implementations.
Ensure that the prompts remain general enough to evaluate any Python function implementations given during inference.
Do not create any instances of the function signature and docstring in the prompt, because they are not provided now.
The child prompt must follow the same format as the parent prompts, yet it should be distinct from both of them.
Please output only the text of the resulting child prompt."""

CONSTRUCT_ROLE_INIT = """Create {} distinct prompts to instruct an LLM to write Python code that implements some unknown, unspecified function signatures and docstrings. 
First, determine the roles of these prompts. Next, create the prompts that instruct the LLM to generate the Python implementation code based on the defined roles.
Do not output all the generated roles and prompts at once. Instead, I will request either the k-th role or the k-th prompt individually. When asked, directly output the corresponding content of the role name or prompt one at a time.
You can decide the content and detailed functional instructions of the roles and prompts. You may consider adding more detailed instructions to help the LLM generate the code.
Ensure that the prompts remain general enough to apply to most function signatures and docstrings in Python given as context during inference. 
Do not create any instances of the function signature and docstring in the prompt, because they are not provided now.
The following is an example of a role and the corresponding prompt (also ensure your output is different from the example role and prompt):
"{}"."""

CONSTRUCT_ROLE_INIT_APPENDIX = """You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Here are some examples."""

CONSTRUCT_ROLE_FD = """Generate and output a pair consisting of a role name and its corresponding prompt, designed to define the function of an LLM and instruct it to write Python code implementing some unknown, unspecified function signatures and docstrings.
First, determine the role of the LLM in the coding process. Next, create a prompt based on that role.
Create a prompt that effectively instructs the LLM to generate Python implementation code based on this role.
I will provide two parent role-prompt pairs: one positive and one negative. The positive pair has been proven to be more effective in guiding the LLM to generate high-quality implementation code.
Your task is to carefully analyze both parent pairs, identifying the factors that make the positive pair superior. Based on this analysis, generate and output a child role and prompt pair that improves upon the positive parent pair and leads to even better code generation.
The child prompt must be distinct from both parent prompts while incorporating the lessons learned from their comparison.
Ensure that the prompts remain general enough to apply to most function signatures and docstrings given as context during inference.
Do not create any instances of the function signature and docstring in the prompt, because they are not provided now.
Do not output the role and prompt immediately. I will request them separately, and when asked, provide only the corresponding content—either the role name or the prompt.

The positive parent role is: {}. The negative parent role is: {}."""

PRE_RANK_DEFAULT = "Take functionality, efficiency, and necessity into consideration, choose top 3 agents best suited for the given code generation task. Think it step by step. Put your answer in the form like [1,2,3] or [1,3,4] at the end of your response."

PRE_RANK_FORMAT = "Please ensure to specify the choice of agents strictly in the format of a single list as described above (like [1,2] or [1,2,3] or [1,2,3,4]). The elements of this list should represent the sequence numbers of all selected agents. Avoid expressing your choice in the format like '[agent 1] and [agent 2]'. Also, ensure that anything within '[]' consists solely of sequence numbers of agents, separated by commas, with no additional text like 'agent'."

PRE_RANK_INIT = """Create {} distinct prompts for an LLM to choose some top agents best suited for the general code generation task. 
Each prompt should decide and specify the number of the chosen agents. The minimal number is 2 and maximum number is 4.
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should help to accurately and efficiently identify the top agents best suited for the code generation task.
Note that all information about the task and candidate agents has been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for agent selection.
You may consider adding more detailed and thorough instructions to help the LLM select the top agents better.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

PRE_RANK_FD = """Create and output a child prompt for an LLM to choose some top agents best suited for the general code generation task.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to select more useful and effective agents for code generation.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose top agents more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

STRUCTURE_DEFAULT = "Take functionality, efficiency, and necessity into consideration. Select the top candidate agents whose generated solutions or evaluations for code completion tasks can be most useful as inputs for the current agent to produce improved outputs. Think it step by step. Put your answer in the form like [1,2] or [1,3,2] at the end of your response."

STRUCTURE_FORMAT = "Please ensure to specify the choice of agents strictly in the format of a single list as described above (like [1,2] or [1,2,3] or [1,2,3,4]). The elements of this list should represent the sequence numbers of selected agents. Avoid expressing your choice in the format like '[agent 1] and [agent 2]'. Also, ensure that anything within '[]' consists solely of sequence numbers of agents, separated by commas, with no additional text like 'agent'."

STRUCTURE_INIT = """Create {} distinct prompts for an LLM to choose candidate agents whose generated solutions or evaluations for code completion tasks may be useful as inputs for the current agent to produce improved outputs.
You should decide the number of chosen agents and the content of the prompt.
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should help to accurately and efficiently identify the candidate agents whose outputs are helpful to be taken as the input for the current agent.
Note that all information about the candidate agents and the current agent has been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for agent selection.
You may consider adding more detailed and thorough instructions to help the LLM select the candidate agents better.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

STRUCTURE_FD = """Create and output a child prompt for an LLM to choose candidate agents whose generated solutions or evaluations for code completion tasks may be useful as inputs for the current agent to produce improved outputs.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to select more useful and effective agents.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose top agents more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""


def construct_prerank_message(base_roles, optimized_prompts, rank_prompt, question):
    prefix_string = "Here is the question:\n" + question

    roles_description = "\nThese are the agents and their functional description: "
    for rl_num, role in enumerate(base_roles):
        roles_description += "\nAgent {}: Its name is: {}\n".format(rl_num+1, role)
        if role in optimized_prompts:
            description = optimized_prompts[role][1]
        elif role not in ROLE_MAP:
            description = optimized_prompts['construct-role'][1]
        else:
            description = ROLE_MAP[role]
        roles_description += "Its function description is:\n" + """{}""".format(description)
    return prefix_string + roles_description + '\n' + rank_prompt

def get_agent_description(role, optimized_prompts):
    if role in optimized_prompts:
        return optimized_prompts[role][1]
    if role in ROLE_MAP:
        return ROLE_MAP[role]
    if role in JUDGE_MAP:
        return JUDGE_MAP[role]
    if 'construct-role' in optimized_prompts and role == optimized_prompts['construct-role'][0]:
        return optimized_prompts['construct-role'][1]
    return optimized_prompts.get('construct-role', [role, role])[1]

def construct_structure_message(current_agent, candidate_agents, structure_prompt, optimized_prompts):
    prefix_string = structure_prompt + "\n\nHere is the current agent whose role is: {}\n".format(current_agent)
    prefix_string += "Its function description is:\n{}".format(get_agent_description(current_agent, optimized_prompts))

    roles_description = "\nThese are the candidate agents and their functional descriptions: "
    for rl_num, role in enumerate(candidate_agents):
        roles_description += "\nAgent {}: Its name is: {}\n".format(rl_num+1, role)
        roles_description += "Its function description is:\n{}".format(get_agent_description(role, optimized_prompts))
    return prefix_string + roles_description + '\n' + STRUCTURE_FORMAT
