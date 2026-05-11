import re
import random

TEMPERATURE = 0.8
MAX_TOKENS = 1024


SYSTEM_PROMPT = "It's a debate. Explain your reasons at each round thoroughly.\nFollow the given examples and answer the mathematics problem."
EXAMPLES_ORIG = """Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted. The answer is 6.
###
Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.
###
Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.
###
Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.
###
Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9.
###
Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = 20 computers were added. Now 9 + 20 = 29 computers are now in the server room. The answer is 29.
###
Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = 35 at the end of Tuesday, and 35 - 2 = 33 at the end of wednesday. The answer is 33.
###
Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 
Answer: Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = 15 dollars. Now she has 23 - 15 = 8 dollars left. The answer is 8."""
EXAMPLES_COT= r"""Problem: Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.
Answer: Let's think step by step
Kevin hops $1/3$ of the remaining distance with every hop.
His first hop takes $1/3$ closer.
For his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.
For his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.
In general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.
We want to find how far he has hopped after five hops.
This is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.
Thus, Kevin has hopped $\frac{\frac{1}{3}\left(1-\left(\frac{2}{3}\right)^5\right)}{1-\frac{2}{3}} = \boxed{\frac{211}{243}}$.
The answer is \frac{211}{243}

Problem: What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?
Answer: Let's think step by step
We rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,
resulting in  $(x+7)^2-49 + (y-2)^2-4=10$,
or $(x+7)^2+(y-2)^2=63$.
This is the equation of a circle with center $(-7, 2)$ and radius $\sqrt{63},$
so the area of this region is $\pi r^2 = \boxed{63\pi}$.
The answer is 63\pi

Problem: If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?
Answer: Let's think step by step
If $(x,y)$ lies on the circle,
so does $(x,-y),$ $(-x,y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),
so we can assume that $x \ge 0$ and $y \ge 0.$
Then $|x| + |y| = x + y.$  Squaring, we get
\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\]
Note that $(x - y)^2 \ge 0.$
Expanding, we get $x^2 - 2xy + y^2 \ge 0,$ so $2xy \le x^2 + y^2 = 1.$
Hence,\[1 + 2xy \le 2,\]which means $x + y \le \sqrt{2}.$
Equality occurs when $x = y = \frac{1}{\sqrt{2}},$
so the maximum value of $|x| + |y|$ is $\boxed{\sqrt{2}}.$
The answer is \sqrt{2}

Problem: If $f(x)=\frac{ax+b}{cx+d}, abcd\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?
Answer: Let's think step by step
The condition $f(f(x))$ means that $f$ is the inverse of itself,
so its graph is symmetrical about the line $y = x$.
With a rational function of this form, we will have two asymptotes:
a vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,
and a horizontal one at $y=a/c$,
if we take the limit of $f(x)$ as $x$ goes to $\pm\infty$.
In order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$
so that it and its asymptotes reflect onto themselves.
This means that $-d/c=a/c$,
and therefore $-d=a$ and $a+d=\boxed{0}$.
The answer is 0

Problem: A math teacher requires Noelle to do one homework assignment for each of the first five homework points she wants to earn; for each of the next five homework points, she needs to do two homework assignments; and so on, so that to earn the $n^{\text{th}}$ homework point, she has to do $n\div5$ (rounded up) homework assignments. For example, when she has 11 points, it will take $12\div5=2.4\rightarrow3$ homework assignments to earn her $12^{\text{th}}$ point. What is the smallest number of homework assignments necessary to earn a total of 25 homework points?
Answer: Let's think step by step
Noelle only has to do 1 homework assignment to earn her first point,
and the same is true for each of her first five points.
She must then do 2 homework assignments to earn her sixth point, seventh point, and so on, up to her tenth point.
Continuing, we see that Noelle must do a total of \[1+1+1+1+1+2+2+2+2+2+\dots+5+5+5+5+5\] homework assignments to earn 25 points.
This sum may be rewritten as $5(1+2+3+4+5)=5(15)=\boxed{75}$.
The answer is 75

Problem: The quadratic equation $x^2+mx+n=0$ has roots that are twice those of $x^2+px+m=0,$ and none of $m,$ $n,$ and $p$ is zero. What is the value of $n/p?$
Answer: Let's think step by step
Let $r_1$ and $r_2$ be the roots of $x^2+px+m=0.$
Since the roots of $x^2+mx+n=0$ are $2r_1$ and $2r_2,$ we have the following relationships: \[
m=r_1 r_2,\quad n=4r_1 r_2,\quad p=-(r_1+r_2), \quad\text{and}\quad
m=-2(r_1+r_2).
\] So \[
n = 4m, \quad p = \frac{1}{2}m,
\quad\text{and}\quad
\frac{n}{p}=\frac{4m}{\frac{1}{2}m}=\boxed{8}.
\]
Alternatively, the roots of \[
\left(\frac{x}{2}\right)^2 + p\left(\frac{x}{2}\right) + m = 0
\] are twice those of $x^2 + px + m = 0.$
Since the first equation is equivalent to $x^2 + 2px + 4m = 0,$
we have \[m = 2p \quad\text{and}\quad n = 4m, \quad\text{so}\quad \frac{n}{p} = \boxed{8}.\]
The answer is 8

Problem: Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.
Answer: Let's think step by step
$$\begin{array}{crrrrrrr}
& & & 3z^3 & & -2z & + 1 & \\
\times & & & & 2z^2 & +5z & -6 \\
\cline{1-7}\rule{0pt}{0.17in}
& & & -18z^3 & & +12z & -6 & \\
& & +15z^4 & & -10z^2 & +5z & & \\
+ & 6z^5 & & -4z^3 & +2z^2 & & & \\
\cline{1-7}\rule{0pt}{0.17in}
& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &
\end{array}$$
The answer is 6z^5+15z^4-22z^3-8z^2+17z-6.

Problem: Find the mean of all solutions for $x$ when $x^3 + 3x^2 - 10x = 0$.
Answer: Let's think step by step
First, we factor the equation as $x(x^2 +3x - 10) = 0$.
So, one solution is $x=0$ and the other two solutions are the solutions to $x^2 + 3x-10=0$.
We could either factor the quadratic, or note that the sum of the solutions to this quadratic is $-(3/1)=-3$,
so the mean of the three solutions to the original equation is $-3/3=\boxed{-1}$.
The answer is -1"""



RANK_DEFAULT = "Please choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."

RANK_FORMAT = "Please ensure to specify the choice of solutions strictly in the format of a single list as described above (like [1,2] or [1,2,3] or [1,2,3,4]). The elements of this list should represent the sequence numbers of all selected solutions. Avoid expressing your choice in the format like '[function impl 1] and [function impl 2]'. Also, ensure that anything within '[]' consists solely of sequence numbers of solutions, separated by commas, with no additional text like 'function impl'. Note that the maximum number of solutions you can choose is {}."

RANK_INIT = """Create {} different prompts for an LLM to choose some top solutions to resolve general mathematics problems best. 
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
You can decide the number of the chosen solutions and the content of the prompt.
The prompt should help to accurately and efficiently select the top solutions that resolve the mathematics problem best.
Note that all the solutions were previously provided as the context. The prompt generated here will be added to the context to form the final prompt for solution selection.
You may consider adding more detailed and thorough instructions to help the LLM select the top solutions better.
The generated prompt should specify the output format like the given example (also ensure that it is different from the example prompt):
"{}"."""

RANK_FD = """Create and output a child prompt for an LLM to choose some top solutions to resolve general mathematics problems best.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to select more useful and effective solutions to resolve mathematics problems.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose top solutions more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

# "It's a debate. Explain your reasons at each round thoroughly.\n Follow the given examples and answer the mathematics problem."
SYSTEM_INIT = """Generate {} distinct prompts, each as a system prompt to instruct an LLM to give answers to general mathematics problems during a multi-turn debate.
Each prompt should guide the LLM to accurately and efficiently analyze any given mathematical problem and thoroughly reason through it. At the end, the prompt should instruct the LLM to provide the correct answer to the problem.
Do not output all the prompts at once. Instead, I will provide a sequence number, and you should return only the corresponding prompt one by one. 
You should consider adding more detailed and thorough instructions to help the LLM reason, analyze, and finally resolve the problem.
Ensure that the prompts are not for a specific mathematics problem. Instead, they should be general enough to apply to all possible mathematics problems that will be provided during inference.
Do not create any specific instances of the mathematics problems in the prompt, because they are not provided now.
The following is a given example prompt (also ensure your output is different from the example prompt):
"{}"."""

SYSTEM_FD = """Generate and output a child prompt for an LLM to give answers to general mathematics problems during a multi-turn debate.
At the end, a pair of parent prompts is provided: one positive and one negative. The positive parent prompt has been shown to be more effective and efficient in guiding the LLM to generate high-quality answers.
Your task is to carefully compare the two parent prompts, identifying the key reasons why the positive parent prompt performs better. Based on these insights, generate and output a child prompt that further improves upon the positive parent prompt to enhance answer quality.
Ensure that the prompts are not for a specific mathematics problem. Instead, they should be general enough to apply to all possible mathematics problems that will be provided during inference.
Do not create any instances of the mathematics problems in the prompt, because they are not provided now.
The child prompt should be distinct from both. Output only the content of the child prompt."""
# Ensure that the prompts remain general enough to mathematics problems given during inference. 
# The child prompt must follow the format of the parent prompts but should be distinct from both.

EXAMPLES_INIT = """Generate {} distinct example sets to help an LLM learn how to give answers to some unknown mathematics problems given during inference. Every example in each example set should comprise a problem and its answer. Each example set should contain 8 to 12 examples.
The generated example sets should help the model learn how to accurately and efficiently reason, analyze, and resolve the mathematics problems.
The examples in each example set should cover a wide range of mathematics problems, including but not limited to algebra, counting and probability, geometry, number theory, intermediate algebra, prealgebra, and precalculus problems.
Do not output all the example sets at once. Instead, I will provide a sequence number, and you should return only the corresponding example set. 
Ensure that the generated example sets follow the format of the following example set but differ from the example itself. You may consider creating more diverse, exemplary, and general problems and their answers. The given instance of an example set is as follows:
"{}"."""

EXAMPLES_FD = """Generate and output a child example set to help an LLM learn how to give answers to some unknown mathematics problems given during inference. Every example in each example set should comprise a problem and its answer. Each example set should contain 8 to 12 examples.
At the end, a pair of parent example sets is provided: one positive and one negative. The positive parent example set has been shown to be more effective and efficient in guiding the LLM to generate high-quality answers to mathematics problems.
Your task is to carefully compare the two parent example sets, identifying the key reasons why the positive parent example set performs better. Based on these insights, generate and output a child example set that further improves upon the positive parent example set to enhance answer quality.
The examples in each example set should cover a wide range of mathematics problems, including but not limited to algebra, counting and probability, geometry, number theory, intermediate algebra, prealgebra, and precalculus problems.
The child example set must follow the format of the parent example sets but should be distinct from both. Output only the content of the child example set."""

PRE_RANK_DEFAULT = "Take functionality, efficiency, and necessity into consideration, choose top 3 agents best suited to resolve the given mathematics problem. Think it step by step. Put your answer in the form like [1,2,3] or [1,3,4] at the end of your response."
    
PRE_RANK_FORMAT = "Please ensure to specify the choice of agents strictly in the format of a single list as described above (like [1,2] or [1,2,3] or [1,2,3,4]). The elements of this list should represent the sequence numbers of all selected agents. Avoid expressing your choice in the format like '[agent 1] and [agent 2]'. Also, ensure that anything within '[]' consists solely of sequence numbers of agents, separated by commas, with no additional text like 'agent'."

PRE_RANK_INIT = """Create {} distinct prompts for an LLM to choose some top agents best suited to collaboratively resolve some unknown mathematics problems. 
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should decide and specify the number of the chosen agents. The number should be between 2 and 4.
Each prompt should help to accurately and efficiently identify the top agents that resolve mathematics problems best.
Note that all information about the candidate agents has been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for agent determination.
You may consider adding more detailed and thorough instructions to help the LLM select the top agents better.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

PRE_RANK_FD = """Create and output a child prompt for an LLM to choose some top agents to collaboratively resolve some unknown mathematics problems.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to decide the top agents required for resolving mathematics problems best.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose the top agents more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

STRUCTURE_DEFAULT = "Take functionality, efficiency, and necessity into consideration. Select top 3 candidate agents whose generated solutions to some mathematics problems may be useful as inputs for the current agent to produce improved solutions. Think it step by step. Put your answer in the form like [1,2,3] or [1,3,4] at the end of your response."

STRUCTURE_FORMAT = "Please ensure to specify the choice of agents strictly in the format of a single list as described above (like [1,2] or [1,2,3] or [1,2,3,4]). The elements of this list should represent the sequence numbers of all selected agents. Avoid expressing your choice in the format like '[agent 1] and [agent 2]'. Also, ensure that anything within '[]' consists solely of sequence numbers of agents, separated by commas, with no additional text like 'agent'."

STRUCTURE_INIT = """Create {} distinct prompts for an LLM to choose some candidate agents whose generated solutions to some mathematics problems may be useful as inputs for the current agent to produce improved solutions. 
You should decide the number of the chosen agents and the content of the prompt. The number of chosen agents should be between 2 and 4.
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should help to accurately and efficiently identify the top candidate agents whose generated solutions are helpful to be taken as the input for the current agent.
Note that all information about the candidate agents and the current agent has been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for agent selection.
You may consider adding more detailed and thorough instructions to help the LLM select the candidate agents better.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

STRUCTURE_FD = """Create and output a child prompt for an LLM to choose some candidate agents whose generated solutions to some mathematics problems may be useful as inputs for the current agent to produce improved solutions. 
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to select more useful and effective agents.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to choose top agents more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""

def construct_prerank_message(base_roles, optimized_prompts, rank_prompt):
    roles_description = "The following are the system prompt and the examples provided for each agent (note all agents share the same system prompt and examples): "

    sys_pt = SYSTEM_PROMPT if 'System' not in optimized_prompts else optimized_prompts['System'][0]
    examples = EXAMPLES_ORIG if 'Examples_cot' not in base_roles else EXAMPLES_COT
    if 'Examples' in optimized_prompts:
        examples = optimized_prompts['Examples'][0]
    if 'Examples_cot' in optimized_prompts:
        examples = EXAMPLES_COT
    roles_description += "\nThe system prompt is: {}\n".format(sys_pt)
    roles_description += "\nThe provided examples are:\n{}\n".format(examples)
    return roles_description + '\n' + rank_prompt


def construct_structure_message(current_agent, agents_to_select, structure_prompt):
    prefix_string = structure_prompt + "\n\nHere is the system prompt of the current agent:\n{}\n".format(current_agent[0]["content"])

    roles_description = "\nThese are the candidate agents and their functional description: "
    for rl_num, agent in enumerate(agents_to_select):
        roles_description += "\n\nSystem prompt of agent {} is: {}\n".format(rl_num+1, agent[0]["content"])
    return prefix_string + roles_description + '\n' + STRUCTURE_FORMAT

def parse_ranks(completion, max_num=4, random_num=3):
    if not isinstance(completion, str):
        content = completion["choices"][0]["message"]["content"]
    else:
        content = completion
    # pattern = r'\[([1234]),\s*([1234])\]'
    pattern = r'\[([1234])(?:,\s*([1234]))?(?:,\s*([1234]))?(?:,\s*([1234]))?\]'
    matches = [tuple(filter(None, match)) for match in re.findall(pattern, content)]

    try:
        match = matches[-1]
        tops = [int(match[i]) for i in range(len(match))]
        def clip(x):
            if x < 2:
                return 2
            if x > max_num:
                return max_num
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        print("content: ", content)
        # tops is a randomly sampled number between 3 and 4
        tops = random.sample([random_num, max_num], 1)

    return tops[0]




MMLU_QUESTION = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} "



SYSTEM_PROMPT_MMLU = "Here's a debate. Explain your reasons at each round thoroughly.\nAll questions are single choice."
SYSTEM_PROMPT_HUMAN_EVAL_INIT = ""
'''Example 1:
[function impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a + b
```

Example 2:
[function impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```
END EXAMPLE'''



'''
PRE_RANK_DEFAULT = "Take functionality, efficiency, and necessity into consideration, decide how many agents are required to collaborate with each other to resolve some unknown mathematics problems. Think it through thoroughly and include your reasoning process. Put your answer in the form like [2] or [3] at the end of your response."

PRE_RANK_FORMAT = "Please ensure to specify the choice of number of agents strictly in the format of as described above (like [2] or [3]). The elements of this list should represent the number of agents. Avoid expressing your choice in the format like '3 agents'. Also, ensure that anything within '[]' consists solely of the number of agents, with no additional text like 'agents'."

PRE_RANK_INIT = """Create {} distinct prompts for an LLM to decide how many agents are required to collaboratively resolve some unknown mathematics problems. 
Don't directly output all the generated prompts. I will provide you with the sequence number of the prompt. Then you should directly output the content text of the corresponding prompt one by one.
Each prompt should decide and specify the number of the chosen agents. The number should be between 2 and 4.
Each prompt should help to accurately and efficiently identify the required number of agents that resolve mathematics problems best.
Note that all information about the candidate agents has been previously provided as the context. The prompt generated here will be added to the context to form the final prompt for agent number determination.
You may consider adding more detailed and thorough instructions to help the LLM select the agent number better.
The following is an example of a prompt (also ensure your output is different from the example prompt):
"{}"."""

PRE_RANK_FD = """Create and output a child prompt for an LLM to decide how many agents are required to collaboratively resolve some unknown mathematics problems.
I will provide you with a pair of parent prompts. Then you should only output a child prompt according to the following instructions:
The positive parent prompt is proven to be more helpful and efficient to instruct the LLM to decide the number of agents required for resolving mathematics problems best.
You should carefully compare the two parent prompts, finding the potential reasons why the positive parent prompt is better than the negative parent prompt.
Based on that, you should generate and output a child prompt that can help to decide the number of agents more effectively and efficiently than the positive parent prompt.
The child prompt should follow the format of the parent prompts.
The child prompt should be different from the parent prompts. Directly output the content text of the child prompt."""
'''
