# OMAC: Optimization for Multi-Agent Collaboration

## Overview

This repository provides OMAC experiments for three benchmarks: MMLU, HumanEval, and MATH. OMAC defines five optimization points for LLM-based multi-agent collaboration:

1. **Fun-1**: Optimize existing candidate agents, such as instruction prompts and context examples.
2. **Fun-2**: Optimize an LLM to construct new agents for collaboration.
3. **Str-1**: Use an LLM controller to choose candidate agents before collaboration.
4. **Str-2**: Use an LLM controller to select participating agents during a collaboration step.
5. **Str-3**: Use an LLM controller to decide how agents communicate with each other in each step.

The HumanEval implementation follows the POMA-style parameter assignment and collaboration-control design while retaining the original OMAC Str-3 structure controller. The same configuration pattern is applied to MMLU and MATH.

## Structure

- `code/requirements.txt`: shared Python dependencies.
- `code/MMLU`: OMAC on MMLU multiple-choice reasoning.
- `code/HumanEval`: OMAC on HumanEval code generation.
- `code/MATH`: OMAC on competition math problems.
- `data/mmlu`: bundled downsampled MMLU data.
- `data/MATH`: bundled MATH test data.

HumanEval expects the `human_eval` package/data to be installed, and MATH expects the MATH dataset under `data/MATH/test`. If those data folders are not present, copy or install them before running the corresponding benchmark.

## Installation

```bash
conda create -n OMAC python=3.9
conda activate OMAC
cd code
pip install -r requirements.txt
```

For HumanEval, also install the HumanEval package if using a local copy:

```bash
cd ../data
pip install -e human-eval
```

Set `OPENAI_API_KEY` in the environment, or fill it in the relevant `run_evol.py` / listwise script.

## Running

All three benchmark folders expose the same high-level interface through `run_evol.py`. Run commands from the benchmark directory and pass an ordered optimization sequence with `--optimization_points`.

```bash
cd code/<BENCHMARK>
python -u run_evol.py --optimization_points <POINT_1>_<POINT_2>_...
```

The sequence order means that earlier optimized prompts are retained and used when optimizing later points. For example, on HumanEval:

```bash
cd code/HumanEval
python -u run_evol.py --optimization_points ComputerScientist_rank_structure
```

This optimizes `ComputerScientist`, then optimizes `rank` using the optimized `ComputerScientist`, then optimizes `structure` using both earlier results.

Common examples:

```bash
cd code/MMLU
python -u run_evol.py --optimization_points Mathematician_rank

cd ../HumanEval
python -u run_evol.py --optimization_points ComputerScientist_rank_structure

cd ../MATH
python -u run_evol.py --optimization_points System_rank
```


Useful shared runtime arguments:

- `--initialize_num`: number of Semantic Initializer prompts.
- `--model_name`: OpenAI model used for prompt evolution.
- `--FD_max_iter`: maximum contrastive feedback-mutation steps.
- `--FD_min_improvement`: minimum score improvement before tolerance increases.
- `--FD_tolerance_iter`: tolerated low-improvement mutation steps.
- `--FD_sample_threshold`: prompt sampling threshold for parent selection.
- `--Iter_num`: repeated optimization passes over the sequence.

OpenAI API calls use a small retry loop for transient API/network errors. Tune it with `OMAC_OPENAI_MAX_RETRIES` and `OMAC_OPENAI_RETRY_SLEEP` if your run environment is noisy.


## Configuration

Each benchmark has an `agent_config.json` file controlling the collaboration space:

- `candidate_agents`: optimizable existing agents or prompt components.
- `candidate_controllers`: `construct-role`, `pre-rank`, `rank`, and `structure`.
- `type_to_agents`: maps collaboration layer types to available agents.
- `layer_type_list`: layer-by-layer collaboration structure.

The same ideas apply to all benchmarks. Note that we use the same default agent collaboration setup before optimization as DyLAN:

- **MMLU** uses one `Reasoner` layer type containing all subject-matter agents.
- **HumanEval** uses alternating `Coder` and `Verifier` layer types.
- **MATH** uses a `Solver` setup with `base_roles` because its agents are prompt bundles rather than named personas.

HumanEval is a concrete example:

```json
{
  "candidate_agents": ["PythonAssistant", "AlgorithmDeveloper", "ComputerScientist", "Programmer", "Passer", "Tester", "Reflector"],
  "candidate_controllers": ["construct-role", "pre-rank", "rank", "structure"],
  "type_to_agents": {
    "Coder": ["PythonAssistant", "AlgorithmDeveloper", "ComputerScientist", "Programmer"],
    "Verifier": ["Reflector", "Tester", "Passer", "Ranker"]
  },
  "layer_type_list": ["Coder", "Verifier", "Coder", "Verifier", "Coder"]
}
```

In that configuration, the first layer generates code, the second layer evaluates it, and later layers refine it.

## Optimization Points

- **MMLU Fun-1**: `Economist`, `Doctor`, `Lawyer`, `Mathematician`, `Psychologist`, `Programmer`, `Historian`
- **HumanEval Fun-1**: `PythonAssistant`, `AlgorithmDeveloper`, `ComputerScientist`, `Programmer`, `Passer`, `Tester`, `Reflector`
- **MATH Fun-1**: `System`, `Examples`
- **Shared controllers**: `construct-role`, `pre-rank`, `rank`, `structure`

The order in `--optimization_points` matters. For example, `ComputerScientist_rank_structure` first optimizes `ComputerScientist`, then optimizes the rank controller with that prompt retained, then optimizes the communication-structure controller with both prior optimizations retained.

## Acknowledgments

This project incorporates code from the [DyLAN package](https://github.com/SALT-NLP/DyLAN), originally licensed under the MIT License. Significant modifications have been made to adapt it to OMAC's goals and architecture.
