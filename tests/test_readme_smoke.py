import importlib.util
import json
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SHARED_CONTROLLERS = {"construct-role", "pre-rank", "rank", "structure"}


@contextmanager
def benchmark_import_context(name):
    bench_dir = CODE / name
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    for module_name in list(sys.modules):
        if module_name in {
            "CoLLMLP",
            "LLMLP",
            "LLM_Agent",
            "LLM_Neuron",
            "listwise_human_eval",
            "listwise_math",
            "listwise_mmlu",
            "prompt_lib",
            "run_evol",
            "utils",
            "utils_evo",
        } or module_name.startswith("prompt_iteration"):
            del sys.modules[module_name]
    os.chdir(bench_dir)
    sys.path.insert(0, str(bench_dir))
    try:
        yield bench_dir
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadmeSmokeTests(unittest.TestCase):
    def test_benchmark_data_is_available(self):
        mmlu_train = ROOT / "data" / "mmlu" / "data" / "downsampled" / "train"
        mmlu_test = ROOT / "data" / "mmlu" / "data" / "downsampled" / "test"
        math_test = ROOT / "data" / "MATH" / "test"

        self.assertTrue(any(mmlu_train.glob("*.csv")), "MMLU train CSVs are missing")
        self.assertTrue(any(mmlu_test.glob("*.csv")), "MMLU test CSVs are missing")
        self.assertTrue(any(math_test.rglob("*.json")), "MATH test JSON files are missing")

    def test_agent_configs_expose_shared_optimization_points(self):
        for name in ["MMLU", "HumanEval", "MATH"]:
            with self.subTest(benchmark=name):
                config_path = CODE / name / "agent_config.json"
                config = json.loads(config_path.read_text())
                self.assertTrue(SHARED_CONTROLLERS.issubset(config["candidate_controllers"]))
                self.assertEqual(len(config["layer_type_list"]), len(config["layer_max_agents"]))
                self.assertGreater(len(config["candidate_agents"]), 0)

    def test_readme_cli_entrypoints_import(self):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in ["MMLU", "HumanEval", "MATH"]:
            with self.subTest(benchmark=name):
                result = subprocess.run(
                    [sys.executable, "run_evol.py", "--help"],
                    cwd=CODE / name,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_mmlu_and_humaneval_collaboration_graphs_build_offline(self):
        with benchmark_import_context("MMLU") as bench_dir:
            module = load_module("mmlu_llmlp", bench_dir / "LLMLP.py")
            roles = json.loads((bench_dir / "agent_config.json").read_text())["candidate_agents"]
            model = module.LLMLP(
                "gpt-3.5-turbo",
                len(roles),
                roles,
                3,
                "listwise",
                "single_choice",
                "gpt-3.5-turbo",
                ["Mathematician", "Mathematician", "test prompt"],
                {},
            )
            self.assertEqual(model.num_agents, [7, 7, 7])

        with benchmark_import_context("HumanEval") as bench_dir:
            module = load_module("humaneval_collmlp", bench_dir / "CoLLMLP.py")
            config = json.loads((bench_dir / "agent_config.json").read_text())
            coders = config["type_to_agents"]["Coder"]
            verifiers = config["type_to_agents"]["Verifier"]
            model = module.CoLLMLP(
                "gpt-3.5-turbo",
                len(coders),
                coders,
                len(verifiers),
                verifiers,
                3,
                "listwise",
                "code_completion",
                "gpt-3.5-turbo",
                ["ComputerScientist", "ComputerScientist", "test prompt"],
                {},
            )
            self.assertEqual(model.num_agents, [4, 4, 4, 4, 4])

    def test_math_optimization_point_parser_handles_examples_cot(self):
        with benchmark_import_context("MATH") as bench_dir:
            module = load_module("math_run_evol", bench_dir / "run_evol.py")
            config = json.loads((bench_dir / "agent_config.json").read_text())
            allowed = config["candidate_agents"] + config["candidate_controllers"]
            self.assertEqual(
                module.parse_optimization_points("Examples_cot,rank", allowed),
                ["Examples_cot", "rank"],
            )
            self.assertEqual(
                module.parse_optimization_points("System_rank_structure", allowed),
                ["System", "rank", "structure"],
            )


if __name__ == "__main__":
    unittest.main()
