import importlib
import itertools
import json
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
REAL_API_MODEL = os.getenv("OMAC_REAL_API_MODEL", "gpt-4o-mini")
REAL_API_MAX_TOKENS = int(os.getenv("OMAC_REAL_API_MAX_TOKENS", "128"))
BENCHMARKS = ["MMLU", "HumanEval", "MATH"]


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


def load_config(bench_dir):
    return json.loads((bench_dir / "agent_config.json").read_text())


def all_points(config):
    return config["candidate_agents"] + config["candidate_controllers"]


def sequence_to_cli(sequence):
    if any("_" in point for point in sequence):
        return ",".join(sequence)
    return "_".join(sequence)


def initialize_prompt(prompt, benchmark, point, config, initialize_num):
    if benchmark == "MATH":
        prompt.initialize(point, initialize_num, config.get("base_roles", ["System"] * 4))
    else:
        prompt.initialize(point, initialize_num)


def remember_optimized_prompt(prompt, benchmark, point):
    if benchmark == "MATH":
        prompt.optimized_prompts[point] = prompt.id2prompt[0]
    else:
        prompt.optimized_prompts[point] = [prompt.role_names[0], prompt.id2prompt[0]]


def assert_generated_prompt(testcase, prompt, benchmark, point):
    testcase.assertGreaterEqual(
        len(prompt.new_prompts),
        2,
        f"{benchmark}/{point} did not append a generated prompt.",
    )
    testcase.assertIn(1, prompt.id2prompt)
    generated_prompt = prompt.id2prompt[1]
    testcase.assertTrue(generated_prompt, f"{benchmark}/{point} generated an empty prompt.")
    generated_bundle = generated_prompt if isinstance(generated_prompt, list) else [generated_prompt]
    testcase.assertTrue(
        all(isinstance(item, str) and item.strip() for item in generated_bundle),
        f"{benchmark}/{point} generated a malformed prompt bundle.",
    )


class OptimizationSettingMatrixTests(unittest.TestCase):
    def test_every_single_and_ordered_pair_setting_initializes_offline(self):
        for benchmark in BENCHMARKS:
            with self.subTest(benchmark=benchmark), benchmark_import_context(benchmark) as bench_dir:
                config = load_config(bench_dir)
                points = all_points(config)
                prompt_module = importlib.import_module("prompt_iteration.prompt")

                sequences = [(point,) for point in points]
                sequences.extend(itertools.permutations(points, 2))
                self.assertEqual(len(sequences), len(points) + len(points) * (len(points) - 1))

                if benchmark == "MATH":
                    run_evol = importlib.import_module("run_evol")
                    for sequence in sequences:
                        parsed = run_evol.parse_optimization_points(sequence_to_cli(sequence), points)
                        self.assertEqual(parsed, list(sequence))

                for sequence in sequences:
                    prompt = prompt_module.Prompt("offline-model", benchmark.lower())
                    for point in sequence:
                        initialize_prompt(prompt, benchmark, point, config, initialize_num=1)
                        self.assertGreater(prompt.new_prompts, [])
                        remember_optimized_prompt(prompt, benchmark, point)


@unittest.skipUnless(
    os.getenv("RUN_REAL_API_TESTS") == "1" and bool(os.getenv("OPENAI_API_KEY")),
    "Set RUN_REAL_API_TESTS=1 and OPENAI_API_KEY to run real OpenAI API integration tests.",
)
class RealApiPromptEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import openai

        openai.api_key = os.environ["OPENAI_API_KEY"]
        try:
            response = openai.ChatCompletion.create(
                model=REAL_API_MODEL,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                temperature=0,
                max_tokens=4,
            )
        except cls._connectivity_errors(openai) as exc:
            raise unittest.SkipTest(f"OpenAI API is not reachable from this environment: {exc}")
        cls.reachability_content = response["choices"][0]["message"]["content"].strip().lower()

    @staticmethod
    def _connectivity_errors(openai):
        names = ("APIConnectionError", "Timeout", "ServiceUnavailableError")
        return tuple(
            error_type
            for name in names
            for error_type in [getattr(openai.error, name, None)]
            if error_type is not None
        )

    def test_real_openai_chat_completion_is_reachable(self):
        self.assertIn("ok", self.reachability_content)

    def test_every_optimization_dimension_initializes_with_real_api(self):
        for benchmark in BENCHMARKS:
            with self.subTest(benchmark=benchmark), benchmark_import_context(benchmark) as bench_dir:
                config = load_config(bench_dir)
                prompt_lib = importlib.import_module("prompt_lib")
                chat_service = importlib.import_module("prompt_iteration.chat_service")
                prompt_module = importlib.import_module("prompt_iteration.prompt")

                prompt_lib.MAX_TOKENS = REAL_API_MAX_TOKENS
                prompt_lib.TEMPERATURE = 0
                chat_service.MAX_TOKENS = REAL_API_MAX_TOKENS
                chat_service.TEMPERATURE = 0

                for point in all_points(config):
                    with self.subTest(benchmark=benchmark, point=point):
                        prompt = prompt_module.Prompt(REAL_API_MODEL, benchmark.lower())
                        initialize_prompt(prompt, benchmark, point, config, initialize_num=2)
                        assert_generated_prompt(self, prompt, benchmark, point)

    @unittest.skipUnless(
        os.getenv("RUN_REAL_API_PAIR_MATRIX") == "1",
        "Set RUN_REAL_API_PAIR_MATRIX=1 to run the exhaustive real-API ordered-pair matrix.",
    )
    def test_every_single_and_ordered_pair_setting_initializes_with_real_api(self):
        for benchmark in BENCHMARKS:
            with self.subTest(benchmark=benchmark), benchmark_import_context(benchmark) as bench_dir:
                config = load_config(bench_dir)
                prompt_lib = importlib.import_module("prompt_lib")
                chat_service = importlib.import_module("prompt_iteration.chat_service")
                prompt_module = importlib.import_module("prompt_iteration.prompt")

                prompt_lib.MAX_TOKENS = REAL_API_MAX_TOKENS
                prompt_lib.TEMPERATURE = 0
                chat_service.MAX_TOKENS = REAL_API_MAX_TOKENS
                chat_service.TEMPERATURE = 0

                points = all_points(config)
                sequences = [(point,) for point in points]
                sequences.extend(itertools.permutations(points, 2))

                for sequence in sequences:
                    with self.subTest(benchmark=benchmark, sequence=sequence):
                        prompt = prompt_module.Prompt(REAL_API_MODEL, benchmark.lower())
                        for point in sequence:
                            initialize_prompt(prompt, benchmark, point, config, initialize_num=2)
                            assert_generated_prompt(self, prompt, benchmark, point)
                            remember_optimized_prompt(prompt, benchmark, point)


if __name__ == "__main__":
    unittest.main()
