import unittest
from dataclasses import asdict
from pathlib import Path
import sys
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_modal.config import DEFAULT_TRAINING, INFERENCE_EXAMPLES, EvaluationConfig
from math_modal.data import format_prompts, load_evaluation_data, load_training_data
from math_modal.evaluation import adapter_mode, extract_answer, normalize_number
from math_modal.inference import format_inference_prompt, generate
from math_modal.model import load_inference_pipeline, save_adapter


class WorkflowTests(unittest.TestCase):
    def test_training_prompt_exact_whitespace_and_special_token(self):
        result = format_prompts({"question": ["Q?"], "answer": ["A"]})
        self.assertEqual(result["text"], ["### Instruction:\nQ?\n\n### Response:\nA</s>"])

    def test_inference_prompt_and_generation_arguments(self):
        prompt = "### Instruction:\nQuestion?\n\n### Response:\n"
        self.assertEqual(format_inference_prompt("Question?"), prompt)
        calls = []
        def fake_generator(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"generated_text": prompt + "answer"}]
        self.assertEqual(generate(fake_generator, "Question?", 500), prompt + "answer")
        self.assertEqual(calls, [((prompt,), {"max_new_tokens": 500, "do_sample": False})])

    def test_notebook_configuration(self):
        config = asdict(DEFAULT_TRAINING)
        self.assertEqual(config["model_id"], "4bit/Mistral-7B-v0.2-hf")
        self.assertEqual((config["dataset_name"], config["dataset_config"], config["train_split"]),
                         ("openai/gsm8k", "main", "train[:2000]"))
        self.assertEqual(config["test_size"], 0.1)
        self.assertEqual((config["load_in_4bit"], config["bnb_4bit_quant_type"],
                          config["bnb_4bit_compute_dtype"], config["bnb_4bit_use_double_quant"]),
                         (True, "nf4", "float16", True))
        self.assertEqual((config["lora_r"], config["lora_alpha"], config["target_modules"],
                          config["lora_dropout"], config["lora_bias"]),
                         (16, 32, ("q_proj", "v_proj", "k_proj", "o_proj"), 0.05, "none"))
        self.assertEqual((config["per_device_train_batch_size"], config["gradient_accumulation_steps"],
                          config["learning_rate"], config["max_length"], config["optim"]),
                         (2, 4, 2e-4, 512, "adamw_torch"))
        self.assertEqual([item.max_new_tokens for item in INFERENCE_EXAMPLES.values()], [50, 500, 1000])

    def test_answer_parser_edge_cases(self):
        self.assertEqual(normalize_number("1,200.0"), "1200")
        self.assertEqual(extract_answer("Solution.\n#### 1,200.0"), "1200")
        self.assertEqual(extract_answer("#### 1/2"), "1/2")
        self.assertEqual(extract_answer("The answer is 72.", relaxed=True), "72")
        self.assertIsNone(extract_answer("The answer is 72."))
        self.assertIsNone(extract_answer("#### 1/0"))
        self.assertEqual(extract_answer("#### -2\n### Instruction:\n#### 4"), "-2")

    def test_dataset_subset_split_and_evaluation_selection_arguments(self):
        calls = []

        class TrainingDataset:
            def map(self, function, batched):
                calls.append(("map", function, batched))
                return self
            def train_test_split(self, **kwargs):
                calls.append(("split", kwargs))
                return {"train": [], "test": []}

        class EvaluationDataset:
            def __len__(self): return 321
            def add_column(self, name, values):
                calls.append(("add_column", name, len(values)))
                return self
            def shuffle(self, **kwargs):
                calls.append(("shuffle", kwargs))
                return self
            def select(self, selected):
                calls.append(("select", list(selected)))
                return self

        def fake_load_dataset(name, config, split):
            calls.append(("load", name, config, split))
            return TrainingDataset() if split == "train[:2000]" else EvaluationDataset()

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            load_training_data(DEFAULT_TRAINING)
            evaluation_config = EvaluationConfig()
            load_evaluation_data(evaluation_config)
        self.assertIn(("load", "openai/gsm8k", "main", "train[:2000]"), calls)
        self.assertIn(("split", {"test_size": 0.1}), calls)
        self.assertIn(("load", "openai/gsm8k", "main", "test"), calls)
        self.assertIn(("shuffle", {"seed": 42}), calls)
        self.assertIn(("select", list(range(200))), calls)

    def test_adapter_activation_context_restores_layers(self):
        events = []
        class Model:
            def disable_adapters(self): events.append("disable")
            def enable_adapters(self): events.append("enable")
        with adapter_mode(Model(), use_adapter=False):
            events.append("inference")
        self.assertEqual(events, ["disable", "inference", "enable"])

    def test_inference_loader_quantization_path_and_adapter_save_calls(self):
        calls = []
        class FakeBitsConfig:
            def __init__(self, **kwargs): self.values = kwargs
        class FakeAutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append(("model", path, kwargs))
                return "loaded-model"
        class FakeTokenizer:
            @staticmethod
            def from_pretrained(path):
                calls.append(("tokenizer", path))
                return "loaded-tokenizer"
        def fake_pipeline(*args, **kwargs):
            calls.append(("pipeline", args, kwargs))
            return "generator"
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BitsAndBytesConfig = FakeBitsConfig
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeTokenizer
        fake_transformers.pipeline = fake_pipeline
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            self.assertEqual(load_inference_pipeline("./openai-gsm8k", DEFAULT_TRAINING), "generator")
        model_call = next(call for call in calls if call[0] == "model")
        self.assertEqual(model_call[1], "./openai-gsm8k")
        self.assertEqual(model_call[2]["device_map"], "auto")
        self.assertEqual(model_call[2]["torch_dtype"], "float16")
        self.assertEqual(model_call[2]["quantization_config"].values,
                         {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": "float16", "bnb_4bit_use_double_quant": True})
        class Saveable:
            def save_pretrained(self, path): calls.append(("save", path))
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            save_adapter(Saveable(), Saveable(), Path(folder) / "adapter")
        self.assertEqual([call[0] for call in calls if call[0] == "save"], ["save", "save"])


if __name__ == "__main__":
    unittest.main()
