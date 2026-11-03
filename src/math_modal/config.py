"""Notebook-derived defaults and configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str = "4bit/Mistral-7B-v0.2-hf"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    train_split: str = "train[:2000]"
    output_dir: str = "./openai-gsm8k"
    test_size: float = 0.1
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    torch_dtype: str = "float16"
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    task_type: str = "CAUSAL_LM"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    logging_steps: int = 10
    num_train_epochs: int = 1
    save_strategy: str = "epoch"
    dataset_text_field: str = "text"
    max_length: int = 512
    fp16: bool = False
    bf16: bool = False
    dataloader_drop_last: bool = True
    optim: str = "adamw_torch"


@dataclass(frozen=True)
class InferenceExample:
    question: str
    max_new_tokens: int
    do_sample: bool = False


INFERENCE_EXAMPLES = {
    "linear": InferenceExample("Find the value of x if 3x + 7 = 22.", 50),
    "chocolates": InferenceExample(
        "Mark has a box of 40 chocolates. He eats 5 chocolates, gives half of the remaining chocolates to his sister, and then buys 10 more chocolates the next day. How many chocolates does Mark have now?",
        500,
    ),
    "cosine": InferenceExample("\n### Instruction:\n\nWhat is the Cos of Pi/2\n### Response:\n\n", 1000),
}


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "test"
    n_examples: Optional[int] = 200
    seed: int = 42
    max_new_tokens: int = 512
    run_dir: Path = field(default_factory=lambda: Path("gsm8k_evaluation"))


DEFAULT_TRAINING = TrainingConfig()
