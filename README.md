# Math Modal fine-tuning project

A small Python project for GSM8K fine-tuning, inference, evaluation, and a local
FastAPI service. Training and inference use a 4-bit Mistral model with LoRA
adapters.

## Project structure

- `src/math_modal/config.py` — model, data, training, and evaluation defaults.
- `src/math_modal/data.py` — GSM8K loading, subset selection, and prompt formatting.
- `src/math_modal/model.py` — quantization, model/tokenizer setup, and adapter saving.
- `src/math_modal/train.py` — supervised fine-tuning workflow.
- `src/math_modal/inference.py` — shared prompt and generation implementation.
- `src/math_modal/evaluation.py` — base/adapter comparison, scoring, and reports.
- `src/math_modal/api.py` — FastAPI health and inference endpoints.
- `src/math_modal/cli.py` — training, inference, and evaluation commands.
- `Colab_Quickstart.ipynb` — hosted Colab setup and CLI walkthrough.

## Environment

Use Python 3.10 or newer. Install the package and dependencies in an environment
that matches your accelerator:

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python -m pip install -e .
```

Training and inference use PyTorch, Transformers, TRL, PEFT, Accelerate, and
bitsandbytes 4-bit quantization. Dependency versions are not pinned because a
resolved environment was not available. The only declared version constraint
is `bitsandbytes>=0.46.1`. Select PyTorch and bitsandbytes builds compatible
with the host's NVIDIA driver, CUDA runtime, and GPU. GPU and package
compatibility have not been verified here. `torchao` is listed to match the
environment setup; the application does not call its API.

## Training

```sh
math-modal train
# Optional path/model overrides:
math-modal train --model-id 4bit/Mistral-7B-v0.2-hf --output-dir ./openai-gsm8k
```

Defaults: GSM8K `main`, `train[:2000]`, and a 10% train/test split using the
dataset library's default randomness (no seed supplied). The prompt is exactly
`### Instruction:\n{question}\n\n### Response:\n{answer}</s>`. The model is
`4bit/Mistral-7B-v0.2-hf`, loaded with 4-bit NF4, float16 compute, and double
quantization. Training casts any bfloat16 parameters to float16 and sets the
tokenizer EOS token as padding with right padding. LoRA uses rank 16, alpha 32,
target modules `q_proj`, `v_proj`, `k_proj`, and `o_proj`, dropout 0.05,
`bias="none"`, and `CAUSAL_LM`. SFT uses per-device batch size 2, gradient
accumulation 4, learning rate 0.0002, one epoch, logging every 10 steps,
epoch checkpoints, maximum length 512, `fp16=False`, `bf16=False`, dropped last
batches, and `adamw_torch`.

The adapter and tokenizer are saved to `./openai-gsm8k` by default, with
checkpoints saved at each epoch. To store outputs in Google Drive from Colab,
explicitly mount Drive with `math_modal.colab.mount_drive()` and pass
`/content/drive/MyDrive/openai-gsm8k` as `--output-dir`.

For a hosted Colab workflow, open [`Colab_Quickstart.ipynb`](Colab_Quickstart.ipynb).
It clones the public project, installs dependencies, and runs the same CLI
commands shown here. The setup cell asks for your public GitHub clone URL.

## Inference

```sh
math-modal infer --model-path ./openai-gsm8k --question "Find the value of x if 3x + 7 = 22." --max-new-tokens 50
```

The prompt is `### Instruction:\n{question}\n\n### Response:\n`. The CLI
requires an explicit token limit; the included example configurations use 50,
500, and 1000 tokens. Sampling is disabled unless `--do-sample` is passed.
The text-generation pipeline's `generated_text` is returned unchanged.

## Evaluation

```sh
math-modal evaluate --model-path ./openai-gsm8k
```

Evaluation uses the GSM8K `main` test split, assigns dataset test indices,
shuffles with seed 42, selects up to 200 examples, and uses greedy generation
with 512 new tokens, one beam, one return sequence, cache enabled, and EOS as
pad/EOS. It compares the adapter disabled and enabled on the same loaded
quantized model. Results are written under a timestamped
`gsm8k_evaluation/` directory as JSONL, CSV, and JSON files. Configure the
sample size, seed, token limit, and output directory with `--n-examples`,
`--seed`, `--max-new-tokens`, and `--run-dir`.

Strict scoring requires a numeric answer on the line after `####`; relaxed
scoring falls back to the last numeric value when the marker is absent. This
fallback can count an intermediate value as an answer.

## Local API

```sh
MODEL_PATH=./openai-gsm8k uvicorn math_modal.api:app --host 127.0.0.1 --port 8000
```

On PowerShell, set the environment variable with `$env:MODEL_PATH = '.\openai-gsm8k'`
before running Uvicorn. `GET /health` reports whether startup loaded the model.
`POST /infer` accepts `{"question":"...","max_new_tokens":50,"do_sample":false}`
and returns `{"generated_text":"..."}`. `max_new_tokens` defaults to 50 and
is limited to 1–4096 at request validation. Model initialization happens once
during application startup; the endpoint uses the same `generate` function as
the CLI.

## Docker

Build the image with `docker build -t math-modal .`. Run it on a host with
Docker's NVIDIA Container Toolkit, a compatible NVIDIA GPU/driver, and a local
adapter directory:

```sh
docker run --rm --gpus all -p 8000:8000 \
  -e MODEL_PATH=/models/openai-gsm8k \
  -v "$PWD/openai-gsm8k:/models/openai-gsm8k:ro" math-modal
```

The image contains application code and Python dependencies only; model weights
and credentials are mounted or supplied at runtime. The Docker base image does
not establish a verified CUDA/PyTorch compatibility matrix. Validate package,
driver, and GPU compatibility on the target host before use.

## Known limitations

- Dependency versions are not pinned; package compatibility depends on the
  selected Python, Transformers, PEFT, TRL, PyTorch, and bitsandbytes versions.
- Loading an adapter directory with `AutoModelForCausalLM.from_pretrained`
  depends on the installed Transformers PEFT integration and has not been
  verified in this environment.
- Training split randomness is left to `datasets.train_test_split`; no seed is
  passed.
- The relaxed answer parser uses the final number when `####` is absent, which
  may select an intermediate value.
- Full training, model downloads, GPU loading, API startup, and benchmark
  execution were not performed as part of local verification.
