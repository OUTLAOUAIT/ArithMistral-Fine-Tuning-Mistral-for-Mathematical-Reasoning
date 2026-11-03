"""Shared pipeline based inference used by CLI and API."""
from .config import DEFAULT_TRAINING
from .model import load_inference_pipeline


def format_inference_prompt(question):
    return f"### Instruction:\n{question}\n\n### Response:\n"


def load_generator(model_path, config=DEFAULT_TRAINING):
    return load_inference_pipeline(model_path, config)


def generate(generator, question, max_new_tokens, do_sample=False):
    prompt = format_inference_prompt(question)
    results = generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return results[0]["generated_text"]
