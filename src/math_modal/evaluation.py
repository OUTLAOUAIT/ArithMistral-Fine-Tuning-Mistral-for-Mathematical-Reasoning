"""GSM8K base-versus-adapter evaluation from notebook cells 22–29."""
import json
import re
import time
from importlib.metadata import PackageNotFoundError, version
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction

DECIMAL = r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBER = rf"{DECIMAL}(?:\s*/\s*{DECIMAL})?"


def normalize_number(value):
    try:
        parts = value.replace(",", "").split("/")
        result = Fraction(parts[0].strip())
        if len(parts) == 2:
            result /= Fraction(parts[1].strip())
        return str(result)
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text, relaxed=False):
    text = text.split("### Instruction:")[0].replace("\u2212", "-")
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
        line = tail.splitlines()[0].strip() if tail else ""
        match = re.fullmatch(rf"\$?\s*({NUMBER})\s*[.!]?", line)
        return normalize_number(match.group(1)) if match else None
    if relaxed:
        matches = re.findall(NUMBER, text)
        return normalize_number(matches[-1]) if matches else None
    return None


@contextmanager
def adapter_mode(model, use_adapter):
    if use_adapter:
        yield
    elif getattr(model, "disable_adapter", None):
        with model.disable_adapter():
            yield
    else:
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()


def evaluate_model(model, tokenizer, dataset, label, use_adapter, config, run_dir):
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    generation = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    rows = []
    started = time.perf_counter()
    device = model.get_input_embeddings().weight.device
    output_path = run_dir / f"{label}_predictions.jsonl"
    with (adapter_mode(model, use_adapter), torch.inference_mode(),
          output_path.open("w", encoding="utf-8")) as output_file:
        for example in tqdm(dataset, desc=label):
            gold = extract_answer(example["answer"])
            if gold is None:
                raise ValueError(f"Unparseable reference: test_id={example['test_id']}")
            prompt = f"### Instruction:\n{example['question']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
            output = model.generate(**inputs, **generation)
            new_ids = output[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            strict = extract_answer(response)
            relaxed = extract_answer(response, relaxed=True)
            hit_limit = (new_ids.numel() >= config.max_new_tokens
                         and new_ids[-1].item() != tokenizer.eos_token_id)
            row = {
                "model": label, "test_id": example["test_id"],
                "question": example["question"], "reference_answer": example["answer"],
                "gold": gold, "prediction_strict": strict, "prediction_relaxed": relaxed,
                "correct_strict": strict is not None and strict == gold,
                "correct_relaxed": relaxed is not None and relaxed == gold,
                "unparsed_strict": strict is None, "hit_token_limit": bool(hit_limit),
                "generated_tokens": new_ids.numel(), "response": response,
            }
            rows.append(row)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
    frame = pd.DataFrame(rows)
    metrics = {
        "model": label, "n": len(frame),
        "strict_correct": int(frame["correct_strict"].sum()),
        "strict_accuracy_pct": 100 * frame["correct_strict"].mean(),
        "relaxed_accuracy_pct": 100 * frame["correct_relaxed"].mean(),
        "unparsed_strict": int(frame["unparsed_strict"].sum()),
        "hit_token_limit": int(frame["hit_token_limit"].sum()),
        "evaluation_minutes": (time.perf_counter() - started) / 60,
    }
    return frame, metrics


def evaluate(model, tokenizer, dataset, config):
    import pandas as pd
    import torch
    from peft import PeftModel

    if not getattr(model, "peft_config", None):
        raise ValueError("The selected model has no loaded LoRA adapter.")
    if any(getattr(module, "merged", False) is True for module in model.modules()):
        raise ValueError("Reload the unmerged LoRA checkpoint before comparing models.")
    if isinstance(model, PeftModel):
        model.base_model.enable_adapter_layers()
    else:
        model.enable_adapters()
    model.eval()
    run_dir = config.run_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    base, base_metrics = evaluate_model(model, tokenizer, dataset, "base", False, config, run_dir)
    fine, fine_metrics = evaluate_model(model, tokenizer, dataset, "fine_tuned", True, config, run_dir)
    summary = pd.DataFrame([base_metrics, fine_metrics]).set_index("model")
    predictions = pd.concat([base, fine], ignore_index=True)
    summary.to_csv(run_dir / "summary.csv")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    strict_gain = fine_metrics["strict_accuracy_pct"] - base_metrics["strict_accuracy_pct"]
    relaxed_gain = fine_metrics["relaxed_accuracy_pct"] - base_metrics["relaxed_accuracy_pct"]
    if strict_gain > 0 and relaxed_gain > 0:
        verdict = "The fine-tuned model scored higher on both accuracy metrics."
    elif strict_gain < 0 and relaxed_gain < 0:
        verdict = "The fine-tuned model scored lower on both accuracy metrics."
    elif strict_gain == 0 and relaxed_gain == 0:
        verdict = "Both models achieved the same accuracy on this evaluation."
    elif strict_gain > 0 and relaxed_gain <= 0:
        verdict = ("Strict accuracy improved, but relaxed accuracy did not. "
                   "Better answer formatting may contribute to the strict-score gain.")
    else:
        verdict = ("The results differ between scoring methods, "
                   "so the comparison does not show a consistent accuracy improvement.")
    conclusion = (
        f"On {base_metrics['n']:,} test examples, strict accuracy changed by "
        f"{strict_gain:+.2f} percentage points and relaxed accuracy by "
        f"{relaxed_gain:+.2f} percentage points. {verdict}"
    )
    if base_metrics["hit_token_limit"] or fine_metrics["hit_token_limit"]:
        conclusion += (" Some responses reached the generation limit and may be incomplete. "
                       "Check these cases and consider rerunning both models with a larger token budget.")
    adapter_config = next(iter(model.peft_config.values()))
    report = {
        "dataset": config.dataset_name, "configuration": config.dataset_config,
        "split": config.split, "n_examples": len(dataset), "test_ids": list(dataset["test_id"]),
        "seed": config.seed, "base_model": adapter_config.base_model_name_or_path,
        "generation": {"max_new_tokens": config.max_new_tokens, "do_sample": False,
                       "num_beams": 1, "num_return_sequences": 1, "use_cache": True,
                       "pad_token_id": tokenizer.eos_token_id, "eos_token_id": tokenizer.eos_token_id},
        "prompt_template": "### Instruction:\\n{question}\\n\\n### Response:\\n",
        "strict_metric": "Exact normalized numeric answer on the line after ####.",
        "relaxed_metric": "Strict parser; if #### is absent, use the last number.",
        "versions": _package_versions(),
        "metrics": [base_metrics, fine_metrics],
        "accuracy_change_percentage_points": {
            "strict": strict_gain, "relaxed": relaxed_gain,
        },
        "conclusion": conclusion,
        "fine_tuned_errors_relaxed": int((~fine["correct_relaxed"]).sum()),
        "fine_tuned_formatting_only": int((fine["correct_relaxed"] & ~fine["correct_strict"]).sum()),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(2).to_string())
    print(conclusion)
    return summary, predictions, run_dir


def _package_versions():
    versions = {}
    for name in ["torch", "transformers", "peft", "datasets", "bitsandbytes"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions
