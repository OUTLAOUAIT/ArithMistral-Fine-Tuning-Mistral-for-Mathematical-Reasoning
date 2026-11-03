"""Command line entry points for training, inference, and evaluation."""
import argparse
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_TRAINING, EvaluationConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="math-modal")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output-dir", default=DEFAULT_TRAINING.output_dir)
    train_parser.add_argument("--model-id", default=DEFAULT_TRAINING.model_id)
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--model-path", default=DEFAULT_TRAINING.output_dir)
    infer_parser.add_argument("--question", required=True)
    infer_parser.add_argument("--max-new-tokens", type=int, required=True)
    infer_parser.add_argument("--do-sample", action="store_true", default=False)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--model-path", default=DEFAULT_TRAINING.output_dir)
    eval_parser.add_argument("--n-examples", type=int, default=200)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--max-new-tokens", type=int, default=512)
    eval_parser.add_argument("--run-dir", type=Path, default=Path("gsm8k_evaluation"))
    args = parser.parse_args(argv)

    if args.command == "train":
        from .train import train
        train(replace(DEFAULT_TRAINING, output_dir=args.output_dir, model_id=args.model_id))
    elif args.command == "infer":
        from .inference import generate, load_generator
        print(generate(load_generator(args.model_path), args.question,
                       args.max_new_tokens, args.do_sample))
    else:
        from .data import load_evaluation_data
        from .evaluation import evaluate
        from .model import load_inference_pipeline
        config = replace(DEFAULT_TRAINING, output_dir=args.model_path)
        generator = load_inference_pipeline(args.model_path, config)
        evaluation_config = EvaluationConfig(
            n_examples=args.n_examples, seed=args.seed,
            max_new_tokens=args.max_new_tokens, run_dir=args.run_dir,
        )
        dataset = load_evaluation_data(evaluation_config)
        _, _, run_dir = evaluate(generator.model, generator.tokenizer, dataset, evaluation_config)
        print(f"Saved evaluation to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
