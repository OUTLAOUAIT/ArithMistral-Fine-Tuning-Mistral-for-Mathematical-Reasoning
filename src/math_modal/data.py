"""Dataset loading and prompt formatting from notebook cells 4, 7, and 23."""


def format_prompts(batch):
    formatted_texts = []
    for question, answer in zip(batch["question"], batch["answer"]):
        formatted_texts.append(
            f"### Instruction:\n{question}\n\n### Response:\n{answer}</s>"
        )
    return {"text": formatted_texts}


def load_training_data(config):
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_name, config.dataset_config, split=config.train_split
    )
    dataset = dataset.map(format_prompts, batched=True)
    if isinstance(dataset, dict) and "train" in dataset:
        dataset = dataset["train"]
    return dataset.train_test_split(test_size=config.test_size)


def load_evaluation_data(config):
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
    dataset = dataset.add_column("test_id", list(range(len(dataset))))
    if config.n_examples is not None:
        if not isinstance(config.n_examples, int) or config.n_examples <= 0:
            raise ValueError("N_EXAMPLES must be a positive integer or None.")
        dataset = dataset.shuffle(seed=config.seed).select(
            range(min(config.n_examples, len(dataset)))
        )
    return dataset
