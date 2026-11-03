"""Training workflow corresponding to notebook cells 4–11."""
from .config import DEFAULT_TRAINING
from .data import load_training_data
from .model import load_training_model, lora_config, save_adapter
from pathlib import Path


def train(config=DEFAULT_TRAINING):
    from trl import SFTConfig, SFTTrainer

    dataset = load_training_data(config)
    model, tokenizer = load_training_model(config)
    args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        num_train_epochs=config.num_train_epochs,
        save_strategy=config.save_strategy,
        dataset_text_field=config.dataset_text_field,
        max_length=config.max_length,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_drop_last=config.dataloader_drop_last,
        optim=config.optim,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config(config),
        args=args,
        processing_class=tokenizer,
    )
    trainer.train()
    output_dir = Path(config.output_dir)
    save_adapter(trainer.model, tokenizer, output_dir)
    return trainer
