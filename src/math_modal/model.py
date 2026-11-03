"""Model, tokenizer, quantization, and adapter operations."""


def bitsandbytes_config(config):
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def load_training_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bitsandbytes_config(config),
        device_map="auto",
        torch_dtype=getattr(torch, config.torch_dtype),
    )
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def lora_config(config):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.task_type,
    )


def load_inference_pipeline(model_path, config):
    """Mirror notebook loading: quantized AutoModel + local tokenizer + pipeline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bitsandbytes_config(config),
        device_map="auto",
        torch_dtype=getattr(torch, config.torch_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def save_adapter(model, tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
