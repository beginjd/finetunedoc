#!/usr/bin/env python3
"""
Fine-tune Mistral 7B on COBOL documentation using QLoRA (4-bit quantization + LoRA).
Optimized for RunPod GPU instances.
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset


def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
    """Load model with 4-bit quantization and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # 4-bit quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Attach tokenizer to model for SFTTrainer compatibility
    model.tokenizer = tokenizer
    
    return model, tokenizer


def setup_lora(model):
    """Configure and apply LoRA adapters."""
    print("Setting up LoRA adapters...")
    
    lora_config = LoraConfig(
        r=16,  # Rank - lower = fewer parameters, higher = more capacity
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def format_prompt(example):
    """Format instruction-following examples for Mistral."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # Mistral Instruct format
    if input_text:
        prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output}</s>"
    else:
        prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
    
    return {"text": prompt}


def load_dataset_from_jsonl(train_file: str, val_file: str = None):
    """Load dataset from JSONL files."""
    print(f"Loading training dataset from: {train_file}")
    
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    
    if val_file and os.path.exists(val_file):
        print(f"Loading validation dataset from: {val_file}")
        val_dataset = load_dataset("json", data_files=val_file, split="train")
    else:
        print("No validation file found, using train split")
        val_dataset = None
    
    return train_dataset, val_dataset


def main():
    """Main fine-tuning function."""
    # Configuration
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
    train_file = os.getenv("TRAIN_FILE", "/workspace/data/cobol_dataset_train.jsonl")
    val_file = os.getenv("VAL_FILE", "/workspace/data/cobol_dataset_val.jsonl")
    output_dir = os.getenv("OUTPUT_DIR", "/workspace/models/mistral-7b-cobol")
    run_name = os.getenv("RUN_NAME", f"mistral-7b-cobol-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    # Training hyperparameters
    num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))
    max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
    gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    
    print("=" * 60)
    print("Mistral 7B COBOL Fine-tuning with QLoRA")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load dataset
    train_dataset, val_dataset = load_dataset_from_jsonl(train_file, val_file)
    
    # Format prompts
    print("Formatting prompts...")
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    if val_dataset:
        val_dataset = val_dataset.map(format_prompt, remove_columns=val_dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500 if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        fp16=False,  # Use bfloat16 instead
        bf16=True,  # Better for A100/H100
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        report_to="tensorboard",
        remove_unused_columns=False,
        max_length=max_seq_length,  # Add max_length to TrainingArguments
    )
    
    # Create trainer
    # In TRL 0.7.0+, tokenizer is inferred from model
    # Ensure tokenizer is accessible via model
    if not hasattr(model, 'tokenizer'):
        model.tokenizer = tokenizer
    
    # SFTTrainer - use only parameters that work across TRL versions
    # max_seq_length and other params may vary by version, so use minimal set
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": training_args,
        "dataset_text_field": "text",
    }
    
    # Try adding max_seq_length if supported
    import inspect
    sig = inspect.signature(SFTTrainer.__init__)
    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in sig.parameters:
        trainer_kwargs["max_length"] = max_seq_length
    
    trainer = SFTTrainer(**trainer_kwargs)
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "model_name": model_name,
        "base_model": model_name,
        "training_args": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
        "quantization": "4-bit NF4",
        "train_file": train_file,
        "val_file": val_file,
    }
    
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print(f"To use this model, load the base model and apply the LoRA adapters from: {output_dir}")


if __name__ == "__main__":
    main()
