#!/usr/bin/env python3
"""
Load and run the fine-tuned Mistral 7B model for inference.
Optimized for RTX 4070 (12GB VRAM) with 4-bit quantization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os


def load_finetuned_model(
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_path: str = "./models/mistral-7b-cobol",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load the fine-tuned model with LoRA adapters.
    
    Args:
        base_model_name: Base Mistral model name
        adapter_path: Path to saved LoRA adapter weights
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    print(f"Loading base model: {base_model_name}")
    
    # 4-bit quantization config for inference (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapters into base model (optional, for faster inference)
    # Uncomment if you want to merge (uses more memory but faster)
    # model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate a response from the model.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated text
    """
    # Format prompt in Mistral Instruct format
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Example usage
    import sys
    
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "./models/mistral-7b-cobol"
    
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(adapter_path=adapter_path)
    
    # Test query
    test_prompt = "What is the syntax for PERFORM VARYING in COBOL?"
    print(f"\nTest prompt: {test_prompt}")
    print("\nGenerating response...")
    response = generate_response(model, tokenizer, test_prompt)
    print(f"\nResponse: {response}")
