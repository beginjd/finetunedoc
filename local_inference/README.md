# Fine-tuned Model Format and Usage

## Model Format

After fine-tuning on RunPod, your model will be saved in **LoRA adapter format** (not a full model).

### What Gets Saved

The fine-tuning script saves to `/workspace/models/mistral-7b-cobol/`:

```
mistral-7b-cobol/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA adapter weights (~100-500MB)
├── training_config.json         # Training hyperparameters
├── tokenizer_config.json        # Tokenizer configuration
├── tokenizer.json               # Tokenizer files
├── special_tokens_map.json
└── vocab.json
```

### Key Points

1. **LoRA Adapters Only**: Only the adapter weights are saved (~100-500MB), not the full 7B model (~14GB)
2. **Base Model Required**: You need the base model (`mistralai/Mistral-7B-Instruct-v0.3`) separately
3. **4-bit Quantized**: The base model is loaded with 4-bit quantization for memory efficiency
4. **PEFT Format**: Uses HuggingFace PEFT (Parameter-Efficient Fine-Tuning) format

## How to Run the Model

### Option 1: Using the Load Script (Recommended)

```bash
# Download the adapter weights from RunPod to ./models/mistral-7b-cobol/
python local_inference/load_model.py ./models/mistral-7b-cobol
```

### Option 2: Python Code

```python
from local_inference.load_model import load_finetuned_model, generate_response

# Load model
model, tokenizer = load_finetuned_model(
    adapter_path="./models/mistral-7b-cobol"
)

# Generate response
response = generate_response(
    model, 
    tokenizer, 
    "What is the syntax for PERFORM VARYING in COBOL?"
)
print(response)
```

### Option 3: Manual Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./models/mistral-7b-cobol")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/mistral-7b-cobol")

# Generate
prompt = "<s>[INST] What is PERFORM VARYING? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Memory Requirements

- **Base Model (4-bit)**: ~4-5GB VRAM
- **LoRA Adapters**: ~100-500MB
- **Total**: ~5-6GB VRAM (fits on RTX 4070 12GB)

## Merging Adapters (Optional)

For faster inference, you can merge the adapters into the base model:

```python
model = PeftModel.from_pretrained(base_model, "./models/mistral-7b-cobol")
model = model.merge_and_unload()  # Merges adapters into base model
model.save_pretrained("./models/mistral-7b-cobol-merged")
```

**Note**: Merged model will be larger (~4-5GB) but inference will be slightly faster.

## Downloading from RunPod

After training completes on RunPod:

1. **Via RunPod File Manager**: Download `/workspace/models/mistral-7b-cobol/` directory
2. **Via SCP/SFTP**: 
   ```bash
   scp -r pod-XXXXX:/workspace/models/mistral-7b-cobol ./models/
   ```

## Next Steps

1. Download adapter weights from RunPod
2. Test locally with `load_model.py`
3. Set up MCP server (Phase 4) to integrate with Cursor
