# Phase 2: RunPod Fine-tuning Setup

This directory contains scripts and configuration for fine-tuning Mistral 7B on COBOL documentation using RunPod GPU instances.

## Files

- `finetune_mistral7b.py` - Main fine-tuning script using QLoRA
- `start.sh` - RunPod startup script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Prerequisites

1. RunPod account (sign up at https://runpod.io)
2. Dataset files from Phase 1:
   - `data/cobol_dataset_train.jsonl`
   - `data/cobol_dataset_val.jsonl`

## RunPod Setup

### Step 1: Create a RunPod Template

1. Go to RunPod Console → Templates
2. Create a new template with:
   - **Container Image**: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel`
   - **Container Disk**: 20GB (minimum)
   - **Start Script**: Copy contents of `start.sh`
   - **Environment Variables** (optional):
     - `MODEL_NAME`: `mistralai/Mistral-7B-Instruct-v0.3` (default)
     - `NUM_EPOCHS`: `3` (default)
     - `BATCH_SIZE`: `4` (default)
     - `LEARNING_RATE`: `2e-4` (default)

### Step 2: Upload Files to RunPod

You need to upload:
1. `runpod/finetune_mistral7b.py` → `/workspace/runpod/finetune_mistral7b.py`
2. `runpod/requirements.txt` → `/workspace/runpod/requirements.txt`
3. `data/cobol_dataset_train.jsonl` → `/workspace/data/cobol_dataset_train.jsonl`
4. `data/cobol_dataset_val.jsonl` → `/workspace/data/cobol_dataset_val.jsonl`

**Option A: Using RunPod's file upload**
- Use RunPod's file manager in the pod interface
- Upload files to the correct paths

**Option B: Using SFTP (Recommended over SCP)**
```bash
# SFTP is more reliable than SCP on RunPod
sftp -i ~/.ssh/id_ed25519 1vtu46wv1fktma-64411e51@ssh.runpod.io
# Once connected:
mkdir -p /workspace/runpod
mkdir -p /workspace/data
put runpod/*.py /workspace/runpod/
put runpod/*.txt /workspace/runpod/
put data/*.jsonl /workspace/data/
exit
```

**Option B2: Using rsync (Alternative)**
```bash
# rsync works better than SCP on RunPod
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  runpod/*.py runpod/*.txt \
  1vtu46wv1fktma-64411e51@ssh.runpod.io:/workspace/runpod/

rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  data/*.jsonl \
  1vtu46wv1fktma-64411e51@ssh.runpod.io:/workspace/data/
```

**Option C: Using Git (Easiest for code)**
- Push files to a private GitHub repo
- Update `start.sh` to clone before running:
```bash
# Add to start.sh before installing dependencies:
if [ ! -d "/workspace/repo" ]; then
    git clone https://github.com/yourusername/yourrepo.git /workspace/repo
    cp -r /workspace/repo/runpod /workspace/
    cp -r /workspace/repo/data /workspace/
fi
```

**Option D: Using RunPod File Manager (Web UI)**
1. Go to your pod in RunPod console
2. Click "Connect" → "HTTP Service" or use the file manager
3. Upload files directly through the web interface

**Option E: Using wget/curl from hosted location**
- Upload files to a temporary hosting service (GitHub Gist, pastebin, etc.)
- Download in startup script:
```bash
# Add to start.sh:
wget https://raw.githubusercontent.com/yourusername/repo/main/runpod/finetune_mistral7b.py -O /workspace/runpod/finetune_mistral7b.py
```

### Step 3: Start a GPU Pod

1. Go to RunPod → Pods
2. Select GPU type:
   - **Recommended**: RTX 3090 (24GB) or RTX 4090 (24GB)
   - **Budget**: RTX 3060 (12GB) - may need smaller batch size
   - **High-end**: A100 (40GB/80GB) or H100
3. Select your template
4. Start the pod

### Step 4: Monitor Training

Training will start automatically. Monitor progress:

1. **Logs**: Check pod logs in RunPod console
2. **TensorBoard**: Access via RunPod's Jupyter/SSH interface
3. **Output**: Model saved to `/workspace/models/mistral-7b-cobol`

## Configuration

### Environment Variables

Set these in RunPod template or pod settings:

- `MODEL_NAME`: Base model (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `TRAIN_FILE`: Training dataset path (default: `/workspace/data/cobol_dataset_train.jsonl`)
- `VAL_FILE`: Validation dataset path (default: `/workspace/data/cobol_dataset_val.jsonl`)
- `OUTPUT_DIR`: Model output directory (default: `/workspace/models/mistral-7b-cobol`)
- `NUM_EPOCHS`: Number of training epochs (default: `3`)
- `BATCH_SIZE`: Per-device batch size (default: `4`)
- `LEARNING_RATE`: Learning rate (default: `2e-4`)
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: `2048`)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation (default: `4`)

### Adjusting for Different GPUs

**RTX 3090/4090 (24GB)**:
- Batch size: 4-8
- Gradient accumulation: 4
- Works well with default settings

**RTX 3060 (12GB)**:
- Batch size: 2
- Gradient accumulation: 8
- May need to reduce `MAX_SEQ_LENGTH` to 1024

**A100 (40GB/80GB)**:
- Batch size: 8-16
- Gradient accumulation: 2
- Can use larger sequences

## Training Time Estimates

- **RTX 3090**: ~2-4 hours for 3 epochs
- **RTX 4090**: ~1.5-3 hours for 3 epochs
- **A100**: ~1-2 hours for 3 epochs

(Depends on dataset size and batch size)

## Cost Estimates

- **RTX 3090**: ~$0.50-1.00/hour → $1-4 per run
- **RTX 4090**: ~$0.80-1.20/hour → $1.50-5 per run
- **A100 40GB**: ~$1.50-2.50/hour → $2-8 per run

## Downloading the Model

After training completes:

1. **Via RunPod file manager**: Download `/workspace/models/mistral-7b-cobol/`
2. **Via SCP**:
```bash
scp -r pod-XXXXX:/workspace/models/mistral-7b-cobol ./models/
```

The model directory contains:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA adapter weights
- `training_config.json` - Training configuration
- Tokenizer files

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `BATCH_SIZE` (try 2 or 1)
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Reduce `MAX_SEQ_LENGTH`

### Slow Training

- Increase `BATCH_SIZE` if you have VRAM
- Use a faster GPU (A100/H100)
- Reduce `MAX_SEQ_LENGTH`

### Dataset Not Found

- Verify files are uploaded to correct paths
- Check file permissions
- Ensure JSONL format is correct

## Next Steps

After fine-tuning:
1. Download the model adapter weights
2. Proceed to Phase 3: Local Inference Setup
3. Test the fine-tuned model locally
