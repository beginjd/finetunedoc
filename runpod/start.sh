#!/bin/bash
# RunPod startup script for fine-tuning Mistral 7B

set -e

echo "=== RunPod Fine-tuning Setup ==="
echo ""

# Install system dependencies if needed
echo "Checking system dependencies..."
apt-get update -qq
apt-get install -y -qq git wget curl > /dev/null 2>&1 || true

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Navigate to workspace
cd /workspace

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
uv pip install --upgrade pip
# Install sentencepiece first (needed for Mistral tokenizer)
uv pip install sentencepiece>=0.1.99 protobuf>=3.20.0
uv pip install -r /workspace/runpod/requirements.txt

# Verify CUDA/GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Check if dataset files exist
if [ ! -f "/workspace/data/cobol_dataset_train.jsonl" ]; then
    echo "Warning: Training dataset not found at /workspace/data/cobol_dataset_train.jsonl"
    echo "Please upload your dataset files to /workspace/data/"
    exit 1
fi

# Run fine-tuning
echo ""
echo "Starting fine-tuning..."
python /workspace/runpod/finetune_mistral7b.py

echo ""
echo "=== Fine-tuning Complete ==="
echo "Model saved to: /workspace/models/mistral-7b-cobol"
