#!/bin/bash
# RunPod startup script for fine-tuning Mistral 7B on AMD MI300X with ROCm

set -e

echo "=== RunPod Fine-tuning Setup (AMD MI300X) ==="
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

# Install standard dependencies (PyTorch should come pre-installed in ROCm container)
uv pip install -r /workspace/runpod/requirements_amd.txt

# Install ROCm-compatible bitsandbytes from pre-built wheel
echo ""
echo "Installing ROCm-compatible bitsandbytes..."
uv pip install --no-deps --force-reinstall \
    'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl' || {
    echo "Warning: Failed to install pre-built bitsandbytes wheel"
    echo "Attempting to install from PyPI (may not work on ROCm)..."
    uv pip install bitsandbytes>=0.41.0 || {
        echo "Error: bitsandbytes installation failed. Please check ROCm compatibility."
        exit 1
    }
}

# Verify ROCm/GPU availability
echo ""
echo "Checking ROCm/GPU availability..."

# Check if rocm-smi is available
if command -v rocm-smi &> /dev/null; then
    echo "ROCm System Management Interface:"
    rocm-smi --version || true
    echo ""
    rocm-smi || true
    echo ""
fi

# Check PyTorch ROCm detection (PyTorch ROCm still uses torch.cuda API)
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    # Check if using ROCm backend
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f'ROCm/HIP version: {torch.version.hip}')
    else:
        print('Note: PyTorch may not be built with ROCm support')
else:
    print('ERROR: ROCm/CUDA not available!')
    exit(1)
"

# Verify bitsandbytes can detect GPU
echo ""
echo "Verifying bitsandbytes ROCm support..."
python -c "
try:
    import bitsandbytes as bnb
    print(f'bitsandbytes version: {bnb.__version__}')
    # Try to create a quantization config to verify it works
    from transformers import BitsAndBytesConfig
    import torch
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print('✓ bitsandbytes ROCm support verified')
except Exception as e:
    print(f'Warning: bitsandbytes verification failed: {e}')
    print('Training may still work, but quantization might not function correctly')
" || echo "Warning: Could not verify bitsandbytes"

# Check if dataset files exist
if [ ! -f "/workspace/data/cobol_dataset_train.jsonl" ]; then
    echo ""
    echo "Warning: Training dataset not found at /workspace/data/cobol_dataset_train.jsonl"
    echo "Please upload your dataset files to /workspace/data/"
    exit 1
fi

# Run fine-tuning
echo ""
echo "Starting fine-tuning on AMD MI300X..."
python /workspace/runpod/finetune_mistral7b_amd.py

echo ""
echo "=== Fine-tuning Complete ==="
echo "Model saved to: /workspace/models/mistral-7b-cobol"
