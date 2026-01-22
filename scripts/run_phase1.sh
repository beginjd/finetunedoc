#!/bin/bash
# Run Phase 1: PDF Extraction and Dataset Preparation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Phase 1: PDF Extraction and Dataset Preparation ==="
echo ""

# Check if PDFs exist
if [ ! -f "cics-api-reference.pdf" ] || [ ! -f "lrmvs.pdf" ] || [ ! -f "pgmvs.pdf" ]; then
    echo "Error: PDF files not found in current directory"
    echo "Expected: cics-api-reference.pdf, lrmvs.pdf, pgmvs.pdf"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies using uv
echo "Installing dependencies with uv..."
if [ -f "pyproject.toml" ]; then
    uv pip install -e ".[phase1]"
else
    # Fallback to requirements file
    uv pip install -r scripts/requirements_phase1.txt
fi

# Step 1: Extract PDFs
echo ""
echo "Step 1: Extracting text from PDFs..."
python scripts/extract_pdfs.py

# Step 2: Prepare dataset
echo ""
echo "Step 2: Preparing dataset..."
python scripts/prepare_dataset.py

echo ""
echo "=== Phase 1 Complete ==="
echo ""
echo "Next steps:"
echo "1. Review the extracted data in: data/extracted/"
echo "2. Review the dataset in: data/cobol_dataset_train.jsonl"
echo "3. Check statistics in: data/dataset_stats.json"
echo "4. Proceed to Phase 2: RunPod Fine-tuning Setup"
