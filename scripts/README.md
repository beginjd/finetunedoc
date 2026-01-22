# Phase 1: PDF Extraction and Dataset Preparation

This directory contains scripts for extracting text from COBOL documentation PDFs and preparing them for fine-tuning.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup

### Option 1: Using the automated script (Recommended)

```bash
./scripts/run_phase1.sh
```

This script will:
- Check for required PDFs
- Create a virtual environment with `uv venv`
- Install dependencies with `uv pip install`
- Run both extraction and dataset preparation steps

### Option 2: Manual setup

1. Create virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
# Using pyproject.toml (recommended)
uv pip install -e ".[phase1]"

# Or using requirements file
uv pip install -r scripts/requirements_phase1.txt
```

## Usage

### Step 1: Extract PDFs

Extract text from all PDFs in the parent directory:

```bash
python scripts/extract_pdfs.py
```

This will:
- Extract text from `cics-api-reference.pdf`, `lrmvs.pdf`, and `pgmvs.pdf`
- Preserve document structure (chapters, sections, code blocks)
- Extract tables
- Save extracted data to `data/extracted/`

### Step 2: Prepare Dataset

Convert extracted text to instruction-following format:

```bash
python scripts/prepare_dataset.py
```

This will:
- Process all extracted PDF data
- Create instruction-input-output examples
- Split into train/validation sets (90/10)
- Save to `data/cobol_dataset_train.jsonl` and `data/cobol_dataset_val.jsonl`

## Output Files

- `data/extracted/*_extracted.json` - Raw extracted text from each PDF
- `data/cobol_dataset_train.jsonl` - Training dataset (JSONL format)
- `data/cobol_dataset_val.jsonl` - Validation dataset (JSONL format)
- `data/dataset_stats.json` - Dataset statistics

## Dataset Format

Each line in the JSONL files is a JSON object:

```json
{
  "instruction": "What is the syntax for PERFORM VARYING?",
  "input": "",
  "output": "PERFORM VARYING identifier-1 FROM identifier-2 BY identifier-3 UNTIL condition..."
}
```

This format is compatible with Mistral fine-tuning.

## Troubleshooting

### Virtual environment activation

If you get permission errors when activating the virtual environment:

```bash
# Use source instead of executing
source .venv/bin/activate

# Or recreate the venv
rm -rf .venv
uv venv
source .venv/bin/activate
```

### Missing dependencies

If dependencies are missing, reinstall:

```bash
uv pip install -e ".[phase1]"
```

### WSL-specific notes

- Make sure you're in the WSL environment (not Windows)
- Use forward slashes in paths
- The virtual environment will be created in `.venv/` directory
