# COBOL Compiler Fine-tuning

Fine-tune Mistral 7B on COBOL documentation for compiler development assistance.

## Project Structure

```
.
├── scripts/          # Phase 1: PDF extraction and dataset preparation
├── runpod/           # Phase 2: RunPod fine-tuning setup
├── data/             # Extracted datasets (JSONL format)
├── pyproject.toml    # Python project configuration
└── README.md         # This file
```

## Quick Start

### Phase 1: Extract PDFs and Prepare Dataset

```bash
./scripts/run_phase1.sh
```

### Phase 2: Fine-tune on RunPod

1. Upload files to RunPod pod
2. Use `runpod/start.sh` as startup script
3. Training runs automatically

See `runpod/README.md` for detailed instructions.

## Requirements

- Python 3.10+
- `uv` package manager
- RunPod account (for fine-tuning)
- RTX 4070 or better (for local inference)

## License

[Add your license here]
