# AGENTS.md - COBOL Compiler Fine-tuning Project

This document provides context and guidance for AI agents working on this project.

## Project Overview

This project fine-tunes Mistral 7B on COBOL documentation to create a specialized assistant for COBOL compiler development. The fine-tuned model is exposed via an MCP (Model Context Protocol) server that integrates with Cursor IDE, allowing frontier models (Claude, GPT-4) to automatically query the specialized COBOL model when needed.

## Architecture

```
Phase 1: PDF Extraction → Dataset Preparation
    ↓
Phase 2: RunPod Fine-tuning (QLoRA)
    ↓
Phase 3: Local Inference Testing
    ↓
Phase 4: MCP Server Integration
    ↓
Cursor Frontier Models → MCP Server → Fine-tuned Model
```

## Project Structure

```
/home/jdbegin/bar/docs/
├── scripts/              # Phase 1: PDF extraction and dataset preparation
│   ├── extract_pdfs.py
│   ├── prepare_dataset.py
│   └── run_phase1.sh
├── data/                 # Datasets (JSONL format)
│   ├── cobol_dataset_train.jsonl
│   ├── cobol_dataset_val.jsonl
│   └── dataset_stats.json
├── runpod/               # Phase 2: Fine-tuning scripts for RunPod
│   ├── finetune_mistral7b.py
│   ├── start.sh
│   └── requirements.txt
├── local_inference/      # Phase 3: Local model testing
│   ├── load_model.py
│   └── README.md
├── mcp_server/           # Phase 4: MCP server for Cursor integration
│   ├── server.py         # Main MCP server
│   ├── model_client.py   # Model loading and querying
│   ├── config.py         # Configuration management
│   └── requirements.txt
├── models/               # Fine-tuned model adapters (created after Phase 2)
│   └── mistral-7b-cobol/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
├── pyproject.toml        # Python project configuration
└── .venv/                # Virtual environment (created with `uv venv`)
```

## Key Paths and Configuration

### Project Root
- **Absolute path**: `/home/jdbegin/bar/docs`
- **Virtual environment**: `/home/jdbegin/bar/docs/.venv`
- **Python executable**: `/home/jdbegin/bar/docs/.venv/bin/python`

### Model Paths
- **Fine-tuned model**: `./models/mistral-7b-cobol/` (relative to project root)
- **Base model**: `mistralai/Mistral-7B-Instruct-v0.3` (downloaded from HuggingFace)

### Cursor MCP Configuration
- **Config file**: `~/.config/Cursor/mcp.json` (Linux)
- **Example config**: `cursor_mcp_config.json` (in project root)

### Environment Variables
- `MODEL_PATH`: Path to fine-tuned adapter weights (default: `./models/mistral-7b-cobol`)
- `BASE_MODEL`: Base model name (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `MAX_TOKENS`: Max tokens per response (default: `512`)
- `TEMPERATURE`: Generation temperature (default: `0.7`)
- `TOP_P`: Nucleus sampling parameter (default: `0.9`)

## Development Workflows

### Phase 1: Dataset Preparation

**Goal**: Extract COBOL documentation from PDFs and prepare training dataset.

**Steps**:
1. Ensure PDFs are in parent directory: `cics-api-reference.pdf`, `lrmvs.pdf`, `pgmvs.pdf`
2. Run automated script: `./scripts/run_phase1.sh`
   - Creates virtual environment
   - Installs dependencies (`uv pip install -e ".[phase1]"`)
   - Extracts PDFs to `data/extracted/`
   - Prepares dataset to `data/cobol_dataset_train.jsonl` and `data/cobol_dataset_val.jsonl`

**Manual steps** (if needed):
```bash
source .venv/bin/activate
python scripts/extract_pdfs.py
python scripts/prepare_dataset.py
```

**Output**: JSONL files with instruction-input-output format compatible with Mistral fine-tuning.

### Phase 2: Fine-tuning on RunPod

**Goal**: Fine-tune Mistral 7B using QLoRA on RunPod GPU instances.

**Steps**:
1. Upload files to RunPod pod:
   - `runpod/finetune_mistral7b.py`
   - `runpod/requirements.txt`
   - `data/cobol_dataset_train.jsonl`
   - `data/cobol_dataset_val.jsonl`
2. Start GPU pod (RTX 3090/4090 recommended)
3. Run `start.sh` to get the ball rolling
4. Training runs automatically, saves to `/workspace/models/mistral-7b-cobol/`
5. Download adapter weights to local `./models/mistral-7b-cobol/`

**Key files**:
- `runpod/finetune_mistral7b.py`: QLoRA fine-tuning script
- `runpod/start.sh`: RunPod startup script

**Model format**: LoRA adapters only (~100-500MB), not full model. Base model required separately.

### Phase 3: Local Inference Testing

**Goal**: Test the fine-tuned model locally before MCP integration.

**Steps**:
```bash
source .venv/bin/activate
python local_inference/load_model.py ./models/mistral-7b-cobol
```

**Memory requirements**: ~5-6GB VRAM with 4-bit quantization (fits on RTX 4070).

### Phase 4: MCP Server Setup

**Goal**: Expose fine-tuned model as MCP tools for Cursor integration.

**Installation**:
```bash
source .venv/bin/activate
uv pip install -r mcp_server/requirements.txt
uv pip install -e .
```

**Configuration**:
1. Edit `~/.config/Cursor/mcp.json`:
   - Use **absolute path** to `.venv/bin/python`
   - Set `PYTHONPATH` to project root
   - Configure `MODEL_PATH` environment variable
2. Restart Cursor

**Available MCP Tools**:
- `query_cobol_syntax`: COBOL syntax and language features
- `query_cobol_compiler`: Compiler implementation questions
- `get_cobol_example`: Code examples and patterns
- `query_cobol_reference`: Reference lookups

**Testing**:
```bash
python -m mcp_server  # Test server directly
python mcp_server/test_server.py  # Test tools
```

## Package Management

This project uses `uv` for package management.

**Virtual environment**:
```bash
uv venv                    # Create virtual environment
source .venv/bin/activate  # Activate
```

**Install dependencies**:
```bash
# Phase 1 (PDF extraction)
uv pip install -e ".[phase1]"

# Phase 2 (Fine-tuning - on RunPod)
uv pip install -r runpod/requirements.txt

# Phase 4 (MCP server)
uv pip install -r mcp_server/requirements.txt
uv pip install -e .  # Install package in editable mode
```

**Project configuration**: See `pyproject.toml` for dependency groups.

## Model Details

### Base Model
- **Name**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Size**: ~14GB (full model)
- **Format**: HuggingFace transformers

### Fine-tuned Model
- **Format**: LoRA adapters (PEFT)
- **Size**: ~100-500MB (adapter weights only)
- **Quantization**: 4-bit (NF4) for memory efficiency
- **Memory**: ~5-6GB VRAM when loaded
- **Location**: `./models/mistral-7b-cobol/`

### Model Loading
- **Lazy loading**: Model loads on first MCP tool call (30-60 seconds)
- **Caching**: Model stays loaded for subsequent calls
- **Device**: Auto-detected (CUDA if available)

## Common Tasks

### Adding New MCP Tools

1. Edit `mcp_server/server.py`
2. Add tool function with `@mcp.tool()` decorator
3. Include clear docstring (frontier models use this to decide when to call)
4. Test with `python mcp_server/test_server.py`
5. Restart Cursor

### Modifying Fine-tuning Parameters

1. Edit `runpod/finetune_mistral7b.py`
2. Adjust hyperparameters (learning rate, batch size, epochs)
3. Update RunPod template or environment variables
4. Re-run fine-tuning

### Testing Model Locally

```python
from local_inference.load_model import load_finetuned_model, generate_response

model, tokenizer = load_finetuned_model("./models/mistral-7b-cobol")
response = generate_response(model, tokenizer, "Your question here")
print(response)
```

### Debugging MCP Server

1. Check Cursor logs for errors
2. Verify Python path in `mcp.json` (must be absolute path to venv Python)
3. Test server directly: `python -m mcp_server`
4. Verify module import: `python -c "import mcp_server"`
5. Check model path exists and contains adapter files

## COBOL-Specific Context

This project focuses on COBOL compiler development assistance. The fine-tuned model is trained on:
- COBOL language reference (LRMVS)
- CICS API reference
- Programmer's guide (PGMVS)

**Use cases**:
- COBOL syntax questions
- Compiler implementation guidance
- Code generation for COBOL
- Language specification lookups

**MCP tools are automatically called** by Cursor's frontier models when they detect COBOL-related queries. No manual tool invocation needed.

## Testing and Validation

### Dataset Validation
- Check `data/dataset_stats.json` for dataset statistics
- Verify JSONL format: each line is valid JSON with `instruction`, `input`, `output` fields

### Model Testing
- Use `local_inference/load_model.py` for quick tests
- Use `mcp_server/test_server.py` for MCP tool testing
- Verify responses are COBOL-specific and accurate

### MCP Integration Testing
1. Configure MCP server in Cursor
2. Ask COBOL questions in Cursor chat
3. Verify frontier models automatically call MCP tools
4. Check responses are from fine-tuned model

## Troubleshooting

### Module Not Found Errors
- Ensure package is installed: `uv pip install -e .`
- Check `PYTHONPATH` in MCP config points to project root
- Verify using venv Python (absolute path in config)

### Model Not Found
- Verify `MODEL_PATH` points to adapter directory
- Check adapter files exist: `adapter_config.json`, `adapter_model.safetensors`
- Use absolute paths in configuration

### Out of Memory
- Ensure 4-bit quantization is enabled (default)
- Close other GPU applications
- Check GPU memory: `nvidia-smi`
- Model requires ~5-6GB VRAM

### MCP Server Not Appearing in Cursor
- Verify config file location: `~/.config/Cursor/mcp.json`
- Check JSON syntax is valid
- Use absolute paths for Python executable
- Restart Cursor after config changes
- Check Cursor logs for errors

### Slow Responses
- First call is slow (model loading: 30-60 seconds)
- Subsequent calls should be fast
- If consistently slow, check GPU utilization

## Best Practices

1. **Always use absolute paths** in Cursor MCP configuration
2. **Use virtual environment Python** (`.venv/bin/python`) in MCP config
3. **Test locally** before deploying to RunPod
4. **Monitor GPU memory** when loading models
5. **Keep adapter weights small** - only save LoRA adapters, not full model
6. **Version control** code but not model files (add `models/` to `.gitignore`)
7. **Document changes** to MCP tools with clear docstrings

## Next Steps for Agents

When working on this project:

1. **Understand the phase**: Determine which phase you're working on (1-4)
2. **Check dependencies**: Ensure correct virtual environment and packages
3. **Verify paths**: Use absolute paths for configuration, relative paths in code
4. **Test incrementally**: Test each component before integration
5. **Check logs**: Review Cursor logs and terminal output for errors
6. **Respect memory limits**: Be aware of GPU memory constraints
7. **Maintain compatibility**: Keep MCP tool interfaces stable for Cursor integration

## Additional Resources

- **Phase 1 docs**: `scripts/README.md`
- **Phase 2 docs**: `runpod/README.md`
- **Phase 3 docs**: `local_inference/README.md`
- **Phase 4 docs**: `mcp_server/README.md`, `mcp_server/SETUP.md`
- **MCP Server implementation**: `mcp_server/IMPLEMENTATION_SUMMARY.md`
