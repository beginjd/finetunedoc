# COBOL Compiler MCP Server

MCP server that exposes the fine-tuned Mistral 7B COBOL model as tools for Cursor's frontier models (Claude, GPT-4).

## Overview

This MCP server allows Cursor's frontier models to automatically query your fine-tuned COBOL model when they detect COBOL-specific questions. The server uses direct model loading - the model is loaded in the MCP server process.

## Prerequisites

1. Fine-tuned model from Phase 2 (adapter weights in `./models/mistral-7b-cobol/`)
2. Python 3.10+
3. CUDA-capable GPU (RTX 4070 or better recommended)
4. `uv` package manager

## Installation

1. Create/activate virtual environment:
```bash
cd /home/jdbegin/bar/docs
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r mcp_server/requirements.txt
```

3. Install the package:
```bash
uv pip install -e .
```

## Configuration

### Environment Variables

Set these before running or in Cursor's MCP config:

- `MODEL_PATH`: Path to fine-tuned adapter weights (default: `./models/mistral-7b-cobol`)
- `BASE_MODEL`: Base model name (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `MAX_TOKENS`: Max tokens per response (default: `512`)
- `TEMPERATURE`: Generation temperature (default: `0.7`)
- `TOP_P`: Nucleus sampling parameter (default: `0.9`)

### Cursor Configuration

1. Locate Cursor's MCP config file:
   - **Linux**: `~/.config/Cursor/mcp.json`
   - **macOS**: `~/Library/Application Support/Cursor/mcp.json`
   - **Windows**: `%APPDATA%\Cursor\mcp.json`

2. Add the server configuration:

```json
{
  "mcpServers": {
    "cobol-compiler-assistant": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "MODEL_PATH": "/absolute/path/to/models/mistral-7b-cobol",
        "BASE_MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
        "PYTHONPATH": "/absolute/path/to/docs"
      }
    }
  }
}
```

**Important**: 
- Use absolute paths in the config file
- **Use the Python executable from your virtual environment** (`.venv/bin/python`)
- Set `PYTHONPATH` to the docs directory so the `mcp_server` module can be found
- Example: If your project is at `/home/jdbegin/bar/docs`, use `/home/jdbegin/bar/docs/.venv/bin/python`

3. Restart Cursor for changes to take effect.

**Alternative: Using wrapper script**

You can also use the provided wrapper script:

```json
{
  "mcpServers": {
    "cobol-compiler-assistant": {
      "command": "/absolute/path/to/mcp_server/start_mcp.sh",
      "args": [],
      "env": {
        "MODEL_PATH": "/absolute/path/to/models/mistral-7b-cobol",
        "BASE_MODEL": "mistralai/Mistral-7B-Instruct-v0.3"
      }
    }
  }
}
```

## Available Tools

The server exposes 4 tools that frontier models can call:

### 1. `query_cobol_syntax`
Query COBOL syntax, language features, or compiler directives.

**When to use**: Syntax questions, language constructs, data types, file handling.

### 2. `query_cobol_compiler`
Query compiler implementation details and optimization strategies.

**When to use**: Compiler development, code generation, parser implementation.

### 3. `get_cobol_example`
Get COBOL code examples and patterns.

**When to use**: Code examples, implementation patterns, best practices.

### 4. `query_cobol_reference`
Look up COBOL language reference information.

**When to use**: Specification details, standard compliance, reference lookups.

## Usage

Once configured in Cursor, the frontier models will automatically call these tools when they detect COBOL-specific queries. You don't need to manually call them.

### Testing Locally

You can test the server directly:

```bash
# From the docs directory with venv activated
source .venv/bin/activate
python -m mcp_server
```

The server communicates via stdin/stdout (stdio transport), so it's designed to be run by Cursor, not directly.

## Model Loading

- **First tool call**: Model is loaded (takes 30-60 seconds)
- **Subsequent calls**: Model is reused (fast)
- **Memory**: ~5-6GB VRAM with 4-bit quantization

## Troubleshooting

### Model not found

If you see "Model path not found":
- Ensure the fine-tuned model is downloaded from RunPod
- Check the `MODEL_PATH` in your config
- Use absolute paths

### Out of memory

If you get CUDA OOM errors:
- Ensure you're using 4-bit quantization (default)
- Close other GPU applications
- Check GPU memory: `nvidia-smi`

### Server not appearing in Cursor

- Check Cursor logs for errors
- Verify Python path in config (must be full path to `.venv/bin/python`)
- Ensure virtual environment is activated and dependencies installed
- Check that `mcp_server` module is importable: `python -c "import mcp_server"`

### ModuleNotFoundError

If you get module not found errors:
- Ensure you're using the venv's Python (full path in config)
- Set `PYTHONPATH` in config to the docs directory
- Install package: `uv pip install -e .`
- Reinstall dependencies: `uv pip install -r mcp_server/requirements.txt`

### Slow responses

- First call is slow (model loading)
- Subsequent calls should be fast
- If consistently slow, check GPU utilization

## Architecture

```
Cursor → MCP Server Process → Fine-tuned Model (Direct Loading)
```

The model is loaded directly in the MCP server process when first needed.

## Virtual Environment Setup

See `VENV_SETUP.md` for detailed information about virtual environment configuration.

## Next Steps

1. Fine-tune your model on RunPod (Phase 2)
2. Download adapter weights to `./models/mistral-7b-cobol/`
3. Configure Cursor with this MCP server (use full path to venv Python!)
4. Test by asking COBOL questions in Cursor chat
