# MCP Server Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install MCP server dependencies
uv pip install -r mcp_server/requirements.txt

# Or install via pyproject.toml
uv pip install -e ".[phase4]"
```

### 2. Verify Model Path

Ensure your fine-tuned model is available:

```bash
# Check if model directory exists
ls -la models/mistral-7b-cobol/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
```

### 3. Test the Server

```bash
# Test model loading and queries
python mcp_server/test_server.py
```

### 4. Configure Cursor

1. Find Cursor's MCP config file:
   - **Linux**: `~/.config/Cursor/mcp.json`
   - **macOS**: `~/Library/Application Support/Cursor/mcp.json`

2. Create or edit the file with:

```json
{
  "mcpServers": {
    "cobol-compiler-assistant": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "MODEL_PATH": "/absolute/path/to/bar/docs/models/mistral-7b-cobol",
        "BASE_MODEL": "mistralai/Mistral-7B-Instruct-v0.3"
      }
    }
  }
}
```

**Important**: 
- Use absolute paths
- Ensure Python can find the `mcp_server` module (install with `uv pip install -e .`)

3. Restart Cursor

### 5. Verify in Cursor

1. Open Cursor
2. Check MCP server status (should show "cobol-compiler-assistant" as connected)
3. Ask a COBOL question in chat
4. Frontier model should automatically call the tools

## Troubleshooting

### Module not found

If you get `ModuleNotFoundError: No module named 'mcp_server'`:

```bash
# Install the package in editable mode
cd /home/jdbegin/bar/docs
uv pip install -e .
```

### Model not found

If model path doesn't exist:
- Download adapter weights from RunPod
- Place in `models/mistral-7b-cobol/` directory
- Update `MODEL_PATH` in Cursor config

### Server not starting

- Check Cursor logs for errors
- Verify Python path is correct
- Ensure virtual environment is activated
- Test server manually: `python -m mcp_server`
