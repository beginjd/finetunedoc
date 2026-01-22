# Virtual Environment Setup for MCP Server

## The Problem

Cursor runs MCP servers as subprocesses. When it runs `python -m mcp_server`, it needs to:
1. Use the Python from your virtual environment (with all dependencies)
2. Have access to the `mcp_server` module
3. Have all dependencies (FastMCP, transformers, torch, etc.) available

## Solution: Use Full Path to Venv Python

### Option 1: Direct Python Path (Recommended)

In your Cursor MCP config, use the full path to the venv's Python:

```json
{
  "mcpServers": {
    "cobol-compiler-assistant": {
      "command": "/home/jdbegin/bar/docs/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "MODEL_PATH": "/home/jdbegin/bar/docs/models/mistral-7b-cobol",
        "BASE_MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
        "PYTHONPATH": "/home/jdbegin/bar/docs"
      }
    }
  }
}
```

**Why this works:**
- `.venv/bin/python` has all dependencies installed
- `PYTHONPATH` ensures the `mcp_server` module can be found
- No need to activate venv manually

### Option 2: Wrapper Script

Use the provided wrapper script:

```json
{
  "mcpServers": {
    "cobol-compiler-assistant": {
      "command": "/home/jdbegin/bar/docs/mcp_server/start_mcp.sh",
      "args": [],
      "env": {
        "MODEL_PATH": "/home/jdbegin/bar/docs/models/mistral-7b-cobol"
      }
    }
  }
}
```

The wrapper script activates the venv before running.

## Setup Steps

1. **Create and activate venv**:
   ```bash
   cd /home/jdbegin/bar/docs
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -r mcp_server/requirements.txt
   uv pip install -e .
   ```

3. **Verify Python path**:
   ```bash
   which python  # Should show: /home/jdbegin/bar/docs/.venv/bin/python
   ```

4. **Test MCP server**:
   ```bash
   python -m mcp_server
   # Should start without errors (will wait for stdin)
   ```

5. **Update Cursor config** with the full path to `.venv/bin/python`

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError: No module named 'mcp_server'`:
- Ensure `PYTHONPATH` is set in Cursor config
- Or install package: `uv pip install -e .`

### ModuleNotFoundError for dependencies

If you get errors about missing packages (fastmcp, transformers, etc.):
- Ensure you're using the venv's Python (check the path)
- Reinstall dependencies: `uv pip install -r mcp_server/requirements.txt`

### Wrong Python version

If you get Python version errors:
- Ensure venv was created with correct Python: `uv venv --python python3.10`
- Check Python version: `.venv/bin/python --version`
