# MCP Server Implementation Plan

## Overview

Create a Python MCP server using FastMCP that exposes the fine-tuned Mistral 7B COBOL model as tools. Cursor's frontier models (Claude, GPT-4) will automatically call these tools when they detect COBOL-specific queries.

## Architecture

```
Cursor Frontier Model → MCP Server → Local Inference (Fine-tuned Model)
```

The MCP server will:
1. Expose tools with clear descriptions so frontier models know when to call them
2. Load and manage the fine-tuned model (or connect to inference server)
3. Format queries appropriately for the model
4. Return responses in MCP format

## Implementation Steps

### Step 1: Create MCP Server Structure

**File**: `mcp_server/server.py`
- Main MCP server using FastMCP
- Define tools with clear, descriptive names and docstrings
- Tools should be discoverable by frontier models

**Tools to expose:**
1. `query_cobol_syntax(question: str, context: str = "")` - COBOL syntax questions
2. `query_cobol_compiler(question: str, code_context: str = "")` - Compiler implementation questions
3. `get_cobol_example(requirement: str, style: str = "modern")` - Code examples
4. `query_cobol_reference(topic: str, detail_level: str = "detailed")` - Reference lookups

### Step 2: Create Model Client

**File**: `mcp_server/model_client.py`
- Client to interact with the fine-tuned model
- Two modes:
  - **Direct mode**: Load model directly in MCP server process
  - **Server mode**: Connect to local inference server (FastAPI)
- Handle model loading, caching, and query formatting
- Error handling and retries

### Step 3: Create Configuration

**File**: `mcp_server/config.py`
- Configuration for model paths, inference settings
- Environment variable support
- Default values for RTX 4070 setup

### Step 4: Create Local Inference Server (Optional)

**File**: `local_inference/server.py`
- FastAPI server with OpenAI-compatible API
- Loads fine-tuned model once, serves multiple requests
- Better for multiple MCP clients or other applications
- Endpoints:
  - `POST /v1/chat/completions` - Chat completion
  - `GET /v1/models` - List models

### Step 5: Create MCP Server Entry Point

**File**: `mcp_server/__main__.py`
- Run MCP server via stdio transport
- Use logging to stderr (not stdout) for MCP compatibility
- Handle graceful shutdown

### Step 6: Create Cursor Configuration

**File**: `cursor_mcp_config.json`
- Example configuration for Cursor
- Instructions for setup
- Path: `~/.config/Cursor/mcp.json` (Linux) or `~/Library/Application Support/Cursor/mcp.json` (macOS)

### Step 7: Create Requirements

**File**: `mcp_server/requirements.txt`
- `fastmcp>=0.9.0` - FastMCP framework
- `transformers>=4.35.0` - For model loading (if direct mode)
- `torch>=2.0.0` - PyTorch (if direct mode)
- `peft>=0.6.0` - For LoRA adapters (if direct mode)
- `bitsandbytes>=0.41.0` - For 4-bit quantization (if direct mode)
- `httpx>=0.25.0` - For HTTP client (if server mode)
- `fastapi>=0.104.0` - For inference server (optional)
- `uvicorn>=0.24.0` - For inference server (optional)

### Step 8: Create Documentation

**File**: `mcp_server/README.md`
- Setup instructions
- Configuration guide
- Tool descriptions
- Troubleshooting

## Implementation Details

### Tool Descriptions (Critical)

Tool descriptions must be detailed so frontier models know when to call them. Example:

```python
@mcp.tool()
def query_cobol_syntax(question: str, context: str = "") -> str:
    """
    Query the fine-tuned COBOL model for specific syntax rules, language features, or compiler directives.
    
    Use this when the user asks about:
    - COBOL syntax (DATA DIVISION, PROCEDURE DIVISION, etc.)
    - Language-specific constructs (PERFORM, EVALUATE, etc.)
    - Compiler directives and options
    - COBOL data types and their usage
    - File handling in COBOL
    - Any question that requires deep COBOL language knowledge
    
    Args:
        question: The specific COBOL syntax or language feature question
        context: Optional additional context like code snippets or error messages
    
    Returns:
        Detailed answer from the fine-tuned COBOL model
    """
```

### Model Loading Strategy

**Option A: Direct Loading (Simpler)**
- Load model directly in MCP server process
- Pros: Simpler, no separate server needed
- Cons: Model loaded even when not in use, slower startup

**Option B: Inference Server (Recommended)**
- Separate FastAPI server loads model
- MCP server connects via HTTP
- Pros: Model loaded once, can serve multiple clients, faster MCP startup
- Cons: More complex setup

**Recommendation**: Start with Option A, can migrate to Option B later.

### Error Handling

- Model loading errors: Return clear error messages
- Generation errors: Fallback to error response
- Timeout handling: Set reasonable timeouts
- Logging: Use stderr for logs (MCP requirement)

### Performance Optimization

- Load model once at startup, reuse for all requests
- Cache frequent queries (optional)
- Use appropriate generation parameters (temperature, max_tokens)

## File Structure

```
mcp_server/
├── __init__.py
├── __main__.py              # Entry point
├── server.py                # Main MCP server
├── model_client.py          # Model interaction logic
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── README.md                # Documentation

local_inference/
├── server.py                # FastAPI inference server (optional)
├── load_model.py            # Model loading utilities (already exists)
├── config.py                # Inference server config
└── requirements.txt         # Server dependencies

cursor_mcp_config.json       # Example Cursor config
```

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to fine-tuned adapter weights (default: `./models/mistral-7b-cobol`)
- `BASE_MODEL`: Base model name (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `INFERENCE_MODE`: `direct` or `server` (default: `direct`)
- `INFERENCE_SERVER_URL`: URL if using server mode (default: `http://localhost:8000`)
- `MAX_TOKENS`: Max tokens per response (default: `512`)
- `TEMPERATURE`: Generation temperature (default: `0.7`)

### Cursor Configuration

```json
{
  "mcpServers": {
    "cobol-compiler-assistant": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "MODEL_PATH": "/path/to/models/mistral-7b-cobol",
        "INFERENCE_MODE": "direct"
      }
    }
  }
}
```

## Testing

### Test Scripts

**File**: `mcp_server/test_server.py`
- Test MCP server tools directly
- Verify model loading
- Test tool responses

**File**: `tests/test_mcp_integration.py`
- Integration tests with Cursor
- Test tool discovery
- Test tool calling

## Dependencies Summary

- **FastMCP**: MCP server framework
- **Transformers/PyTorch**: Model loading (if direct mode)
- **PEFT/BitsAndBytes**: LoRA adapters and quantization
- **httpx**: HTTP client (if server mode)
- **FastAPI/uvicorn**: Inference server (optional)

## Success Criteria

- MCP server starts without errors
- Tools are discoverable by Cursor
- Frontier models automatically call tools for COBOL questions
- Responses are accurate and helpful
- Server handles errors gracefully
