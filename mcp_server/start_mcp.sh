#!/bin/bash
# Wrapper script to start MCP server with virtual environment
# This ensures the venv is activated before running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Run MCP server
exec python -m mcp_server "$@"
