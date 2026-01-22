#!/usr/bin/env python3
"""
Entry point for MCP server.
Runs the server via stdio transport (required for MCP).
"""

import sys
import logging

# Configure logging to stderr (MCP requirement - never use stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

from .server import mcp

if __name__ == "__main__":
    # Run MCP server via stdio
    mcp.run()
