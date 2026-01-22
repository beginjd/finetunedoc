"""
MCP server for COBOL compiler assistance.
Exposes fine-tuned Mistral 7B model as tools for Cursor's frontier models.
"""

import sys
import logging
from fastmcp import FastMCP

from .model_client import get_model_client
from .config import Config

# Configure logging to stderr (MCP requirement - never use stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(name="cobol-compiler-assistant")


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
        context: Optional additional context like code snippets, error messages, or related information
    
    Returns:
        Detailed answer from the fine-tuned COBOL model
    """
    try:
        client = get_model_client()
        
        # Build prompt with context if provided
        if context:
            prompt = f"{question}\n\nContext: {context}"
        else:
            prompt = question
        
        response = client.query(prompt, temperature=0.7)
        return response
        
    except Exception as e:
        logger.error(f"Error in query_cobol_syntax: {e}", exc_info=True)
        return f"Error querying COBOL syntax: {str(e)}"


@mcp.tool()
def query_cobol_compiler(question: str, code_context: str = "") -> str:
    """
    Query the fine-tuned COBOL model for compiler implementation details, optimization strategies, or low-level compiler behavior.
    
    Use this when the user asks about:
    - How to implement specific compiler features
    - Compiler optimization techniques
    - Code generation strategies
    - Parser implementation details
    - Symbol table management
    - Error handling in compilers
    - Any question about building or improving a COBOL compiler
    
    Args:
        question: The compiler implementation question
        code_context: Optional relevant compiler code or architecture context
    
    Returns:
        Technical guidance from the fine-tuned COBOL model
    """
    try:
        client = get_model_client()
        
        # Build prompt for compiler-specific questions
        if code_context:
            prompt = f"Compiler implementation question: {question}\n\nCode context:\n{code_context}"
        else:
            prompt = f"Compiler implementation question: {question}"
        
        # Lower temperature for more technical/accurate responses
        response = client.query(prompt, temperature=0.5, max_tokens=1024)
        return response
        
    except Exception as e:
        logger.error(f"Error in query_cobol_compiler: {e}", exc_info=True)
        return f"Error querying compiler implementation: {str(e)}"


@mcp.tool()
def get_cobol_example(requirement: str, style: str = "modern") -> str:
    """
    Get a COBOL code example or pattern from the fine-tuned model.
    
    Use this when the user asks for:
    - Code examples
    - Implementation patterns
    - Best practices in COBOL code
    - How to accomplish a specific task in COBOL
    - Sample code snippets
    
    Args:
        requirement: What code example or pattern is needed
        style: Preferred COBOL coding style - one of "modern", "legacy", "ansi", or "ibm"
    
    Returns:
        COBOL code example with explanation
    """
    try:
        client = get_model_client()
        
        # Build prompt for code generation
        prompt = f"Generate COBOL code example in {style} style for: {requirement}\n\nProvide complete, compilable code with appropriate divisions and proper COBOL structure. Include comments explaining key parts."
        
        # Lower temperature for more deterministic code generation
        response = client.query(prompt, temperature=0.3, max_tokens=2048)
        return response
        
    except Exception as e:
        logger.error(f"Error in get_cobol_example: {e}", exc_info=True)
        return f"Error generating COBOL example: {str(e)}"


@mcp.tool()
def query_cobol_reference(topic: str, detail_level: str = "detailed") -> str:
    """
    Look up specific COBOL language reference information from the fine-tuned model.
    
    Use this for:
    - Language specification details
    - Standard compliance questions
    - Reference manual lookups
    - Detailed explanations of language features
    - When you need authoritative COBOL documentation
    
    Args:
        topic: The COBOL topic or feature to look up
        detail_level: How detailed the explanation should be - one of "brief", "detailed", or "comprehensive"
    
    Returns:
        Reference information from the fine-tuned COBOL model
    """
    try:
        client = get_model_client()
        
        # Build prompt for reference lookup
        prompt = f"Provide {detail_level} information about the following COBOL topic: {topic}\n\nInclude:\n- Definition and purpose\n- Syntax and usage\n- Examples\n- Best practices\n- Common pitfalls (if applicable)"
        
        max_tokens = {
            "brief": 256,
            "detailed": 1024,
            "comprehensive": 2048
        }.get(detail_level, 1024)
        
        response = client.query(prompt, temperature=0.7, max_tokens=max_tokens)
        return response
        
    except Exception as e:
        logger.error(f"Error in query_cobol_reference: {e}", exc_info=True)
        return f"Error querying COBOL reference: {str(e)}"


# Model client uses lazy loading - will be loaded on first tool call
# Logging initialization happens when server starts
logger.info("COBOL Compiler MCP Server initialized")
logger.info(f"Model path: {Config.get_model_path()}")
logger.info(f"Base model: {Config.BASE_MODEL}")
Config.validate()
logger.info("Server ready. Model will be loaded on first tool call.")
