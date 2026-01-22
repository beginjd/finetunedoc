"""
Configuration for MCP server.
Supports environment variables and default values.
"""

import os
from pathlib import Path


class Config:
    """Configuration for COBOL MCP server."""
    
    # Model paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/mistral-7b-cobol")
    BASE_MODEL: str = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    
    # Inference settings
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    
    # Model loading
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "auto")
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Get model path as Path object."""
        return Path(cls.MODEL_PATH).expanduser().resolve()
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        model_path = cls.get_model_path()
        if not model_path.exists():
            print(f"Warning: Model path does not exist: {model_path}")
            print("The model will be downloaded on first use.")
        return True
