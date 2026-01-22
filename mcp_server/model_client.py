"""
Model client for loading and querying the fine-tuned COBOL model.
Uses direct loading (Option 1) - model loaded in MCP server process.
"""

import sys
import logging
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .config import Config

# Configure logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ModelClient:
    """Client for interacting with the fine-tuned COBOL model."""
    
    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._loaded = False
    
    def load_model(self) -> None:
        """Load the fine-tuned model with LoRA adapters."""
        if self._loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading base model: {Config.BASE_MODEL}")
            
            # 4-bit quantization config for inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load base model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load LoRA adapters
            model_path = Config.get_model_path()
            logger.info(f"Loading LoRA adapters from: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model path not found: {model_path}\n"
                    f"Please ensure the fine-tuned model is available at this path."
                )
            
            self.model = PeftModel.from_pretrained(self.model, str(model_path))
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model.eval()
            self._loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def query(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Query the fine-tuned model.
        
        Args:
            prompt: Input prompt/question
            max_tokens: Maximum tokens to generate (default: Config.MAX_TOKENS)
            temperature: Sampling temperature (default: Config.TEMPERATURE)
            top_p: Nucleus sampling parameter (default: Config.TOP_P)
        
        Returns:
            Generated response
        """
        if not self._loaded:
            self.load_model()
        
        max_tokens = max_tokens or Config.MAX_TOKENS
        temperature = temperature or Config.TEMPERATURE
        top_p = top_p or Config.TOP_P
        
        try:
            # Format prompt in Mistral Instruct format
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"


# Global model client instance (loaded once)
_model_client: Optional[ModelClient] = None


def get_model_client() -> ModelClient:
    """Get or create the global model client instance."""
    global _model_client
    if _model_client is None:
        _model_client = ModelClient()
    return _model_client
