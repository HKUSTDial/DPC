import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
import openai
from .base_llm import BaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    LLM wrapper for OpenAI API (compatible with other OpenAI-like APIs).
    """
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        temperature: float = 0.0, 
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ask(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exception = e
                logger.error(f"Error calling LLM (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff
        
        raise RuntimeError(f"Failed to get response from LLM after {self.max_retries} attempts. Last error: {last_exception}")

