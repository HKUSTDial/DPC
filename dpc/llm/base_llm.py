from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def ask(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and return the response text.
        Raises an exception if all retries fail.
        """
        pass

