from abc import ABC, abstractmethod
import threading
from typing import List, Dict, Any, Optional, Union

class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._usage_lock = threading.Lock()

    @abstractmethod
    def ask(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a list of messages (multi-turn) to the LLM.
        Returns the response text.
        Raises an exception if all retries fail.
        """
        pass

    def get_usage(self) -> Dict[str, int]:
        """Returns the total token usage so far."""
        with self._usage_lock:
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
            }

    def reset_usage(self):
        """Resets the token usage counters."""
        with self._usage_lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0

    def _add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Accumulates token usage from a provider response."""
        with self._usage_lock:
            self.total_prompt_tokens += int(prompt_tokens or 0)
            self.total_completion_tokens += int(completion_tokens or 0)
