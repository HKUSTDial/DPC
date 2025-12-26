from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..llm.base_llm import BaseLLM
from ..prompts.factory import PromptFactory

class BaseAgent(ABC):
    """
    Base class for all DPC agents.
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's primary logic."""
        pass

    def _ask_and_parse_json(
        self, 
        messages: List[Dict[str, str]], 
        max_retries: int = 3,
        validator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Helper method to call LLM and parse JSON with retries and optional validation.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.llm.ask(messages)
                parsed_json = self._parse_json_response(response)
                
                # Validate the structure if a validator is provided
                if validator:
                    if not validator(parsed_json):
                        raise ValueError("LLM response failed custom validation check.")
                
                return parsed_json
            except Exception as e:
                last_error = e
                logger.warning(f"Agent retry attempt {attempt + 1}/{max_retries} failed: {e}")
                
        raise ValueError(f"Failed to get valid response after {max_retries} attempts. Last error: {last_error}")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Utility to parse JSON from LLM response, focusing on <result> tags.
        """
        import json
        import re
        
        # 1. Extract content from <result> tags
        result_match = re.search(r'<result>\s*(.*?)\s*</result>', response, re.DOTALL)
        if result_match:
            json_str = result_match.group(1).strip()
        else:
            # Fallback for robustness: if no tags, use the whole response
            json_str = response.strip()
            
        # 2. Cleanup potential markdown code blocks (LLM may add them even if not asked)
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'^```\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
            
        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from LLM content: {json_str}") from e

