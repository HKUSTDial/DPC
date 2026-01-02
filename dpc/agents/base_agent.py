import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..llm.base_llm import BaseLLM
from ..prompts.factory import PromptFactory

logger = logging.getLogger(__name__)

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
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"[{self.__class__.__name__}] LLM Interaction (Attempt {attempt + 1}/{max_retries})...")
                response = self.llm.ask(messages)
                parsed_json = self._parse_json_response(response)
                
                # Validate the structure if a validator is provided
                if validator:
                    if not validator(parsed_json):
                        raise ValueError("LLM response failed custom validation check.")
                
                logger.debug(f"[{self.__class__.__name__}] Successfully parsed and validated JSON.")
                return parsed_json
            except Exception as e:
                last_error = e
                logger.warning(f"[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                # In actual agents, we might want to update messages for multi-turn correction,
                # but BaseAgent provides a generic retry. 
                # Individual agents can override this if they want to inject specific feedback.
        
        raise ValueError(f"Failed to get valid response after {max_retries} attempts. Last error: {last_error}")

    def _extract_result(self, response: str) -> str:
        """
        Extracts the content after the LAST <result> tag.
        The closing </result> tag is optional.
        If no <result> tag is found, returns the full response (fallback).
        """
        start_tag = "<result>"
        end_tag = "</result>"
        
        # Find the last occurrence of <result>
        idx = response.rfind(start_tag)
        if idx != -1:
            # Extract everything after the last <result>
            content = response[idx + len(start_tag):]
            # If there's an </result> tag after it, truncate there
            end_idx = content.find(end_tag)
            if end_idx != -1:
                return content[:end_idx].strip()
            return content.strip()
        
        # Fallback: if no <result> is found, return the full response
        return response.strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Utility to parse JSON from LLM response, focusing on the LAST <result> tags.
        """
        import json
        
        # 1. Extract content from <result> tags
        json_str = self._extract_result(response)
            
        # 2. Cleanup potential markdown code blocks (LLM may add them even if not asked)
        import re
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'^```\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
            
        # 3. Remove C-style comments (// and /* */) which are invalid in standard JSON
        # This is a common failure mode for LLMs even when instructed not to.
        def remove_json_comments(text):
            # Regex to match strings OR comments
            pattern = r'("(?:\\.|[^"\\])*")|//.*|/\*[\s\S]*?\*/'
            def replacer(match):
                if match.group(1): # If it's a string, keep it
                    return match.group(1)
                return "" # If it's a comment, remove it
            return re.sub(pattern, replacer, text)

        json_str = remove_json_comments(json_str)

        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from LLM content: {json_str}") from e

