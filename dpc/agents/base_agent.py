import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..llm.base_llm import BaseLLM
from ..utils.response_parser import extract_result_block, parse_json_response

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
        return extract_result_block(response)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Utility to parse JSON from LLM response, focusing on the LAST <result> tags.
        """
        return parse_json_response(response)
