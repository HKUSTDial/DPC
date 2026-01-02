import logging
from typing import List, Dict, Any, Optional, Union
from .base_agent import BaseAgent
from ..prompts.factory import PromptFactory
from ..utils.schema_utils import TableSchema, SchemaExtractor
from ..utils.python_executor import PythonExecutor

logger = logging.getLogger(__name__)

class PythonSolverAgent(BaseAgent):
    """
    Generates Python/Pandas code to solve the NL query using provided test data.
    Includes an iterative self-correction loop for runtime errors.
    """

    def run(
        self,
        question: str,
        test_data: Dict[str, List[Dict[str, Any]]],
        sliced_schema: Dict[str, TableSchema],
        evidence: Optional[str] = None,
        max_correction_attempts: int = 3,
        python_timeout: int = 30
    ) -> Any:
        """
        Generates and executes Pandas code.
        
        Returns:
            The execution result (from the 'result' variable in Python).
            Raises ValueError if it fails after all correction attempts.
        """
        # 1. Prepare initial prompts
        messages = PromptFactory.get_solver_prompt(
            question=question,
            sliced_schema=sliced_schema,
            test_data=test_data,
            evidence=evidence
        )
        
        # 2. Iterative Self-Correction Loop
        for attempt in range(max_correction_attempts + 1):
            try:
                logger.info(f"[PythonSolverAgent] Generation attempt {attempt + 1}/{max_correction_attempts + 1}...")
                response_text = self.llm.ask(messages)
                messages.append({"role": "assistant", "content": response_text})
                
                # Parse the code from <result> tag
                code = self._extract_result(response_text)
                
                if "<result>" in response_text and not code:
                    # If tags exist but extraction failed (unlikely with _extract_result fallback)
                    raise ValueError("Failed to extract code from <result> tags.")
                
                logger.debug(f"[PythonSolverAgent] Executing code:\n{code}")
                
                # 3. Execute the code
                result = PythonExecutor.execute(test_data, code, timeout=python_timeout)
                
                # Check if result is an error (string containing traceback or specific error message)
                if isinstance(result, str) and ("Traceback" in result or result.startswith("Error:")):
                    logger.warning(f"[PythonSolverAgent] Runtime Error:\n{result}")
                    if attempt < max_correction_attempts:
                        # Feed the error back to the LLM as a new user message
                        retry_messages = PromptFactory.get_solver_retry_prompt(result)
                        messages.extend(retry_messages)
                        continue
                    else:
                        raise ValueError(f"Python execution failed after {max_correction_attempts} retries. Last error: {result}")
                
                logger.info("[PythonSolverAgent] Code executed successfully.")
                return result
                
            except Exception as e:
                if attempt < max_correction_attempts:
                    # Unified: use solver-specific retry prompt for any error
                    retry_messages = PromptFactory.get_solver_retry_prompt(str(e))
                    messages.extend(retry_messages)
                else:
                    raise e
                    
        raise ValueError("Unexpected failure in PythonSolverAgent loop.")

