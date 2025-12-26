from typing import List, Dict, Any, Optional, Union
from .base_agent import BaseAgent
from ..prompts.factory import PromptFactory
from ..utils.schema_utils import TableSchema, SchemaExtractor
from ..utils.python_executor import PythonExecutor

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
        max_correction_attempts: int = 3
    ) -> Any:
        """
        Generates and executes Pandas code.
        
        Returns:
            The execution result (from the 'result' variable in Python).
            Raises ValueError if it fails after all correction attempts.
        """
        # 1. Prepare initial prompts
        schema_text = SchemaExtractor.to_readable_text(sliced_schema)
        messages = PromptFactory.get_solver_prompt(
            question=question,
            sliced_schema_text=schema_text,
            test_data=test_data,
            evidence=evidence
        )
        
        # 2. Iterative Self-Correction Loop
        for attempt in range(max_correction_attempts + 1):
            # Pass the WHOLE messages list for multi-turn correction
            response_text = self.llm.ask(messages)
            messages.append({"role": "assistant", "content": response_text})
            
            # Parse the code from <result> tag
            try:
                import re
                code_match = re.search(r'<result>\s*(.*?)\s*</result>', response_text, re.DOTALL)
                if not code_match:
                    raise ValueError("No <result> tag found in LLM response.")
                
                code = code_match.group(1).strip()
                
                # 3. Execute the code
                result = PythonExecutor.execute(test_data, code)
                
                # Check if result is an error (string containing traceback or specific error message)
                if isinstance(result, str) and ("Traceback" in result or result.startswith("Error:")):
                    if attempt < max_correction_attempts:
                        # Feed the error back to the LLM as a new user message
                        retry_messages = PromptFactory.get_solver_retry_prompt(result)
                        messages.extend(retry_messages)
                        continue
                    else:
                        raise ValueError(f"Python execution failed after {max_correction_attempts} retries. Last error: {result}")
                
                return result
                
            except Exception as e:
                if attempt < max_correction_attempts:
                    # Unified: use solver-specific retry prompt for any error
                    retry_messages = PromptFactory.get_solver_retry_prompt(str(e))
                    messages.extend(retry_messages)
                else:
                    raise e
                    
        raise ValueError("Unexpected failure in PythonSolverAgent loop.")

