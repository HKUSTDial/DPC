import logging
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent
from ..prompts.factory import PromptFactory
from ..utils.schema_utils import TableSchema, SchemaExtractor
from ..eval.metrics import DPCEvaluator

logger = logging.getLogger(__name__)

class TesterAgent(BaseAgent):
    """
    Responsible for generating differentiator test data (Slice Data) 
    that causes Champion and Challenger SQLs to return different results.
    Includes a closed-loop validation to ensure data effectiveness.
    """

    def run(
        self,
        question: str,
        sql_1: str,
        sql_2: str,
        sliced_schema: Dict[str, TableSchema],
        evidence: Optional[str] = None,
        max_correction_attempts: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs the test data generation process with closed-loop validation.
        """
        # 1. Prepare schema text - include descriptions and examples to help LLM understand data distribution
        schema_text = SchemaExtractor.to_readable_text(
            sliced_schema, 
            include_stats=True, 
            include_examples=True,
            include_descriptions=True
        )
        
        # 2. Get initial messages
        messages = PromptFactory.get_tester_prompt(
            question=question,
            sql_1=sql_1,
            sql_2=sql_2,
            sliced_schema_text=schema_text,
            evidence=evidence
        )
        
        # 3. Iterative Correction Loop
        def validate_json_format(data: Dict[str, Any]) -> bool:
            if not isinstance(data, dict) or "test_data" not in data:
                return False
            test_data = data["test_data"]
            return isinstance(test_data, dict) and any(isinstance(rows, list) for rows in test_data.values())

        for attempt in range(max_correction_attempts + 1):
            try:
                logger.info(f"[TesterAgent] Generation attempt {attempt + 1}/{max_correction_attempts + 1}...")
                response_text = self.llm.ask(messages)
                messages.append({"role": "assistant", "content": response_text})
                
                # Basic JSON and Schema Alignment
                parsed = self._parse_json_response(response_text)
                if not validate_json_format(parsed):
                    raise ValueError("Missing 'test_data' key or table rows are not lists.")
                
                raw_test_data = parsed["test_data"]
                aligned_data = self._align_test_data(raw_test_data, sliced_schema)
                
                if not aligned_data:
                    raise ValueError("None of the generated tables match the provided Sliced Schema.")

                logger.info(f"[TesterAgent] Successfully generated data for {len(aligned_data)} tables. Verifying distinction...")

                # 4. Closed-loop Validation: Do the SQLs actually differ on this data?
                distinction_error = self._verify_distinction(sql_1, sql_2, aligned_data)
                
                if not distinction_error:
                    logger.info("[TesterAgent] Data verified: SQLs yield DIFFERENT results. Success!")
                    return aligned_data
                else:
                    # Data is ineffective
                    logger.warning(f"[TesterAgent] Data verification failed: {distinction_error}")
                    if attempt < max_correction_attempts:
                        retry_messages = PromptFactory.get_tester_retry_prompt(distinction_error)
                        messages.extend(retry_messages)
                        continue
                    else:
                        raise ValueError(f"Tester failed to generate distinguishing data after {max_correction_attempts} retries. Last error: {distinction_error}")

            except Exception as e:
                if attempt < max_correction_attempts:
                    retry_messages = PromptFactory.get_tester_retry_prompt(str(e))
                    messages.extend(retry_messages)
                else:
                    raise e
                    
        raise ValueError("TesterAgent failed unexpectedly.")

    def _verify_distinction(self, sql_1: str, sql_2: str, test_data: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Executes both SQLs on the test data and checks if results are DIFFERENT.
        Returns an error message if they are the same or fail, None otherwise.
        """
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(":memory:")
        try:
            # 1. Load data into memory
            for table_name, rows in test_data.items():
                pd.DataFrame(rows).to_sql(table_name, conn, index=False)
            
            # 2. Execute SQL 1
            try:
                res1 = pd.read_sql_query(sql_1, conn)
            except Exception as e:
                return f"SQL 1 failed to execute on generated data: {str(e)}"
                
            # 3. Execute SQL 2
            try:
                res2 = pd.read_sql_query(sql_2, conn)
            except Exception as e:
                return f"SQL 2 failed to execute on generated data: {str(e)}"
            
            # 4. Compare results using Soft-F1 logic (if F1 is 1.0, they are the same)
            # We want them to be DIFFERENT (F1 < 1.0)
            similarity = DPCEvaluator.evaluate(res1, res2)
            
            if similarity > 0.99: # Practically identical
                return "The generated test data is INEFFECTIVE: Both SQLs yielded the same result set. Please create data that exposes their logical difference."
            
            return None # Success! Results are different.
            
        finally:
            conn.close()

    def _align_test_data(
        self, 
        raw_data: Dict[str, Any], 
        sliced_schema: Dict[str, TableSchema]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Aligns generated table/column names with the schema (case-insensitive).
        Raises ValueError with detailed feedback if data alignment fails.
        """
        table_lookup = {name.lower(): name for name in sliced_schema.keys()}
        aligned_data = {}
        errors = []

        # 1. Check for extra tables
        for raw_table_name in raw_data.keys():
            if raw_table_name.lower() not in table_lookup:
                errors.append(f"- Table '{raw_table_name}' is not in the Sliced Schema.")

        # 2. Check for missing tables and alignment
        for target_table_name, table_schema in sliced_schema.items():
            raw_table_match = next((k for k in raw_data.keys() if k.lower() == target_table_name.lower()), None)
            
            if not raw_table_match:
                errors.append(f"- Table '{target_table_name}' is missing from your generated data.")
                continue
                
            rows = raw_data[raw_table_match]
            if not isinstance(rows, list) or not rows:
                errors.append(f"- Table '{target_table_name}' must contain a non-empty list of rows.")
                continue

            col_lookup = {c.lower(): c for c in table_schema.columns.keys()}
            aligned_rows = []
            
            for i, row in enumerate(rows):
                if not isinstance(row, dict):
                    errors.append(f"- Row {i} in table '{target_table_name}' is not a JSON object.")
                    continue
                
                aligned_row = {}
                row_cols_lower = {k.lower() for k in row.keys()}
                
                # Check for missing columns
                for target_col in table_schema.columns.keys():
                    if target_col.lower() not in row_cols_lower:
                        errors.append(f"- Column '{target_col}' is missing in table '{target_table_name}' (Row {i}).")
                
                # Check for extra columns and align
                for raw_col_name, val in row.items():
                    col_lower = raw_col_name.lower()
                    if col_lower in col_lookup:
                        aligned_row[col_lookup[col_lower]] = val
                    else:
                        errors.append(f"- Column '{raw_col_name}' in table '{target_table_name}' is not in the schema.")
                
                aligned_rows.append(aligned_row)
                
            aligned_data[target_table_name] = aligned_rows

        if errors:
            feedback = "Test data alignment errors found:\n" + "\n".join(errors)
            raise ValueError(feedback)
                    
        return aligned_data

