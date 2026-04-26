import logging
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent
from ..prompts.factory import PromptFactory
from ..utils.schema_utils import TableSchema, SchemaExtractor
from ..utils.db_utils import ensure_readonly_query

logger = logging.getLogger(__name__)

class SlicerAgent(BaseAgent):
    """
    Responsible for identifying the relevant subset of the database schema (Schema Slicing).
    """

    def run(
        self, 
        candidate_sqls: List[str], 
        full_schema: Dict[str, TableSchema],
        max_correction_attempts: int = 3
    ) -> Dict[str, TableSchema]:
        """
        Runs the schema slicing process with an iterative dry-run correction loop.
        """
        # 1. Convert full schema to readable text - include stats and examples for slicing
        full_schema_text = SchemaExtractor.to_readable_text(
            full_schema, 
            include_stats=True, 
            include_examples=True,
            include_descriptions=True
        )
        
        # 2. Initial messages
        messages = PromptFactory.get_slicer_prompt(
            candidate_sqls=candidate_sqls,
            full_schema_text=full_schema_text
        )
        
        # 3. Iterative Loop
        def validate_json_format(data: Dict[str, Any]) -> bool:
            if not isinstance(data, dict) or "relevant_schema" not in data:
                return False
            for item in data["relevant_schema"]:
                if "table" not in item or "columns" not in item:
                    return False
            return True

        for attempt in range(max_correction_attempts + 1):
            try:
                logger.info(f"[SlicerAgent] Slicing attempt {attempt + 1}/{max_correction_attempts + 1}...")
                response_text = self.llm.ask(messages)
                messages.append({"role": "assistant", "content": response_text})
                
                # Basic JSON structure validation
                parsed = self._parse_json_response(response_text)
                if not validate_json_format(parsed):
                    raise ValueError("Incorrect JSON structure: missing 'relevant_schema' or table/columns keys.")
                
                relevant_items = parsed["relevant_schema"]
                logger.debug(f"[SlicerAgent] LLM suggested {len(relevant_items)} tables.")
                
                # Filter the schema
                sliced_schema = self._filter_schema(full_schema, relevant_items)
                if not sliced_schema:
                    raise ValueError("The identified Schema Slice resulted in an empty set of tables.")

                logger.info(f"[SlicerAgent] Schema filtered to {len(sliced_schema)} tables. Validating via dry-run...")

                # 4. Dry Run Validation: Test if SQLs can execute against this slice
                dry_run_errors = self._dry_run_validation(candidate_sqls, sliced_schema)
                
                if not dry_run_errors:
                    logger.info("[SlicerAgent] Dry-run successful! Schema slice is valid.")
                    return sliced_schema
                else:
                    # Logic error in schema slice (missing columns/tables)
                    error_msg = "\n".join(dry_run_errors)
                    logger.warning(f"[SlicerAgent] Dry-run failed:\n{error_msg}")
                    if attempt < max_correction_attempts:
                        retry_messages = PromptFactory.get_slicer_retry_prompt(error_msg)
                        messages.extend(retry_messages)
                        continue
                    else:
                        raise ValueError(f"Slicer failed dry-run after {max_correction_attempts} retries. Errors: {dry_run_errors}")

            except Exception as e:
                if attempt < max_correction_attempts:
                    # Unified: use slicer-specific retry prompt for any error
                    retry_messages = PromptFactory.get_slicer_retry_prompt(str(e))
                    messages.extend(retry_messages)
                else:
                    raise e
                    
        raise ValueError("SlicerAgent failed unexpectedly.")

    def _dry_run_validation(self, sqls: List[str], sliced_schema: Dict[str, TableSchema]) -> List[str]:
        """
        Attempts to execute the SQLs against an empty in-memory DB containing only the sliced schema.
        Uses Pandas to simplify table creation while preserving original data types.
        """
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(":memory:")
        errors = []
        
        try:
            # Create empty tables based on sliced_schema
            for table_name, table in sliced_schema.items():
                # 1. Create an empty DataFrame
                df = pd.DataFrame(columns=list(table.columns.keys()))
                
                # 2. Extract original data types from our schema objects
                # These are strings like 'INTEGER', 'VARCHAR(255)', 'DATETIME', etc.
                dtype_map = {col_name: col_meta.dtype for col_name, col_meta in table.columns.items()}
                
                # 3. Use Pandas to create the table with correct types
                # The 'dtype' argument in to_sql handles the mapping to SQLite types
                df.to_sql(table_name, conn, index=False, dtype=dtype_map)
            
            # Test each SQL
            for i, sql in enumerate(sqls):
                try:
                    ensure_readonly_query(sql)
                    # 1. Use EXPLAIN to validate the SQL against the sliced schema.
                    # No need for complex cleaning because we're prefixing, not appending.
                    # We just strip whitespace to be clean.
                    test_sql = f"EXPLAIN {sql.strip()}"
                    pd.read_sql_query(test_sql, conn)
                except Exception as e:
                    errors.append(f"SQL {i+1} Error: {str(e)}")
                    
        finally:
            conn.close()
            
        return errors

    def _filter_schema(
        self, 
        full_schema: Dict[str, TableSchema], 
        relevant_items: List[Dict[str, Any]]
    ) -> Dict[str, TableSchema]:
        """
        Filters the full schema into a subset, handling case-insensitivity.
        """
        # Create case-insensitive lookup for tables: {lower_name: original_name}
        table_lookup = {name.lower(): name for name in full_schema.keys()}
        
        sliced_schema = {}
        for item in relevant_items:
            raw_table_name = str(item["table"])
            table_name_lower = raw_table_name.lower()
            
            if table_name_lower in table_lookup:
                original_table_name = table_lookup[table_name_lower]
                original_table = full_schema[original_table_name]
                
                # Create case-insensitive lookup for columns in this table
                col_lookup = {name.lower(): name for name in original_table.columns.keys()}
                
                # Filter columns requested by LLM
                columns_needed = item["columns"]
                matched_columns = {}
                matched_pks = []
                
                for col_name in columns_needed:
                    col_name_lower = str(col_name).lower()
                    if col_name_lower in col_lookup:
                        original_col_name = col_lookup[col_name_lower]
                        matched_columns[original_col_name] = original_table.columns[original_col_name]
                        if matched_columns[original_col_name].is_pk:
                            matched_pks.append(original_col_name)

                # Heuristic: Always include Primary Keys even if LLM missed them
                for pk_name in original_table.primary_keys:
                    if pk_name not in matched_columns:
                        matched_columns[pk_name] = original_table.columns[pk_name]
                        matched_pks.append(pk_name)

                if matched_columns:
                    sliced_schema[original_table_name] = TableSchema(
                        name=original_table_name,
                        columns=matched_columns,
                        foreign_keys=[], # Will be re-added below
                        primary_keys=list(set(matched_pks))
                    )
        
        # 5. Re-add relevant Foreign Keys between the sliced tables
        for table_name, table in sliced_schema.items():
            original_table = full_schema[table_name]
            for fk in original_table.foreign_keys:
                # FK is relevant if both 'from_col' is in the sliced table AND 'to_table' is in the slice
                # (Note: We also ensure 'to_col' exists in the target table of the slice)
                if fk.to_table in sliced_schema and fk.from_col in table.columns:
                    target_table_in_slice = sliced_schema[fk.to_table]
                    if fk.to_col in target_table_in_slice.columns:
                        table.foreign_keys.append(fk)
                    
        return sliced_schema
