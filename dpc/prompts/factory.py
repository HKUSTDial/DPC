from typing import List, Dict, Any, Optional, TYPE_CHECKING
from .slicer_prompts import SLICER_SYSTEM_PROMPT, SLICER_USER_PROMPT_TEMPLATE, SLICER_RETRY_PROMPT_TEMPLATE
from .tester_prompts import TESTER_SYSTEM_PROMPT, TESTER_USER_PROMPT_TEMPLATE, TESTER_RETRY_PROMPT_TEMPLATE
from .solver_prompts import SOLVER_SYSTEM_PROMPT, SOLVER_USER_PROMPT_TEMPLATE, SOLVER_RETRY_PROMPT_TEMPLATE
from .selector_prompts import (
    EQUIVALENCE_GROUPER_SYSTEM_PROMPT,
    EQUIVALENCE_GROUPER_USER_PROMPT_TEMPLATE,
    EQUIVALENCE_GROUPER_RETRY_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from ..utils.schema_utils import TableSchema

class PromptFactory:
    """
    Factory for creating and formatting prompt templates for various agents.
    """
    
    @staticmethod
    def get_slicer_prompt(
        candidate_sqls: List[str], 
        full_schema_text: str
    ) -> List[Dict[str, str]]:
        """
        Generates the messages for the SlicerAgent.
        """
        sqls_str = "\n\n".join([f"- SQL {i+1}:\n{sql}" for i, sql in enumerate(candidate_sqls)])
        
        user_prompt = SLICER_USER_PROMPT_TEMPLATE.format(
            candidate_sqls=sqls_str,
            full_schema=full_schema_text
        )
        
        return [
            {"role": "system", "content": SLICER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def get_slicer_retry_prompt(error_message: str) -> List[Dict[str, str]]:
        """
        Generates the retry prompt for the SlicerAgent when Dry Run fails.
        """
        return [{"role": "user", "content": SLICER_RETRY_PROMPT_TEMPLATE.format(error_message=error_message)}]

    @staticmethod
    def get_tester_prompt(
        question: str,
        sql_1: str,
        sql_2: str,
        sliced_schema_text: str,
        evidence: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generates the messages for the TesterAgent.
        """
        evidence_str = f"External Knowledge: {evidence}" if evidence else ""
        
        user_prompt = TESTER_USER_PROMPT_TEMPLATE.format(
            question=question,
            evidence_str=evidence_str,
            sql_1=sql_1,
            sql_2=sql_2,
            sliced_schema=sliced_schema_text
        )
        
        return [
            {"role": "system", "content": TESTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def get_tester_retry_prompt(error_message: str) -> List[Dict[str, str]]:
        """
        Generates the retry prompt for the TesterAgent.
        """
        return [{"role": "user", "content": TESTER_RETRY_PROMPT_TEMPLATE.format(error_message=error_message)}]

    @staticmethod
    def get_solver_prompt(
        question: str,
        sliced_schema: Dict[str, "TableSchema"],
        test_data: Dict[str, List[Dict[str, Any]]],
        evidence: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generates the initial messages for the PythonSolverAgent.
        """
        from tabulate import tabulate
        import pandas as pd
        
        evidence_str = f"External Knowledge: {evidence}" if evidence else ""
        
        # 1. Database relationships (PK/FK) focusing only on sanitized names
        rel_output = []
        all_fks = []
        for table_name, table in sliced_schema.items():
            var_name = table_name.replace(" ", "_")
            pk_info = f"Table {var_name} Primary Keys: {', '.join(table.primary_keys)}" if table.primary_keys else ""
            if pk_info:
                rel_output.append(pk_info)
            if table.foreign_keys:
                for fk in table.foreign_keys:
                    to_table_var = fk.to_table.replace(" ", "_")
                    all_fks.append(f"- {var_name}.{fk.from_col} -> {to_table_var}.{fk.to_col}")
        if all_fks:
            rel_output.append("\nForeign Key Relationships:")
            rel_output.extend(all_fks)
        relationships_text = "\n".join(rel_output)

        # 2. Sanitize DataFrame variable names
        df_variable_names = [name.replace(" ", "_") for name in test_data.keys()]
        df_names_str = ", ".join(df_variable_names)
        
        # 3. Format all test data into sections: Types + Table
        table_sections = []
        for table_name, rows in test_data.items():
            var_name = table_name.replace(" ", "_")
            section = [f"### DataFrame Variable: {var_name}"]
            
            if rows:
                # Add Types & Descriptions
                df = pd.DataFrame(rows)
                table_schema = sliced_schema.get(table_name)
                type_lines = []
                for col, dtype in df.dtypes.items():
                    desc_parts = []
                    if table_schema and col in table_schema.columns:
                        col_schema = table_schema.columns[col]
                        if col_schema.description:
                            desc_parts.append(f"Column Description: {col_schema.description}")
                        if col_schema.value_description:
                            desc_parts.append(f"Value Description: {col_schema.value_description}")
                    
                    desc_str = f" | {' | '.join(desc_parts)}" if desc_parts else ""
                    type_lines.append(f"  - {col} ({dtype}){desc_str}")
                
                section.append("Column Details:\n" + "\n".join(type_lines))
                
                # Add Table (Prioritize PK columns at the front)
                original_headers = list(rows[0].keys())
                table_schema = sliced_schema.get(table_name)
                pk_cols = table_schema.primary_keys if table_schema else []
                
                # Build reordered headers: PKs first, then others
                new_headers = [c for c in original_headers if c in pk_cols]
                new_headers += [c for c in original_headers if c not in pk_cols]
                
                data_rows = []
                for row in rows:
                    data_rows.append([row.get(h) for h in new_headers])
                table_str = tabulate(data_rows, headers=new_headers, tablefmt="psql")
                section.append("All Data Rows:\n" + table_str)
            else:
                section.append("(Empty)")
                
            table_sections.append("\n".join(section))
                
        combined_test_data = "\n\n" + "\n\n".join(table_sections)
            
        user_prompt = SOLVER_USER_PROMPT_TEMPLATE.format(
            relationships=relationships_text,
            test_data_with_types=combined_test_data,
            df_names=df_names_str,
            question=question,
            evidence_str=evidence_str
        )
        
        return [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def get_solver_retry_prompt(error_message: str) -> List[Dict[str, str]]:
        """
        Generates the retry prompt for the PythonSolverAgent.
        """
        return [{"role": "user", "content": SOLVER_RETRY_PROMPT_TEMPLATE.format(error_message=error_message)}]

    @staticmethod
    def get_equivalence_grouper_prompt(
        question: str,
        candidate_sqls: List[str],
        full_schema_text: str,
        evidence: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generates the messages for the EquivalenceGrouperAgent.
        """
        evidence_str = f"External Knowledge: {evidence}" if evidence else ""
        sqls_str = "\n\n".join([f"{i+1}. {sql}" for i, sql in enumerate(candidate_sqls)])

        user_prompt = EQUIVALENCE_GROUPER_USER_PROMPT_TEMPLATE.format(
            full_schema=full_schema_text,
            question=question,
            evidence_str=evidence_str,
            candidate_sqls=sqls_str
        )

        return [
            {"role": "system", "content": EQUIVALENCE_GROUPER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def get_equivalence_grouper_retry_prompt(error_message: str) -> List[Dict[str, str]]:
        """
        Generates the retry prompt for the EquivalenceGrouperAgent.
        """
        return [{"role": "user", "content": EQUIVALENCE_GROUPER_RETRY_PROMPT_TEMPLATE.format(error_message=error_message)}]

    # Future agents can be added here:
    # @staticmethod
    # def get_tester_prompt(...):
    #     ...

