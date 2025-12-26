from typing import List, Dict, Any, Optional
from .slicer_prompts import SLICER_SYSTEM_PROMPT, SLICER_USER_PROMPT_TEMPLATE, SLICER_RETRY_PROMPT_TEMPLATE
from .tester_prompts import TESTER_SYSTEM_PROMPT, TESTER_USER_PROMPT_TEMPLATE, TESTER_RETRY_PROMPT_TEMPLATE
from .solver_prompts import SOLVER_SYSTEM_PROMPT, SOLVER_USER_PROMPT_TEMPLATE, SOLVER_RETRY_PROMPT_TEMPLATE

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
        sliced_schema_text: str,
        test_data: Dict[str, List[Dict[str, Any]]],
        evidence: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generates the initial messages for the PythonSolverAgent.
        """
        from tabulate import tabulate
        
        evidence_str = f"External Knowledge: {evidence}" if evidence else ""
        df_names = ", ".join(test_data.keys())
        
        # Format all test data into psql-style tables
        table_strings = []
        for table_name, rows in test_data.items():
            if rows:
                headers = rows[0].keys()
                data_rows = [[row[h] for h in headers] for row in rows]
                table_str = f"Table: {table_name}\n" + tabulate(data_rows, headers=headers, tablefmt="psql")
                table_strings.append(table_str)
            else:
                table_strings.append(f"Table: {table_name} (Empty)")
                
        test_data_tables = "\n\n".join(table_strings)
            
        user_prompt = SOLVER_USER_PROMPT_TEMPLATE.format(
            sliced_schema=sliced_schema_text,
            test_data_tables=test_data_tables,
            df_names=df_names,
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

    # Future agents can be added here:
    # @staticmethod
    # def get_tester_prompt(...):
    #     ...

