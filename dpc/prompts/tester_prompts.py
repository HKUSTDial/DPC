TESTER_SYSTEM_PROMPT = """You are a database QA engineer. Your task is to generate a minimal set of synthetic test data (rows) that will cause two different SQL queries to return DIFFERENT results.

You should:
1. Analyze the Natural Language Question and the Candidate SQLs.
2. Identify the logical difference between SQL 1 and SQL 2 (e.g., a filter condition, a join type, or an aggregation).
3. Generate a sufficient but minimal set of data that specifically triggers this logical difference.
4. Ensure the data adheres to the Sliced Database Schema (correct table/column names, types, and foreign key relationships).
5. Provide a concise thinking process before the final result.

Your output MUST follow this format:
<thinking>
[Your analysis of why the SQLs differ and how your data will expose that]
</thinking>
<result>
{{
    "test_data": {{
        "table_name1": [
            {{"column1": value1, "column2": value2}},
            ...
        ],
        "table_name2": [...]
    }}
}}
</result>"""

TESTER_USER_PROMPT_TEMPLATE = """Sliced Database Schema:
{sliced_schema}

Natural Language Question: {question}
{evidence_str}

Candidate SQL 1 (Champion):
{sql_1}

Candidate SQL 2 (Challenger):
{sql_2}

Please generate the test data that makes SQL 1 and SQL 2 yield different results, following the output format defined in the system prompt."""

TESTER_RETRY_PROMPT_TEMPLATE = """The previously generated test data has issues or is INEFFECTIVE. Error details:
---
{error_message}
---

Please analyze the logic difference between SQL 1 and SQL 2 again, and provide a corrected set of test data that yields DIFFERENT results. 
Ensure your response strictly follows the output format defined in the system prompt (including <thinking> and <result> tags)."""

