import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    is_pk: bool = False
    examples: List[str] = field(default_factory=list)
    num_distinct: int = 0
    num_total: int = 0
    num_null: int = 0

@dataclass
class ForeignKey:
    from_col: str
    to_table: str
    to_col: str

@dataclass
class TableSchema:
    name: str
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)

class SchemaExtractor:
    # Class-level cache to store schemas. Key: (db_path, max_value_len, n_examples)
    _cache: Dict[tuple, Dict[str, TableSchema]] = {}

    @classmethod
    def extract(
        cls, 
        db_path: str, 
        max_value_len: int = 128, 
        n_examples: int = 5, 
        force_refresh: bool = False
    ) -> Dict[str, TableSchema]:
        """
        Extracts the database schema. This is a class method with internal caching.
        """
        db_path = os.path.abspath(db_path)
        cache_key = (db_path, max_value_len, n_examples)

        if not force_refresh and cache_key in cls._cache:
            return cls._cache[cache_key]

        schema = {}
        # Internal helper to get connection
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                table_schema = TableSchema(name=table_name)
                
                # 2. Get Column Info
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                columns_info = cursor.fetchall()
                
                for col in columns_info:
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = col[5] > 0
                    
                    column = ColumnSchema(name=col_name, dtype=col_type, is_pk=is_pk)
                    if is_pk:
                        table_schema.primary_keys.append(col_name)
                    
                    # 3. Get Statistics & Examples
                    try:
                        cursor.execute(f'SELECT COUNT(*), COUNT(DISTINCT "{col_name}"), SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) FROM "{table_name}";')
                        stats = cursor.fetchone()
                        column.num_total = stats[0]
                        column.num_distinct = stats[1]
                        column.num_null = stats[2] if stats[2] is not None else 0
                        
                        cursor.execute(f"""
                            SELECT DISTINCT "{col_name}" 
                            FROM "{table_name}" 
                            WHERE "{col_name}" IS NOT NULL 
                            ORDER BY length(CAST("{col_name}" AS TEXT)) ASC 
                            LIMIT {n_examples};
                        """)
                        examples = [str(row[0]) for row in cursor.fetchall() if len(str(row[0])) <= max_value_len]
                        column.examples = examples
                    except Exception:
                        pass
                        
                    table_schema.columns[col_name] = column

                # 4. Get Foreign Keys
                cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
                fk_info = cursor.fetchall()
                for fk in fk_info:
                    target_table = fk[2]
                    from_col = fk[3]
                    to_col = fk[4]
                    
                    if target_table and from_col and to_col:
                        table_schema.foreign_keys.append(ForeignKey(from_col, target_table, to_col))
                
                schema[table_name] = table_schema
        
        # Save to cache
        cls._cache[cache_key] = schema
        return schema

    @staticmethod
    def to_readable_text(schema: Dict[str, TableSchema]) -> str:
        """Converts the structured schema into a prompt-friendly string."""
        output = []
        for table_name, table in schema.items():
            output.append(f"Table {table_name}:")
            for col_name, col in table.columns.items():
                pk_str = " [PK]" if col.is_pk else ""
                stats_str = f"(Distinct: {col.num_distinct}, Nulls: {col.num_null})"
                examples_str = f" | Examples: {col.examples}" if col.examples else ""
                output.append(f"  - {col_name} ({col.dtype}){pk_str} {stats_str}{examples_str}")
            
            if table.foreign_keys:
                for fk in table.foreign_keys:
                    output.append(f"  - Foreign Key: {fk.from_col} -> {fk.to_table}.{fk.to_col}")
        return "\n".join(output)


