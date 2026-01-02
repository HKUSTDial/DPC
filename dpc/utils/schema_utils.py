import sqlite3
import pandas as pd
import chardet
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

def _normalize_description_string(description: str) -> str:
    """Normalize description string for cleaner prompts."""
    if not description:
        return ""
    description = str(description).replace("\r", " ").replace("\n", " ").replace("commonsense evidence:", "").strip()
    while "  " in description:
        description = description.replace("  ", " ")
    return description.strip()

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    is_pk: bool = False
    examples: List[str] = field(default_factory=list)
    num_distinct: int = 0
    num_total: int = 0
    num_null: int = 0
    description: str = "" # From BIRD description files
    value_description: str = "" # From BIRD description files

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
    _lock = threading.Lock()

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

        # Double-checked locking pattern
        if not force_refresh and cache_key in cls._cache:
            return cls._cache[cache_key]

        with cls._lock:
            # Check again inside the lock
            if not force_refresh and cache_key in cls._cache:
                return cls._cache[cache_key]

            # 0. Load external descriptions (BIRD specific)
            descriptions = cls._load_bird_descriptions(db_path)

            schema = {}
            # Internal helper to get connection
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 1. Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table_name in tables:
                    table_schema = TableSchema(name=table_name)
                    table_desc = descriptions.get(table_name.lower(), {})
                    
                    # 2. Get Column Info
                    cursor.execute(f'PRAGMA table_info("{table_name}");')
                    columns_info = cursor.fetchall()
                    
                    for col in columns_info:
                        col_name = col[1]
                        col_type = col[2]
                        is_pk = col[5] > 0
                        
                        column = ColumnSchema(name=col_name, dtype=col_type, is_pk=is_pk)
                        
                        # Attach descriptions if available
                        col_desc_info = table_desc.get(col_name.lower(), {})
                        column.description = col_desc_info.get("column_description", "")
                        column.value_description = col_desc_info.get("value_description", "")

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
    def _load_bird_descriptions(db_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Loads BIRD-style database description CSVs if they exist.
        """
        db_path_obj = Path(db_path)
        db_id = db_path_obj.stem
        db_dir = db_path_obj.parent
        desc_dir = db_dir / "database_description"
        
        if not desc_dir.exists():
            return {}
            
        database_description = {}
        for csv_file in desc_dir.glob("*.csv"):
            table_name_lower = csv_file.stem.lower().strip()
            try:
                raw_bytes = csv_file.read_bytes()
                if not raw_bytes: continue
                encoding = chardet.detect(raw_bytes)["encoding"] or "utf-8"
                
                df = pd.read_csv(csv_file, encoding=encoding, index_col=False)
                table_description = {}
                for _, row in df.iterrows():
                    if pd.isna(row["original_column_name"]):
                        continue
                        
                    orig_col = str(row["original_column_name"]).strip().lower()
                    col_desc = _normalize_description_string(row["column_description"]) if pd.notna(row["column_description"]) else ""
                    val_desc = _normalize_description_string(row["value_description"]) if pd.notna(row["value_description"]) else ""
                    
                    if val_desc.lower().startswith("not useful"):
                        val_desc = val_desc[len("not useful"):].strip()
                        
                    table_description[orig_col] = {
                        "column_description": col_desc,
                        "value_description": val_desc
                    }
                database_description[table_name_lower] = table_description
            except Exception as e:
                logger.warning(f"Failed to load description for table {table_name_lower}: {e}")
                
        return database_description

    @staticmethod
    def get_db_ddl(db_path: str) -> str:
        """
        Extracts the CREATE TABLE statements (DDL) for all tables in the database.
        """
        import sqlite3
        ddls = []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            for row in cursor.fetchall():
                if row[0]:
                    ddls.append(row[0].strip() + ";")
        return "\n\n".join(ddls)

    @staticmethod
    def to_readable_text(
        schema: Dict[str, TableSchema], 
        include_stats: bool = True, 
        include_examples: bool = True,
        include_descriptions: bool = True
    ) -> str:
        """Converts the structured schema into a prompt-friendly string."""
        output = []
        all_fks = []

        for table_name, table in schema.items():
            output.append(f"Table {table_name}:")
            # Prioritize PK columns at the top of the column list
            sorted_columns = sorted(table.columns.items(), key=lambda x: not x[1].is_pk)
            
            for col_name, col in sorted_columns:
                pk_str = " [PK]" if col.is_pk else ""
                
                stats_str = ""
                if include_stats:
                    stats_str = f"(Distinct: {col.num_distinct}, Nulls: {col.num_null})"
                
                examples_str = ""
                if include_examples and col.examples:
                    examples_str = f" | Examples: {col.examples}"
                
                desc_str = ""
                if include_descriptions:
                    if col.description:
                        desc_str += f" | Column Description: {col.description}"
                    if col.value_description:
                        desc_str += f" | Value Description: {col.value_description}"
                
                output.append(f"  - {col_name} ({col.dtype}){pk_str} {stats_str}{examples_str}{desc_str}")
            
            # Collect Foreign Keys for later
            if table.foreign_keys:
                for fk in table.foreign_keys:
                    all_fks.append(f"- {table_name}.{fk.from_col} -> {fk.to_table}.{fk.to_col}")
        
        # Append Foreign Keys section at the end
        if all_fks:
            output.append("\nForeign Key Relationships:")
            output.extend(all_fks)

        return "\n".join(output)


