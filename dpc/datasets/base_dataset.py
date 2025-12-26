from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import os
import json
from dataclasses import dataclass
from ..utils.schema_utils import SchemaExtractor, TableSchema

@dataclass
class TextToSQLItem:
    """Standardized data item for Text-to-SQL tasks."""
    question_id: str
    question: str
    db_id: str
    ground_truth: str
    evidence: Optional[str] = ""
    difficulty: Optional[str] = "unknown"
    additional_info: Dict[str, Any] = None

class BaseDataset(ABC):
    """
    Abstract base class for dataset loaders.
    Provides a unified interface for different Text-to-SQL datasets.
    """

    def __init__(self, data_path: str, db_root_path: str):
        """
        Initialize the dataset loader.

        Args:
            data_path: Path to the query data (e.g., dev.json).
            db_root_path: Directory containing the database folders. 
                          Expected structure: db_root_path/db_id/db_id.sqlite
        """
        self.data_path = data_path
        self.db_root_path = db_root_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Common data loading logic for JSON-based datasets."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @abstractmethod
    def get_item(self, index: int) -> TextToSQLItem:
        """
        Get a single evaluation item as a standardized TextToSQLItem.
        """
        pass

    def get_db_path(self, db_id: str) -> str:
        """Get the absolute path to the SQLite database file for a given db_id."""
        return os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

    def get_schema(self, db_id: str) -> Dict[str, TableSchema]:
        """
        Dynamically extracts schema from the database file using SchemaExtractor.
        Uses internal caching in SchemaExtractor to avoid redundant I/O.
        """
        db_path = self.get_db_path(db_id)
        return SchemaExtractor.extract(db_path)

    def __len__(self) -> int:
        return len(self.data)


