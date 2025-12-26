from typing import List, Dict, Any
from .base_dataset import BaseDataset, TextToSQLItem

class BirdLoader(BaseDataset):
    """
    Dataset loader for the BIRD dataset.
    BIRD includes 'evidence' which is crucial for query generation.
    """

    def get_item(self, index: int) -> TextToSQLItem:
        item = self.data[index]
        return TextToSQLItem(
            question=item["question"],
            db_id=item["db_id"],
            ground_truth=item.get("SQL", ""),
            evidence=item.get("evidence", ""),
            difficulty=item.get("difficulty", "unknown"),
            additional_info=item
        )


