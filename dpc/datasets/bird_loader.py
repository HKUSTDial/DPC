from typing import List, Dict, Any
from .base_dataset import BaseDataset, TextToSQLItem

class BirdLoader(BaseDataset):
    """
    Dataset loader for the BIRD dataset.
    BIRD includes 'evidence' which is crucial for query generation.
    """

    def get_item(self, index: int) -> TextToSQLItem:
        item = self.data[index]
        # BIRD usually has a 'question_id' field
        question_id = str(item.get("question_id", index))
        
        return TextToSQLItem(
            question_id=question_id,
            question=item["question"],
            db_id=item["db_id"],
            ground_truth=item.get("SQL", ""),
            evidence=item.get("evidence", ""),
            difficulty=item.get("difficulty", "unknown"),
            additional_info=item
        )


