from typing import List, Dict, Any
from .base_dataset import BaseDataset, TextToSQLItem

class SpiderLoader(BaseDataset):
    """
    Dataset loader for the Spider dataset.
    """

    def get_item(self, index: int) -> TextToSQLItem:
        item = self.data[index]
        # Spider usually doesn't have a unique question_id, use index as default
        question_id = str(item.get("question_id", index))
        
        return TextToSQLItem(
            question_id=question_id,
            question=item["question"],
            db_id=item["db_id"],
            ground_truth=item.get("query", ""),
            evidence="",
            difficulty=item.get("difficulty", "unknown"),
            additional_info=item
        )


