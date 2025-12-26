from typing import List, Dict, Any
from .base_dataset import BaseDataset, TextToSQLItem

class SpiderLoader(BaseDataset):
    """
    Dataset loader for the Spider dataset.
    """

    def get_item(self, index: int) -> TextToSQLItem:
        item = self.data[index]
        return TextToSQLItem(
            question=item["question"],
            db_id=item["db_id"],
            ground_truth=item.get("query", ""),
            evidence="",
            difficulty=item.get("difficulty", "unknown"),
            additional_info=item
        )


