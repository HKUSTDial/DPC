import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Union, Set
from decimal import Decimal
from datetime import datetime, date

def normalize_value(val: Any) -> Any:
    """
    Normalizes a single value for robust comparison.
    - Handles None/NaN
    - Handles Decimals and Floats (rounding to 4 places)
    - Handles Dates and Timestamps (standardizing format)
    - Strips strings
    """
    # 1. Handle Nulls
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).lower() in ["nan", "none", "null"]:
        return None
    
    # 2. Handle Numeric types
    if isinstance(val, Decimal):
        val = float(val)
        
    if isinstance(val, (float, np.float64, np.float32)):
        return round(float(val), 4)
    
    if isinstance(val, (int, np.int64, np.int32)):
        return int(val)
        
    # 3. Handle Date/Time types
    # Common in Pandas (pd.Timestamp) and SQL (datetime.date/datetime.datetime)
    if isinstance(val, (datetime, date, pd.Timestamp)):
        # If it's just a date (no time component or time is 00:00:00), format as YYYY-MM-DD
        if isinstance(val, datetime) and val.hour == 0 and val.minute == 0 and val.second == 0:
            return val.strftime("%Y-%m-%d")
        elif isinstance(val, date) and not isinstance(val, datetime):
            return val.strftime("%Y-%m-%d")
        else:
            # Otherwise keep the full ISO-like format but normalized
            return val.strftime("%Y-%m-%d %H:%M:%S")

    # 4. Handle Strings and others
    return str(val).strip()

def normalize_result(data: Any) -> List[Tuple[Any, ...]]:
    """
    Standardizes execution results from SQL or Pandas into a List of Tuples.
    """
    if data is None:
        return []
        
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to list of tuples
        # We use itertuples or values.tolist()
        rows = data.values.tolist()
    elif isinstance(data, list):
        rows = data
    else:
        # Scalar value
        rows = [[data]]
        
    normalized_rows = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            row = [row]
        normalized_rows.append(tuple(normalize_value(v) for v in row))
        
    return normalized_rows

def calculate_row_match(predicted_row: Tuple[Any, ...], ground_truth_row: Tuple[Any, ...]) -> Tuple[float, float, float]:
    """
    BIRD official logic: Calculate the matching percentage for a single row.
    """
    if not ground_truth_row:
        return 0.0, 1.0, 0.0
        
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    
    # Check elements in predicted row
    temp_gt = list(ground_truth_row)
    for pred_val in predicted_row:
        if pred_val in temp_gt:
            matches += 1
            temp_gt.remove(pred_val) # Remove to handle duplicates correctly
        else:
            element_in_pred_only += 1
            
    # Elements in truth that were not matched
    element_in_truth_only = len(temp_gt)
    
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    
    return match_percentage, pred_only_percentage, truth_only_percentage

def calculate_soft_f1(predicted: List[Tuple[Any, ...]], ground_truth: List[Tuple[Any, ...]]) -> float:
    """
    Calculates the Soft-F1 score between two result sets (List of Tuples).
    Based on BIRD official evaluation logic.
    """
    # Normalized empty check
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    # Drop duplicates while preserving order (as per BIRD logic)
    # Using dict.fromkeys to maintain order of first appearance
    predicted = list(dict.fromkeys(predicted))
    ground_truth = list(dict.fromkeys(ground_truth))

    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    
    # Calculate scores for pairs based on index
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):
            match_scores.append(0.0)
            truth_only_scores.append(1.0)
            continue
            
        pred_row = predicted[i]
        m, p, t = calculate_row_match(pred_row, gt_row)
        match_scores.append(m)
        pred_only_scores.append(p)
        truth_only_scores.append(t)

    # Handle remaining predicted rows
    if len(predicted) > len(ground_truth):
        for i in range(len(ground_truth), len(predicted)):
            match_scores.append(0.0)
            pred_only_scores.append(1.0)
            truth_only_scores.append(0.0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    return f1_score

class DPCEvaluator:
    """
    Evaluator that standardizes different output formats and computes Soft-F1.
    """
    @staticmethod
    def evaluate(pred: Any, gold: Any) -> float:
        """
        Main entry point for comparison.
        pred and gold can be DataFrame, List[Tuple], or Scalar.
        """
        norm_pred = normalize_result(pred)
        norm_gold = normalize_result(gold)
        return calculate_soft_f1(norm_pred, norm_gold)

