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
    Calculate the matching percentage for a single row based on column position.
    Strictly column-order sensitive.
    """
    if not ground_truth_row:
        return (0.0, 1.0, 0.0) if predicted_row else (1.0, 0.0, 0.0)
        
    total_columns = len(ground_truth_row)
    matches = 0
    
    # Strict positional matching
    for i in range(min(len(predicted_row), total_columns)):
        if predicted_row[i] == ground_truth_row[i]:
            matches += 1
            
    # False Positives: Columns in predicted that don't match or are extra
    element_in_pred_only = len(predicted_row) - matches
    # False Negatives: Columns in truth that were not matched
    element_in_truth_only = total_columns - matches
    
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    
    return match_percentage, pred_only_percentage, truth_only_percentage

def calculate_soft_f1(predicted: List[Tuple[Any, ...]], ground_truth: List[Tuple[Any, ...]]) -> float:
    """
    Calculates the Soft-F1 score between two result sets (List of Tuples).
    Uses the Hungarian Algorithm (via scipy.optimize.linear_sum_assignment)
    to find the GLOBAL OPTIMAL matching between rows.
    """
    # Normalized empty check
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    # Canonicalize results for robust comparison
    # predicted = canonicalize_result(predicted)
    # ground_truth = canonicalize_result(ground_truth)

    n_pred = len(predicted)
    n_gt = len(ground_truth)
    
    # We need a square cost matrix for the assignment problem if we want to penalize
    # unmatched rows purely by the algorithm, but standard implementation allows rectangular matrices.
    # However, to correctly account for FP and FN rows, we'll compute costs between all pairs.
    
    # Cost Matrix Construction:
    # We want to MAXIMIZE match score, so we MINIMIZE cost.
    # Cost = 1.0 - match_score
    
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cost_matrix = np.zeros((n_pred, n_gt))
    
    # Pre-calculate match details to avoid re-computation
    # match_details[i][j] = (match_percentage, pred_only, truth_only)
    match_details = {}

    for i in range(n_pred):
        for j in range(n_gt):
            m, p, t = calculate_row_match(predicted[i], ground_truth[j])
            match_details[(i, j)] = (m, p, t)
            # Cost is the "distance" from a perfect match
            cost_matrix[i, j] = 1.0 - m

    # Apply Hungarian Algorithm
    # row_ind: indices in predicted
    # col_ind: indices in ground_truth
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate scores based on the optimal assignment
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    
    matched_pred_indices = set(row_ind)
    matched_gt_indices = set(col_ind)
    
    # Accumulate scores for matched pairs
    for r, c in zip(row_ind, col_ind):
        m, p, t = match_details[(r, c)]
        tp_sum += m
        fp_sum += p
        fn_sum += t
        
    # Handle unmatched rows
    
    # Unmatched in Predicted -> All elements are False Positives
    # In calculate_row_match logic: (0.0 match, 1.0 pred_only, 0.0 truth_only) for completely new row vs empty
    # But effectively, each column in an unmatched predicted row is a False Positive.
    # Let's align with the previous logic: 
    # For a row that has NO match in GT, it contributes 0.0 to TP, 1.0 * len(row) ? 
    # Wait, calculate_row_match normalizes by len(ground_truth_row).
    # If a predicted row is extra, it's fully False Positive.
    
    # Consistent Logic with previous greedy:
    # "Remaining predicted rows are false positives" -> pred_only_scores.append(1.0)
    for i in range(n_pred):
        if i not in matched_pred_indices:
            # Entire row is False Positive
            tp_sum += 0.0
            fp_sum += 1.0
            fn_sum += 0.0
            
    # Unmatched in Ground Truth -> All elements are False Negatives
    # "Remaining GT rows are false negatives" -> truth_only_scores.append(1.0)
    for j in range(n_gt):
        if j not in matched_gt_indices:
            # Entire row is False Negative
            tp_sum += 0.0
            fp_sum += 0.0
            fn_sum += 1.0

    # Final F1 Calculation
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
        
    return 2 * precision * recall / (precision + recall)

class DPCEvaluator:
    """
    Evaluator that standardizes different output formats and computes Soft-F1.
    """
    @staticmethod
    def evaluate(pred: Any, gold: Any, metric: str = "bs_f1") -> float:
        """
        Main entry point for comparison.
        pred and gold can be DataFrame, List[Tuple], or Scalar.
        """
        metric = metric.lower()

        if metric == "bs_f1":
            norm_pred = normalize_result(pred)
            norm_gold = normalize_result(gold)
            return calculate_soft_f1(norm_pred, norm_gold)
        if metric == "ex":
            # Traditional EX: strict set(tuple(rows)) equivalence without value normalization.
            def to_raw_row_set(x: Any) -> Set[Tuple[Any, ...]]:
                if x is None:
                    return set()
                if isinstance(x, pd.DataFrame):
                    rows = x.values.tolist()
                elif isinstance(x, list):
                    rows = x
                else:
                    rows = [[x]]

                out = set()
                for row in rows:
                    if isinstance(row, (list, tuple)):
                        out.add(tuple(row))
                    else:
                        out.add((row,))
                return out

            return 1.0 if to_raw_row_set(pred) == to_raw_row_set(gold) else 0.0
        raise ValueError(f"Unsupported metric: {metric}. Expected one of ['bs_f1', 'ex'].")

