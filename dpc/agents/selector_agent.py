import logging
from typing import List, Dict, Any, Optional, Tuple
from .base_agent import BaseAgent
from ..prompts.factory import PromptFactory
from ..utils.schema_utils import TableSchema, SchemaExtractor

logger = logging.getLogger(__name__)


class EquivalenceGrouperAgent(BaseAgent):
    """
    Groups SQL candidates by semantic equivalence without SQL execution.
    Supports SC by sampling multiple groupings and merging them by pairwise co-occurrence votes.
    """

    def run(
        self,
        question: str,
        candidate_sqls: List[str],
        full_schema: Dict[str, TableSchema],
        evidence: Optional[str] = None,
        max_correction_attempts: int = 2,
        num_grouping_attempts: int = 1
    ) -> Dict[str, Any]:
        if not candidate_sqls:
            raise ValueError("candidate_sqls is empty.")
        if num_grouping_attempts < 1:
            raise ValueError("num_grouping_attempts must be >= 1.")
        if len(candidate_sqls) == 1:
            return {
                "groups": [{"rank": 1, "member_indices": [1]}],
                "sql_groups": [[candidate_sqls[0]]],
                "equivalence_scores": {"1": 0}
            }

        full_schema_text = SchemaExtractor.to_readable_text(
            full_schema,
            include_stats=True,
            include_examples=True,
            include_descriptions=True
        )

        sampled_groups = []
        for i in range(num_grouping_attempts):
            logger.info("[EquivalenceGrouperAgent] SC sample %s/%s...", i + 1, num_grouping_attempts)
            try:
                groups = self._run_single_grouping(
                    question=question,
                    candidate_sqls=candidate_sqls,
                    full_schema_text=full_schema_text,
                    evidence=evidence,
                    max_correction_attempts=max_correction_attempts
                )
                sampled_groups.append(groups)
            except Exception as e:
                logger.warning("[EquivalenceGrouperAgent] SC sample %s failed: %s", i + 1, e)

        if not sampled_groups:
            raise ValueError("All grouping SC samples failed.")

        merged_groups, eq_scores = self._merge_groups_with_sc(sampled_groups, len(candidate_sqls))
        sql_groups = [[candidate_sqls[idx - 1] for idx in g] for g in merged_groups]
        groups_payload = [{"rank": i + 1, "member_indices": g} for i, g in enumerate(merged_groups)]

        return {
            "groups": groups_payload,
            "sql_groups": sql_groups,
            "equivalence_scores": {str(i + 1): eq_scores[i + 1] for i in range(len(candidate_sqls))}
        }

    def _run_single_grouping(
        self,
        question: str,
        candidate_sqls: List[str],
        full_schema_text: str,
        evidence: Optional[str],
        max_correction_attempts: int
    ) -> List[List[int]]:
        messages = PromptFactory.get_equivalence_grouper_prompt(
            question=question,
            candidate_sqls=candidate_sqls,
            full_schema_text=full_schema_text,
            evidence=evidence
        )

        for attempt in range(max_correction_attempts + 1):
            try:
                logger.info(f"[EquivalenceGrouperAgent] Grouping attempt {attempt + 1}/{max_correction_attempts + 1}...")
                response_text = self.llm.ask(messages)
                messages.append({"role": "assistant", "content": response_text})

                parsed = self._parse_json_response(response_text)
                groups = self._validate_and_normalize_groups(parsed, len(candidate_sqls))
                logger.info("[EquivalenceGrouperAgent] Parsed %s groups.", len(groups))
                return groups
            except Exception as e:
                if attempt < max_correction_attempts:
                    messages.extend(PromptFactory.get_equivalence_grouper_retry_prompt(str(e)))
                else:
                    raise e

        raise ValueError("EquivalenceGrouperAgent single grouping failed unexpectedly.")

    def _validate_and_normalize_groups(self, parsed: Dict[str, Any], n_candidates: int) -> List[List[int]]:
        if not isinstance(parsed, dict):
            raise ValueError("Grouping response must be a JSON object.")
        groups = parsed.get("groups")
        if not isinstance(groups, list):
            raise ValueError("groups must be a list.")

        assigned = set()
        normalized: List[List[int]] = []

        for g in groups:
            if not isinstance(g, dict):
                continue
            members = g.get("member_indices")
            if not isinstance(members, list):
                continue

            clean_members: List[int] = []
            for idx in members:
                if not isinstance(idx, int):
                    continue
                if idx < 1 or idx > n_candidates:
                    continue
                if idx in assigned:
                    continue
                assigned.add(idx)
                clean_members.append(idx)

            if clean_members:
                normalized.append(clean_members)

        # Add missing indices as singleton groups for full coverage.
        for idx in range(1, n_candidates + 1):
            if idx not in assigned:
                normalized.append([idx])

        # Deterministic ordering: larger groups first, then smallest index.
        normalized.sort(key=lambda g: (-len(g), min(g)))
        return normalized

    def _merge_groups_with_sc(
        self,
        sampled_groups: List[List[List[int]]],
        n_candidates: int
    ) -> Tuple[List[List[int]], Dict[int, int]]:
        if len(sampled_groups) == 1:
            groups = sampled_groups[0]
            eq_scores = {idx: 0 for idx in range(1, n_candidates + 1)}
            for group in groups:
                for i in group:
                    eq_scores[i] += len(group) - 1
            return groups, eq_scores

        pair_votes: Dict[Tuple[int, int], int] = {}
        eq_scores = {idx: 0 for idx in range(1, n_candidates + 1)}

        for groups in sampled_groups:
            for group in groups:
                uniq = sorted(set(group))
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        a, b = uniq[i], uniq[j]
                        pair_votes[(a, b)] = pair_votes.get((a, b), 0) + 1

        for (a, b), cnt in pair_votes.items():
            eq_scores[a] += cnt
            eq_scores[b] += cnt

        threshold = len(sampled_groups) // 2 + 1

        parent = {i: i for i in range(1, n_candidates + 1)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (a, b), cnt in pair_votes.items():
            if cnt >= threshold:
                union(a, b)

        components: Dict[int, List[int]] = {}
        for idx in range(1, n_candidates + 1):
            root = find(idx)
            components.setdefault(root, []).append(idx)

        merged_groups = []
        for members in components.values():
            members.sort(key=lambda i: (-eq_scores[i], i))
            merged_groups.append(members)

        merged_groups.sort(key=lambda g: (-len(g), min(g)))

        logger.info(
            "[EquivalenceGrouperAgent] SC merged %s samples into %s groups (majority threshold=%s).",
            len(sampled_groups),
            len(merged_groups),
            threshold
        )
        return merged_groups, eq_scores
