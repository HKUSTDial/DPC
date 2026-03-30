import json
import re
from typing import Any, Dict


def extract_result_block(response: str) -> str:
    """
    Extract content after the last <result> tag.
    Falls back to the full response if no tag is present.
    """
    start_tag = "<result>"
    end_tag = "</result>"

    idx = response.rfind(start_tag)
    if idx == -1:
        return response.strip()

    content = response[idx + len(start_tag):]
    end_idx = content.find(end_tag)
    if end_idx != -1:
        return content[:end_idx].strip()
    return content.strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _remove_json_comments(text: str) -> str:
    pattern = r'("(?:\\.|[^"\\])*")|//.*|/\*[\s\S]*?\*/'

    def replacer(match: re.Match[str]) -> str:
        if match.group(1):
            return match.group(1)
        return ""

    return re.sub(pattern, replacer, text)


def _find_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON object found in response.")


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from a model response, preferring <result> blocks but tolerating
    extra prose/code fences when the prompt did not require tags.
    """
    candidate = _remove_json_comments(_strip_code_fences(extract_result_block(response)))

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        json_str = _find_balanced_json_object(candidate)
        parsed = json.loads(json_str)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object in model response.")
    return parsed
