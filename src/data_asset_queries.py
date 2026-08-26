from __future__ import annotations

import json
import logging
import re
import string
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    """The two top-level request routes supported by the application."""

    SQL_AND_RAG = "sql_and_rag"
    DATA_ASSET_CATALOG = "data_asset_catalog"


# Stage 1: substantive business/data signals have absolute routing priority.
# False positives are safe because SQL/RAG is also the conservative default.
_BUSINESS_REQUEST_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(spend|spent|cost|revenue|sales|invoice|transaction|supplier|contract|procurement)\b",
        r"\b(total|sum|average|mean|median|percentage|percent|ratio|count|number of)\b",
        r"\b(compare|compared|comparison|versus|vs\.?|difference|breakdown|trend)\b",
        r"\b(calculate|compute|measure|analyse|analyze)\b",
        r"\b(top|bottom|highest|lowest|most|least)(?:\s+\d+)?\b",
        r"\b(by|per|between|during|since|before|after)\s+(supplier|category|month|quarter|year|date|region|department|organisation|organization)\b",
        r"\b(SME(?:s)?|small and medium enterprises?)\b",
        r"\b(target|actual|variance|forecast|flagged)\b",
        r"\b(how much|how many)\b",
        r"(?:£|\$|€)\s*\d|\b\d+(?:[.,]\d+)?\s*%",
    )
)

# Stage 2 is an allowlist of complete capability requests, not a topic
# classifier. Full-message matching stops capability wording embedded in a
# larger request from diverting that request away from SQL/RAG.
_CAPABILITY_QUERY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"what (?:kind(?:s)?|type(?:s)?) of questions can I ask(?: (?:you|this (?:agent|assistant|model)))?",
        r"what (?:questions|things) can (?:I|we) ask(?: (?:you|this (?:agent|assistant|model)))?",
        r"what can I ask",
        r"(?:give|show|provide|suggest) (?:me )?(?:some )?(?:sample|example) questions",
        r"what (?:can|are) (?:you|this (?:agent|assistant|model)) (?:do|capable of)",
        r"what (?:are )?(?:your|the (?:agent(?:'s)?|assistant(?:'s)?|model(?:'s)?)) capabilities",
        r"what (?:data|datasets?|data sets?|databases?|tables?|documents?|sources?) (?:do|can) (?:you|this (?:agent|assistant|model)) (?:have|access|use|query|see)(?: access to)?",
        r"what (?:data|datasets?|data sets?|databases?|tables?|documents?|sources?) (?:are|is) (?:you|this (?:agent|assistant|model)) (?:using|able to access|connected to)(?: to give answers with)?",
        r"what (?:data|datasets?|data sets?|databases?|tables?|documents?|sources?) (?:are|is) available(?: to (?:you|this (?:agent|assistant|model)))?",
        r"which (?:data|datasets?|data sets?|databases?|tables?|documents?|sources?) (?:do|can) (?:you|this (?:agent|assistant|model)) (?:have|access|use|query|see)(?: access to)?",
        r"(?:show|list|describe) (?:your|the )?(?:available )?(?:data sources?|data|datasets?|data sets?|databases?|tables?|documents?|sources?|capabilities)",
        r"help(?: me)?(?: (?:use|understand|get started with) (?:this|the) (?:agent|assistant|model))?",
        r"how (?:do|can|should) I use (?:you|this|the) (?:agent|assistant|model)",
        r"what do you know",
    )
)

_CAPABILITY_REFERENCE_QUESTIONS = (
    "what kind of questions can i ask",
    "what questions can i ask you",
    "what can i ask",
    "give me sample questions",
    "what can you do",
    "what are your capabilities",
    "what data do you have access to",
    "what datasets can you query",
    "show available data sources",
    "show available databases",
    "what data is available",
    "help me use this assistant",
    "what do you know",
)
_FUZZY_CAPABILITY_THRESHOLD = 82.0
_MAX_FUZZY_QUERY_WORDS = 10
_EDGE_PUNCTUATION = string.punctuation + "“”‘’…–—"


def _normalise_for_routing(user_input: str) -> str:
    """Create a temporary comparison string without changing display input."""
    if not isinstance(user_input, str):
        return ""
    text = " ".join(user_input.strip().split())
    text = text.strip(_EDGE_PUNCTUATION + " ")
    text = re.sub(
        r"^(?:please\s+|could you\s+|can you\s+tell me\s+)", "", text, flags=re.I
    )
    return text.strip(_EDGE_PUNCTUATION + " ").casefold()


def is_business_data_request(user_input: str) -> bool:
    """Return whether input contains substantive data/metric request signals."""
    normalised_input = _normalise_for_routing(user_input)
    return bool(normalised_input) and any(
        pattern.search(normalised_input) for pattern in _BUSINESS_REQUEST_PATTERNS
    )


def is_fuzzy_capability_match(
    user_input: str, threshold: float = _FUZZY_CAPABILITY_THRESHOLD
) -> bool:
    """Match typo-heavy complete capability questions without mutating input.

    ``ratio`` and ``token_sort_ratio`` tolerate spelling mistakes and reordered
    words while penalising additional business context. Deliberately avoid
    token-set/partial matching: those algorithms can score a capability phrase
    embedded in a larger business question as 100%, causing false routing.
    """
    normalised_input = _normalise_for_routing(user_input)
    if not normalised_input or len(normalised_input.split()) > _MAX_FUZZY_QUERY_WORDS:
        return False

    return any(
        max(
            fuzz.ratio(normalised_input, reference),
            fuzz.token_sort_ratio(normalised_input, reference),
        )
        > threshold
        for reference in _CAPABILITY_REFERENCE_QUESTIONS
    )


def is_explicit_capability_query(user_input: str) -> bool:
    """Return whether the complete input explicitly requests capability help."""
    normalised_input = _normalise_for_routing(user_input)
    if not normalised_input:
        return False
    return any(
        pattern.fullmatch(normalised_input) for pattern in _CAPABILITY_QUERY_PATTERNS
    ) or is_fuzzy_capability_match(normalised_input)


def classify_query_route(user_input: str) -> QueryRoute:
    """Apply Business Request -> Explicit Capability -> default SQL/RAG.

    Routing is deterministic and never invokes an LLM or network service.
    Normalisation exists only in this call path; the original input is neither
    returned nor modified.
    """
    normalised_input = _normalise_for_routing(user_input)

    if is_business_data_request(normalised_input):
        route = QueryRoute.SQL_AND_RAG
        reason = "business_request"
    elif is_explicit_capability_query(normalised_input):
        route = QueryRoute.DATA_ASSET_CATALOG
        reason = "explicit_capability_request"
    else:
        route = QueryRoute.SQL_AND_RAG
        reason = "default"

    logger.info("QUERY_ROUTE_SELECTED route=%s reason=%s", route.value, reason)
    return route


def build_user_display_payload(user_input: str) -> dict[str, Any]:
    """Build a frontend payload preserving the exact original input string."""
    return {
        "role": "user",
        "content": user_input,
        "raw_content": user_input,
        "sources": "",
        "full_table": None,
    }


def is_data_asset_query(user_input: str, llm: Any = None) -> bool:
    """Backward-compatible deterministic wrapper; ``llm`` is never invoked."""
    return classify_query_route(user_input) is QueryRoute.DATA_ASSET_CATALOG


def load_data_assets(path: str | Path) -> dict[str, str]:
    """Load a mapping of display names to public-safe asset descriptions."""
    asset_path = Path(path)
    with asset_path.open(encoding="utf-8") as asset_file:
        payload = json.load(asset_file)

    if not isinstance(payload, dict) or not payload:
        raise ValueError("data_assets.json must contain a non-empty JSON object")

    assets: dict[str, str] = {}
    for name, description in payload.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every data asset must have a non-empty string name")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Data asset '{name}' must have a non-empty description")
        assets[name.strip()] = description.strip()
    return assets


def build_data_asset_answer(assets: Mapping[str, str]) -> str:
    """Build a deterministic answer without sending metadata to Vanna/RAG."""
    if not assets:
        return (
            "I couldn't load the data catalogue right now. Please try again later "
            "or contact the service owner."
        )

    asset_lines = [f"- **{name}:** {description}" for name, description in assets.items()]
    return "\n".join(
        [
            "I can answer questions using these data assets:",
            *asset_lines,
            "",
            "**For example, you can ask:**",
            "- What was the total evidenced spend for this year?",
            "- Can you give me top 5 companies with the highest spend?",
            "- Who is the CEO of BAE system?",
        ]
    )


def answer_data_asset_query(user_input: str, path: str | Path) -> str:
    """Load the catalogue and answer an already-classified metadata question."""
    try:
        assets = load_data_assets(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception(
            "DATA_ASSET_CATALOG_LOAD_FAILED path=%s error_type=%s",
            path,
            type(exc).__name__,
        )
        return build_data_asset_answer({})

    logger.info("DATA_ASSET_QUERY_HANDLED asset_count=%d", len(assets))
    return build_data_asset_answer(assets)
