import json
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import re
import os


SQL_entities_folder = Path(__file__).parents[1] / "SQL_entities"

# Deterministic, high-confidence aliases. Keys are matched as complete words only.
DOMAIN_ACRONYMS: dict[str, str] = {
    "MOD": "Ministry of Defence",
    "HMRC": "HM Revenue and Customs",
    "NHS": "National Health Service",
    "DWP": "Department for Work and Pensions",
    "DFT": "Department for Transport",
    "DFE": "Department for Education",
    "HO": "Home Office",
    "MOJ": "Ministry of Justice",
}


def expand_domain_acronyms(text: str) -> str:
    """Expand known organisation acronyms using strict word boundaries."""
    expanded = text
    for acronym, canonical_name in DOMAIN_ACRONYMS.items():
        expanded = re.sub(
            rf"(?<!\w){re.escape(acronym)}(?!\w)",
            lambda _match, value=canonical_name: value,
            expanded,
            flags=re.IGNORECASE,
        )
    return expanded


def format_financial_years(text: str) -> str:
    """Force financial year variations into the strict ``YYYY/YY`` structure."""
    long_range_pattern = r"\b(\d{4})\s*/\s*(\d{2})(\d{2})\b"

    def collapse_long_range(match):
        return f"{match.group(1)}/{match.group(3)}"

    text = re.sub(long_range_pattern, collapse_long_range, text)

    short_range_pattern = r"\b(\d{2})\s*/\s*(\d{2})\b(?!\s/)"

    def expand_short_range(match):
        return f"20{match.group(1)}/{match.group(2)}"

    text = re.sub(short_range_pattern, expand_short_range, text)

    four_digit_pattern = r"\b(19|20)(\d{2})\b(?!\s/)"

    def expand_standalone_year(match):
        century = match.group(1)
        short_yy = int(match.group(2))
        next_yy = (short_yy + 1) % 100
        return f"{century}{short_yy:02d}/{next_yy:02d}"

    return re.sub(four_digit_pattern, expand_standalone_year, text)


def _replace_phrase_at_word_boundaries(text: str, phrase: str, replacement: str) -> str:
    """Replace a correction only when the complete phrase is present."""
    if not phrase:
        return text
    pattern = re.compile(
        rf"(?<!\w){re.escape(str(phrase))}(?!\w)", flags=re.IGNORECASE
    )
    return pattern.sub(lambda _match: str(replacement), text)


def spell_correct_user_query(
    user_input: str,
    llm: Any,
    json_name: str = "dummy_entities.json",
    catalog_data: dict[str, Any] | None = None,
) -> str:
    """Fix financial years and map shorthand to official entity names.

    Known acronyms are expanded deterministically before invoking the translator,
    so resolution does not depend on aliases being present in the JSON catalog.
    """
    processed_input = expand_domain_acronyms(format_financial_years(user_input))

    if catalog_data is None:
        json_path = os.path.join(SQL_entities_folder, json_name)
        try:
            with open(json_path, "r") as f:
                valid_vocabulary = json.load(f)
        except Exception as err:
            print(f"⚠️ Dictionary read failure: {err}. Returning date-patched input.")
            return processed_input
    else:
        valid_vocabulary = catalog_data

    system_prompt = (
        "You are a strict text entity translator. Analyze the user's input string "
        "and check if any words are misspellings, abbreviations, or casing variants of our official database entries.\n\n"
        f"--- OFFICIAL DATABASE ENTRIES CATALOG ---\n{json.dumps(valid_vocabulary)}\n\n"
        "INSTRUCTIONS:\n"
        "1. Identify text chunks matching or approximating any supplier or framework in our catalog.\n"
        "2. Output a RAW, flat JSON mapping dictionary associating the identified typo or shorthand word "
        'with its exact official translation string (e.g., {"BAEs": "BAE SYSTEMS APPLIED INTELLIGENCE LIMITED"}).\n'
        "3. Ignore financial years or numerical dates entirely; they have already been pre-formatted.\n"
        "4. If all names are perfectly spelled, or no terms match, output an empty JSON block: {}.\n"
        "5. Do not alter already-expanded government organisation names such as Ministry of Defence.\n"
        "6. CRITICAL: Output ONLY valid JSON syntax. Do not write markdown, JSON fences, or commentary."
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=processed_input)]
        )
        clean_json_string = (
            response.content.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        corrections = json.loads(clean_json_string)
        if not isinstance(corrections, dict):
            raise ValueError("Entity translator response must be a JSON object")

        corrected_query = processed_input
        for typo_chunk, official_row_value in corrections.items():
            corrected_query = _replace_phrase_at_word_boundaries(
                corrected_query, str(typo_chunk), str(official_row_value)
            )
        return corrected_query
    except Exception as err:
        print(f"⚠️ Grammar translation fallback active: {err}")
        return processed_input


def _catalog_entries(catalog_data: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    for key in ("suppliers", "frameworks", "customers"):
        values = catalog_data.get(key, [])
        if isinstance(values, list):
            entries.extend(value for value in values if isinstance(value, str))
    return entries


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def harden_vanna_sql(
    sql_str: str,
    json_name: str = "dummy_entities.json",
    catalog_data: dict[str, Any] | None = None,
) -> str:
    """Improve generated SQL without fuzzy or substring entity rewrites.

    Catalog replacement is exact and case-insensitive. Known standalone acronym
    literals are an explicit high-confidence exception. Organisation predicates
    tolerate casing, surrounding text and spacing variations. No training occurs.
    """
    if not sql_str:
        return sql_str

    if catalog_data is None:
        try:
            json_path = SQL_entities_folder / json_name
            with open(json_path, "r") as f:
                catalog_data = json.load(f)
        except Exception as err:
            print(f"⚠️ SQL post-processor failed to load entity catalog: {err}")
            catalog_data = {}

    exact_catalog = {
        entry.strip().casefold(): entry for entry in _catalog_entries(catalog_data or {})
    }
    acronym_lookup = {key.casefold(): value for key, value in DOMAIN_ACRONYMS.items()}
    canonical_to_acronym = {
        canonical.casefold(): acronym for acronym, canonical in DOMAIN_ACRONYMS.items()
    }

    # Limit rewriting to known entity columns and simple single-quoted equality
    # predicates. This intentionally excludes arbitrary literals and expressions.
    entity_pattern = re.compile(
        r"(?P<column>(?:(?:\[[^\]]+\]|\w+)\s*\.\s*)?"
        r"(?P<field>\[?(?:SupplierName|CustomerName|Framework)\]?))"
        r"\s*=\s*'(?P<value>(?:''|[^'])*)'",
        flags=re.IGNORECASE,
    )

    def harden_entity_predicate(match: re.Match) -> str:
        column = match.group("column")
        field = match.group("field").strip("[]").casefold()
        current_value = match.group("value").replace("''", "'").strip()
        folded_value = current_value.casefold()

        # Exact catalog matches may restore official casing. Substring matches are
        # forbidden: "MOD" must never select a historic entry merely ending in MOD.
        resolved_value = exact_catalog.get(folded_value, current_value)
        if folded_value in acronym_lookup:
            resolved_value = acronym_lookup[folded_value]

        escaped = _escape_sql_literal(resolved_value)
        if field == "framework":
            return f"{column} = '{escaped}'"

        compact = _escape_sql_literal(re.sub(r"\s+", "", resolved_value))
        alternatives = [
            f"LOWER({column}) = LOWER('{escaped}')",
            f"LOWER({column}) LIKE LOWER('%{escaped}%')",
            f"LOWER(REPLACE({column}, ' ', '')) LIKE LOWER('%{compact}%')",
        ]

        # A canonical name may still be stored as the bare acronym. Match only
        # exact acronym values; never use '%MOD%', which could include dead agencies.
        acronym = canonical_to_acronym.get(resolved_value.casefold())
        if acronym:
            alternatives.append(f"LOWER({column}) = LOWER('{acronym}')")
        return f"({' OR '.join(alternatives)})"

    sql_str = entity_pattern.sub(harden_entity_predicate, sql_str)

    sql_safe_year_pattern = r"(=\s*|IN\s*\(\s*)['\"](\d{4})['\"]"

    def replace_with_range(match):
        prefix = match.group(1)
        year_str = match.group(2)
        next_yy = (int(year_str[2:]) + 1) % 100
        if "IN" in prefix.upper():
            return f"{prefix}'{year_str}/{next_yy:02d}'"
        return f"= '{year_str}/{next_yy:02d}'"

    return re.sub(sql_safe_year_pattern, replace_with_range, sql_str, flags=re.IGNORECASE)
