import json
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import re
import os


SQL_entities_folder = Path(__file__).parents[1] / "SQL_entities"

def format_financial_years(text: str) -> str:
    """Forces all financial year variations (including YYYY/YYYY) into the strict 'YYYY/YY' structure.

    Examples:
      - "spend in 2025/2026" -> "spend in 2025/26"
      - "spend in 25/26"     -> "spend in 2025/26"
      - "spend in 2025"      -> "spend in 2025/26"
    """

    # --- 1. Catch 4-digit to 4-digit ranges (e.g., '2025/2026' or '2024/2025') ---
    # Looks for 4 digits, a slash, and another 4 digits
    long_range_pattern = r'\b(\d{4})\s*/\s*(\d{2})(\d{2})\b'

    def collapse_long_range(match):
        start_yyyy = match.group(1)  # e.g., "2025"
        end_yy = match.group(3)  # e.g., "26" (ignoring the "20" century part in group 2)
        return f"{start_yyyy}/{end_yy}"

    text = re.sub(long_range_pattern, collapse_long_range, text)

    # --- 2. Catch 2-digit shorthand ranges (e.g., '25/26' or '24/25') ---
    # Looks for two digits, a slash, and two digits
    short_range_pattern = r'\b(\d{2})\s*/\s*(\d{2})\b(?!\s*/)'

    def expand_short_range(match):
        start_yy = match.group(1)
        end_yy = match.group(2)
        return f"20{start_yy}/{end_yy}"

    text = re.sub(short_range_pattern, expand_short_range, text)

    # --- 3. Catch standalone 4-digit years (e.g., '2025' or '2026') ---
    # Looks for a 4-digit year that isn't already followed by a slash /
    four_digit_pattern = r'\b(19|20)(\d{2})\b(?!\s*/)'

    def expand_standalone_year(match):
        century = match.group(1)  # e.g., "20"
        short_yy = int(match.group(2))  # e.g., 25
        next_yy = (short_yy + 1) % 100  # e.g., 26
        return f"{century}{short_yy:02d}/{next_yy:02d}"

    text = re.sub(four_digit_pattern, expand_standalone_year, text)

    return text


def spell_correct_user_query(user_input: str, llm: Any, json_name: str = "dummy_entities.json") -> str:
    """First fixes financial year mathematics, then maps messy shorthand and

    supplier misspellings to ground-truth catalog entries.
    """
    # STEP A: Handle Date Conversion Locally (Zero Tokens Cost)
    # This turns "BAE spend from 2026" into "BAE spend from 2026/27"
    processed_input = format_financial_years(user_input)

    #  Safely load the pruned dictionary catalog

    json_path = os.path.join(SQL_entities_folder, json_name)
    try:
        with open(json_path, "r") as f:
            valid_vocabulary = json.load(f)
    except Exception as err:
        print(f"⚠️ Dictionary read failure: {err}. Returning date-patched input.")
        return processed_input

    # 🌟 STEP B: LLM Supplier & Framework Translator Block
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
        "5. CRITICAL: Output ONLY valid JSON syntax. Do not write markdown, ```json fences, or chat commentary."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=processed_input)
        ])

        clean_json_string = (
            response.content.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        corrections = json.loads(clean_json_string)

        # Patch the string safely in Python
        corrected_query = processed_input
        for typo_chunk, official_row_value in corrections.items():
            insensitive_regex = re.compile(re.escape(typo_chunk), re.IGNORECASE)
            corrected_query = insensitive_regex.sub(official_row_value, corrected_query)
        print("this was used ")
        return corrected_query

    except Exception as err:
        print(f"⚠️ Grammar translation fallback active: {err}")
        return processed_input

# =========================================================================
# 🌟 THE SQL POST-PROCESSOR INTERCEPTOR
# =========================================================================
def harden_vanna_sql(sql_str: str, json_name: str = "dummy_entities.json") -> str:
    """Processes Vanna's SQL output directly to guarantee 100% database compatibility.
    Dynamically loads the catalog to fix casing/naming issues and patches broken relative dates.
    """
    if not sql_str:
        print("nothing happening")
        return sql_str

    # 🌟 1. DYNAMIC CATALOG LOADING: Pull vocabulary directly inside the function
    try:
        print("This is happening")
        json_path = SQL_entities_folder / json_name
        with open(json_path, "r") as f:
            catalog_data = json.load(f)
            # Combine suppliers and frameworks into one dynamic search-and-replace catalog list
            validation_catalog = catalog_data.get("suppliers", []) + catalog_data.get("frameworks", [])
            print(validation_catalog)
    except Exception as err:
        print(f"⚠️ SQL post-processor failed to load entity catalog: {err}")
        validation_catalog = []

    # 🌟 2. Match partial query strings to their full, official catalog counterparts
    # Targets specific patterns like: SupplierName = 'Microsoft' or Framework = 'RM6100'
    entity_pattern = r"(\b(?:SupplierName|Framework)\s*=\s*['\"])([^'\"]+)(['\"])"

    def replace_with_catalog_entry(match):
        prefix = match.group(1)  # Keeps: SupplierName = '
        current_val = match.group(2)  # Extracts: Microsoft
        suffix = match.group(3)  # Keeps: '

        # Check if the short value from the SQL query is a substring of an official catalog entry
        for entry in validation_catalog:
            if current_val.lower() in entry.lower():
                return f"{prefix}{entry}{suffix}"

        # If no match is found in the catalog, leave the original string intact
        return match.group(0)

    sql_str = re.sub(entity_pattern, replace_with_catalog_entry, sql_str, flags=re.IGNORECASE)

    # 3. Force naked 4-digit years (e.g., = '2025') into strict YYYY/YY database matches
    # This safely supports standard assignments (= '2025') and grouping arrays (IN ('2025'))
    sql_safe_year_pattern = r"(=\s*|IN\s*\(\s*)['\"](\d{4})['\"]"

    def replace_with_range(match):
        prefix = match.group(1)  # Keeps the syntax context like "=" or "IN ("
        year_str = match.group(2)  # Extracts "2025"
        short_yy = int(year_str[2:])
        next_yy = (short_yy + 1) % 100

        if "IN" in prefix:
            return f"{prefix}'{year_str}/{next_yy:02d}'"
        return f"= '{year_str}/{next_yy:02d}'"

    sql_str = re.sub(sql_safe_year_pattern, replace_with_range, sql_str)

    return sql_str
# from langchain_openai import  AzureChatOpenAI
# from dotenv import load_dotenv
# load_dotenv()
#
# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     azure_deployment=os.getenv("DEPLOYMENT_NAME"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     temperature=0.0,
# )
#
# a = spell_correct_user_query(user_input="give BAEs total spend from 2026", llm=llm)
# print(a)