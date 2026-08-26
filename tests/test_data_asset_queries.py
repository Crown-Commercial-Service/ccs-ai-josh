import pytest

from src.data_asset_queries import (
    QueryRoute,
    _normalise_for_routing,
    build_user_display_payload,
    classify_query_route,
    is_business_data_request,
    is_data_asset_query,
    is_fuzzy_capability_match,
)


SME_SPEND_QUERY = (
    "We have a target to direct 33% of our procurement spend to Small and Medium "
    "Enterprises (SMEs). Based on our transaction data, what is our total spend "
    "with SME-flagged suppliers compared to non-SMEs?"
)


@pytest.mark.parametrize(
    "query",
    [
        "What data sources do you currently have access to?",
        "Can you list the data sources you have access to please?",
        "What kind of questions can I ask:",
        "Could you list your available data sources?",
        "Please show available databases",
        "wat questons can I ask",
        "What data sources do you have access to?",
        "What data sources are currently available?",
        "What questions can I ask you?",
        "Give me some sample questions.",
        "What can this agent do?",
        "What are your capabilities?",
        "What datasets can this model query?",
        "What tables are available to you?",
        "Which documents can you access?",
        "Help me get started with this assistant.",
        "How should I use this agent?",
        "What data are you using to give answers with?",
        "what do you know",
        "what can I ask",
    ],
)
def test_explicit_capability_queries_route_to_catalog(query):
    assert classify_query_route(query) is QueryRoute.DATA_ASSET_CATALOG


@pytest.mark.parametrize(
    "query",
    [
        "What was the total evidenced spend for this year?",
        "can you give me top 5 companies with the highest spend",
        "Show me all unique market sectors from the 2025/26 financial year.",
        (
            "I'm pulling together the account review for NHS Herts Valleys. What is "
            "the total amount we actually spent with them, and how much did we "
            "generate in commission fees for this current financial year?"
        ),
        (
            "What is the total spend and the average transaction value for all "
            "'Direct' spend, and how many transactions are there in total?"
        ),
        (
            "Show me our top 5 suppliers by total spend in the 'Direct' category, "
            "and what is the average transaction size for each?"
        ),
        (
            "Identify all 'Direct' spend transactions that are over £50,000 where "
            "the supplier name doesn't contain 'Approved' or 'Framework'. What is "
            "the total unapproved spend and who are the worst offenders"
        ),
        (
            "Which supplier has received the highest number of direct award "
            "transactions, what is their total spend, and what percentage of our "
            "overall direct spend does that represent?"
        ),
        SME_SPEND_QUERY,
        "Compare supplier spend between 2024 and 2025.",
        "How many transactions did each department process?",
        "Calculate average contract value by category.",
        "What data can you access, and compare £250,000 against £300,000?",
        "Who is the CEO of BAE Systems?",
    ],
)
def test_business_and_factual_queries_route_to_sql_and_rag(query):
    assert classify_query_route(query) is QueryRoute.SQL_AND_RAG


@pytest.mark.parametrize(
    "query",
    [
        "What data do you have access to, and what is our total SME spend?",
        "What can you do? Also compare spend for SMEs versus non-SMEs.",
        "Give me a sample question, then calculate average invoice value by supplier.",
        "Help me use this assistant to show the top 5 suppliers by spend.",
        "Which tables can you query to find how many transactions occurred this year?",
        "Describe your capabilities and report sales by month.",
        "What do you know about our target of 33% and total SME transaction spend?",
        "What can you do and tell me who leads Acme Holdings in Scotland?",
    ],
)
def test_business_or_mixed_requests_route_to_sql_and_rag(query):
    assert classify_query_route(query) is QueryRoute.SQL_AND_RAG


@pytest.mark.parametrize(
    "query",
    [
        "What data do you have access to, and what is our total SME spend?",
        "What can you do? Also compare spend for SMEs versus non-SMEs.",
        "Give me a sample question, then calculate average invoice value by supplier.",
        "Help me use this assistant to show the top 5 suppliers by spend.",
        "Which tables can you query to find how many transactions occurred this year?",
        "Describe your capabilities and report sales by month.",
        "What do you know about our target of 33% and total SME transaction spend?",
    ],
)
def test_recognised_business_request_has_absolute_priority(query):
    assert is_business_data_request(query) is True
    assert classify_query_route(query) is QueryRoute.SQL_AND_RAG


@pytest.mark.parametrize(
    "query",
    [
        "wat questons can I ask",
        "show available databaases",
        "wht data do you hav access to",
    ],
)
def test_typo_capability_queries_use_fuzzy_matching(query):
    assert is_fuzzy_capability_match(query)
    assert classify_query_route(query) is QueryRoute.DATA_ASSET_CATALOG


@pytest.mark.parametrize(
    "raw_input, expected_normalised",
    [
        ("what kind of questions can I ask:", "what kind of questions can i ask"),
        ("  ...What DATA do you have access to???  ", "what data do you have access to"),
        ("‘what do you know?’", "what do you know"),
        (
            "Can you list the data sources you have access to please?",
            "list the data sources you have access to",
        ),
        ("Could you list your available data sources?", "list your available data sources"),
        ("Please show available databases", "show available databases"),
        ("Tell me what data is available, please!", "what data is available"),
        ("I want to know what questions can I ask:", "what questions can i ask"),
    ],
)
def test_normalisation_handles_punctuation_and_polite_wrappers(
    raw_input, expected_normalised
):
    assert _normalise_for_routing(raw_input) == expected_normalised


def test_routing_and_payload_preserve_exact_original_display_string():
    raw_input = "  Can you list the data sources you have access to please?:  "
    assert classify_query_route(raw_input) is QueryRoute.DATA_ASSET_CATALOG
    payload = build_user_display_payload(raw_input)
    assert payload["content"] == raw_input
    assert payload["raw_content"] == raw_input


def test_sme_query_is_detected_as_business_request():
    assert is_business_data_request(SME_SPEND_QUERY) is True
    assert classify_query_route(SME_SPEND_QUERY) is QueryRoute.SQL_AND_RAG


class FailIfInvokedLLM:
    def __init__(self):
        self.invoke_count = 0

    def invoke(self, *_args, **_kwargs):
        self.invoke_count += 1
        raise AssertionError("Routing must not invoke an LLM")


def test_sme_routing_does_not_invoke_secondary_llm():
    llm = FailIfInvokedLLM()
    assert is_data_asset_query(SME_SPEND_QUERY, llm=llm) is False
    assert llm.invoke_count == 0


def test_capability_routing_does_not_invoke_secondary_llm():
    llm = FailIfInvokedLLM()
    assert is_data_asset_query("What questions can I ask you?", llm=llm) is True
    assert llm.invoke_count == 0


@pytest.mark.parametrize("query", ["", "   ", None, 42])
def test_invalid_or_empty_input_safely_defaults_to_sql_and_rag(query):
    assert classify_query_route(query) is QueryRoute.SQL_AND_RAG
