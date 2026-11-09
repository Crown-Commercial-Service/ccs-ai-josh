from __future__ import annotations

import json
import re
from datetime import date, datetime
from functools import partial
from pathlib import PurePath
from typing import Any, Dict, Iterator, Mapping
from urllib.parse import unquote, urlparse

from langchain_core.documents.base import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.sanitise import sanitise_retrieved_content

RETRIEVAL_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. For any question that could possibly depend on the contents "
    "of the indexed documents, you MUST call the retrieval tool before answering. Use the retrieval tool for factual "
    "questions, policy/document questions, source lookup, or whenever there is any ambiguity. Do not answer from memory "
    "when retrieval could help. If the question is clearly unrelated to the indexed documents, you may answer directly. "
    "When retrieval is used, base your answer only on the retrieved context. For multi-part questions, address every "
    "requested part explicitly. Use three sentences maximum and keep the answer concise."
)

_MONTHS = {
    "jan": "January", "january": "January", "feb": "February", "february": "February",
    "mar": "March", "march": "March", "apr": "April", "april": "April", "may": "May",
    "jun": "June", "june": "June", "jul": "July", "july": "July", "aug": "August",
    "august": "August", "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October", "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_YEAR_RE = re.compile(rf"(?i)(?<![A-Za-z])({_MONTH_PATTERN})[\s_.-]+((?:19|20)\d{{2}})(?!\d)")
_YEAR_MONTH_RE = re.compile(rf"(?i)(?<!\d)((?:19|20)\d{{2}})[\s_.-]+({_MONTH_PATTERN})(?![A-Za-z])")
_ISO_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_/](0?[1-9]|1[0-2])[-_/](?:0?[1-9]|[12]\d|3[01])(?!\d)")
_PUBLICATION_DATE_KEYS = (
    "document_publication_date", "publication_date", "published_date", "publish_date",
    "document_date", "last_updated", "modified_date", "last_modified",
)
_FILENAME_KEYS = ("title", "file_name", "filename", "name", "source", "path", "url")
_SQL_CONTEXT_MARKER = "=== RETRIEVED STRUCTURED SQL DATA ==="
_EMPTY_SQL_MARKERS = (
    "no structured sql database data returned", "there is no structured sql data available",
    "query executed successfully but returned 0 rows", "no structured database matching fields",
    "returned no rows", "empty dataframe",
)


def _expanded_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    expanded = dict(metadata or {})
    nested = expanded.get("metadata")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except (TypeError, json.JSONDecodeError):
            nested = None
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            expanded.setdefault(str(key), value)
    return expanded


def _display_filename(value: Any) -> str:
    if value is None:
        return "Unknown source"
    raw = str(value).strip()
    if not raw:
        return "Unknown source"
    parsed = urlparse(raw)
    path = unquote(parsed.path) if parsed.scheme else unquote(raw.split("?", 1)[0])
    return PurePath(path.replace("\\", "/")).name or raw


def _format_publication_date(value: Any) -> str | None:
    if isinstance(value, (datetime, date)):
        return value.strftime("%B %Y")
    if value is None:
        return None
    text = str(value).strip()
    month_year = _MONTH_YEAR_RE.search(text)
    if month_year:
        return f"{_MONTHS[month_year.group(1).casefold()]} {month_year.group(2)}"
    year_month = _YEAR_MONTH_RE.search(text)
    if year_month:
        return f"{_MONTHS[year_month.group(2).casefold()]} {year_month.group(1)}"
    iso_date = _ISO_DATE_RE.search(text)
    if iso_date:
        return f"{datetime(2000, int(iso_date.group(2)), 1).strftime('%B')} {iso_date.group(1)}"
    return text or None


def document_source_metadata(doc: Document) -> tuple[str, str]:
    metadata = _expanded_metadata(doc.metadata)
    source_value = next((metadata.get(key) for key in _FILENAME_KEYS if metadata.get(key)), None)
    filename = _display_filename(source_value)
    publication_date = None
    for key in _PUBLICATION_DATE_KEYS:
        if metadata.get(key):
            publication_date = _format_publication_date(metadata[key])
            if publication_date:
                break
    if not publication_date:
        publication_date = _format_publication_date(filename)
    return filename, publication_date or "Not provided"


def format_retrieved_documents(retrieved_docs: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(retrieved_docs, start=1):
        filename, publication_date = document_source_metadata(doc)
        blocks.append(
            f"--- DOCUMENT {index} ---\nSource Filename: {filename}\n"
            f"Document Publication Date: {publication_date}\nContent:\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def query_or_respond(state: MessagesState, llm: Any, retrieve_tool: Any):
    llm_with_tools = llm.bind_tools([retrieve_tool], tool_choice="retrieve_bound")
    response = llm_with_tools.invoke([SystemMessage(RETRIEVAL_SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}


def create_bound_retrieve_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve_bound(query: str):
        """Retrieve relevant CCS Commercial Intelligence knowledge base content."""
        retrieved_docs = vector_store.similarity_search(query, k=8)
        return format_retrieved_documents(retrieved_docs), retrieved_docs
    return retrieve_bound


def _message_type(message: Any) -> str | None:
    """Return the LangChain message type for object and dictionary messages."""
    if hasattr(message, "type"):
        return message.type
    if isinstance(message, Mapping):
        return message.get("type") or message.get("role")
    return None


def _message_content(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("content", "")
    return getattr(message, "content", str(message))


def _turn_messages_from_latest_human(messages):
    """Return the latest turn; retained for retrieval artifact/source scoping only."""
    last_human_index = -1
    for idx, message in enumerate(messages):
        if _message_type(message) in ("human", "user"):
            last_human_index = idx
    return messages[last_human_index:] if last_human_index >= 0 else messages


def _conversation_messages(messages):
    """Keep all user/assistant dialogue so follow-up temporal references retain context.

    Tool-call assistant messages and tool payloads are deliberately excluded: their
    useful data is supplied separately in the SQL/document context blocks.
    """
    dialogue = []
    for message in messages:
        message_type = _message_type(message)
        if message_type not in ("human", "user", "ai", "assistant"):
            continue
        tool_calls = (
            message.get("tool_calls") if isinstance(message, Mapping)
            else getattr(message, "tool_calls", None)
        )
        if tool_calls:
            continue
        dialogue.append(message)
    return dialogue


def _extract_current_turn_sources(tool_messages):
    source_names = []
    for message in tool_messages:
        artifact = getattr(message, "artifact", None)
        if not artifact and isinstance(message, Mapping):
            artifact = message.get("artifact")
        if not artifact:
            continue
        for doc in artifact:
            if isinstance(doc, Document):
                filename, _ = document_source_metadata(doc)
                if filename != "Unknown source":
                    source_names.append(filename)
    return list(dict.fromkeys(source_names))


def _has_valid_sql_results(sql_context_str: str) -> bool:
    """Accept any marked SQL payload except a known-empty or null-only result."""
    if not isinstance(sql_context_str, str) or _SQL_CONTEXT_MARKER not in sql_context_str:
        return False
    result_text = sql_context_str.split(_SQL_CONTEXT_MARKER, 1)[1].strip()
    if not result_text:
        return False
    folded = result_text.casefold()
    if any(marker in folded for marker in _EMPTY_SQL_MARKERS):
        return False
    compact = re.sub(r"[\s|,:;=\-]+", " ", folded).strip()
    if re.fullmatch(r"(?:total\w*|sum\w*|average\w*|avg\w*)\s+(?:none|null|nan|n/a)", compact):
        return False
    return True


def _build_synthesis_prompt(sql_context_str: str, safe_context: str) -> str:
    """Build a synthesis prompt with SQL rules only when a valid result is present."""
    sql_rules = ""
    if _has_valid_sql_results(sql_context_str):
        sql_rules = (
            "HYBRID ANSWER SYNTHESIS INSTRUCTIONS & STRICT DATA HIERARCHY:\n"
            "1. SQL DATA ABSOLUTE PRIORITY: Structured SQL data in <sql_database_context> is the absolute source of truth "
            "and the SINGLE SOURCE OF TRUTH for numerical spend figures, customer breakdowns, and supplier totals. Answer "
            "with the SQL result first. NEVER prioritize PDF market reports or unstructured figures over SQL results.\n"
            "2. CALENDAR AND FINANCIAL-YEAR ANCHOR (MANDATORY): The current calendar year is 2026 and the current financial "
            "year is 2026/27. Therefore 'this year' means FY 2026/27, while 'last year' and 'previous financial year' mean "
            "FY 2025/26. A returned SQL row for 2025/26 is the correct direct answer to 'What about last year?'. NEVER "
            "reinterpret 'last year' as 2024/25.\n"
            "3. FOLLOW-UP RESOLUTION: Use the complete conversation history to resolve pronouns, omitted metrics, entities, "
            "and relative dates. For example, after asking for 'total evidenced spend for this year', 'what about last year?' "
            "means total evidenced spend for FY 2025/26.\n"
            "4. SQL QUERY PROVENANCE (MANDATORY): Inspect the Executed SQL Query section and its WHERE clause. Apply every "
            "supplier, customer, date, and other filter to every result row, even if a filtered field is omitted from SELECT.\n"
            "5. NO DENIAL HALLUCINATIONS: If a valid SQL result answers the request, do not say it is unavailable, not directly "
            "provided, outside the covered period, or that documents must substitute for it. In particular, do not claim SQL "
            "only covers 2025/26 when the user asks for last year: 2025/26 is exactly last year under the calendar anchor.\n"
            "6. OPENING ANCHOR (MANDATORY): Sentence 1 MUST directly state the requested SQL answer, entity, metric, and FY. "
            "For an organizational total, treat it as the official total for that organization.\n"
            "7. DOCUMENT SUB-TOTALS (MANDATORY): Only after the direct SQL answer, introduce retrieved material with wording "
            "such as 'I also found information in the documents...' and present it as secondary supporting sub-totals or "
            "category breakdowns. Never let documents override or replace the SQL answer.\n"
            "8. NO CONTRADICTION CLAIMS: Explain genuine scope differences without casting doubt on the SQL total.\n"
            "9. SQL-ONLY ANSWERS: Use SQL numbers directly even when no documents were retrieved.\n"
            "10. DATE PROVENANCE: Database creation, ingestion, and ETL dates are not document publication dates.\n\n"
            "### CRITICAL INSTRUCTION: Data Comparison & Source Reconciliation Rules\n"
            "1. NEVER COPY SQL VALUES INTO FACTSHEET COLUMNS: A Factsheet or Document column MUST ONLY contain numbers "
            "explicitly written in retrieved document text.\n"
            "2. MANDATORY NUMERIC & SCOPE COMPARISON: If no document total is explicitly reported, state 'Not Reported as "
            "Total' or 'Category Spend Only'; do not infer, calculate, or borrow a total from SQL.\n"
            "3. STRICT ALIGNMENT GUARDRAIL: Category-specific factsheet spend DOES NOT match total organizational SQL spend.\n"
            "4. PARTIAL DOCUMENT COVERAGE: Never manufacture a factsheet total from SQL data.\n\n"
        )

    return (
        "You are an assistant that synthesizes context relevant to the user's question. Write a fluent, natural, concise "
        "answer grounded only in the supplied context.\n\n"
        "DIRECT ANSWER FIRST: Open with one sentence that directly answers the user's request before supporting details.\n"
        "QUESTION DECOMPOSITION (MANDATORY): Explicitly answer every sub-question.\n"
        "DOCUMENT PROVENANCE: Identify supporting documents by Source Filename and use Document Publication Date for "
        "publication/update questions. Do not claim a document exists without a supporting retrieved block.\n"
        "MARKDOWN TABLE FORMAT: Include a blank line before and after every Markdown table.\n\n"
        f"{sql_rules}<sql_database_context>\n{sql_context_str}\n</sql_database_context>\n\n"
        f"<document_context>\n{safe_context}\n</document_context>"
    )


def generate(state: MessagesState, llm: Any):
    all_messages = state["messages"]

    # Scope retrieved artifacts to this turn, but do not scope dialogue to this turn:
    # earlier user/assistant messages are essential for resolving follow-up questions.
    current_turn_messages = _turn_messages_from_latest_human(all_messages)
    recent_tool_messages = [m for m in current_turn_messages if _message_type(m) == "tool"]
    source_names = _extract_current_turn_sources(recent_tool_messages)
    docs_content = "\n\n".join(str(_message_content(msg)) for msg in recent_tool_messages)
    safe_context = sanitise_retrieved_content(docs_content)

    sql_context_str = "No structured SQL database data returned for this query."
    for msg in reversed(all_messages):
        content = _message_content(msg)
        if _message_type(msg) == "system" and _SQL_CONTEXT_MARKER in str(content):
            sql_context_str = str(content)
            break

    prompt = _build_synthesis_prompt(sql_context_str, safe_context)
    response = llm.invoke([SystemMessage(prompt), *_conversation_messages(all_messages)])
    response.additional_kwargs["source_names"] = source_names
    return {"messages": [response]}


def stream_turn(graph, user_input: str, thread_id: str = "abc123", stream_mode: str = "values") -> Iterator[Dict[str, Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    yield from graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode=stream_mode, config=config)


def answer_once(graph, user_input: str, thread_id: str = "abc123", state_pre_updated: bool = False):
    last_ai_content, final_messages = "", []
    _ = None if state_pre_updated else user_input
    for step in stream_turn(graph, user_input, thread_id):
        messages = step.get("messages", [])
        if messages:
            final_messages = messages
            msg = messages[-1]
            last_ai_content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", last_ai_content)

    source_names, source_contents = [], []
    for message in reversed(final_messages):
        if _message_type(message) == "tool":
            artifact = getattr(message, "artifact", None)
            if artifact:
                for doc in artifact:
                    if isinstance(doc, Document):
                        filename, _ = document_source_metadata(doc)
                        if filename != "Unknown source":
                            source_names.append(filename)
                        source_contents.append(doc.page_content)
            break
    return {"answer": last_ai_content, "source_names": list(dict.fromkeys(source_names)), "source_contents": source_contents}


def build_graph(llm, vector_store, checkpointer):
    retrieve_bound = create_bound_retrieve_tool(vector_store)
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", partial(query_or_respond, llm=llm, retrieve_tool=retrieve_bound))
    graph_builder.add_node("tools", ToolNode([retrieve_bound]))
    graph_builder.add_node("generate", partial(generate, llm=llm))
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile(checkpointer=checkpointer)


def format_sources(source_names, CI_docs_URLs):
    if not source_names:
        return None
    unique_sources = list(dict.fromkeys(source_names))
    source_links = []
    for source_name in unique_sources:
        doc_row = CI_docs_URLs[CI_docs_URLs["File Name"] == source_name]
        if doc_row.shape[0] > 0:
            source_links.append(f"[{source_name}]({doc_row.iloc[0, :]['File URL']})")
        else:
            source_links.append(source_name)
    content = f"**Most Relevant Document:**\n- {source_links[0]}"
    if len(source_links) > 1:
        content += "\n\n**Other Related Documents:**\n" + "\n".join(f"- {link}" for link in source_links[1:])
    return content
