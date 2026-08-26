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
    "You are an assistant for question-answering tasks. "
    "For any question that could possibly depend on the contents of the indexed documents, "
    "you MUST call the retrieval tool before answering. "
    "Use the retrieval tool for factual questions, policy/document questions, source lookup, "
    "or whenever there is any ambiguity. "
    "Do not answer from memory when retrieval could help. "
    "If the question is clearly unrelated to the indexed documents, you may answer directly. "
    "When retrieval is used, base your answer only on the retrieved context. "
    "For multi-part questions, address every requested part explicitly. "
    "Use three sentences maximum and keep the answer concise."
)

_MONTHS = {
    "jan": "January", "january": "January", "feb": "February", "february": "February",
    "mar": "March", "march": "March", "apr": "April", "april": "April",
    "may": "May", "jun": "June", "june": "June", "jul": "July", "july": "July",
    "aug": "August", "august": "August", "sep": "September", "sept": "September",
    "september": "September", "oct": "October", "october": "October",
    "nov": "November", "november": "November", "dec": "December", "december": "December",
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


def _expanded_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return metadata plus any JSON object stored in its ``metadata`` field."""
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
    """Convert a source path/URL to a clean filename for model context and citations."""
    if value is None:
        return "Unknown source"
    raw = str(value).strip()
    if not raw:
        return "Unknown source"
    parsed = urlparse(raw)
    path = unquote(parsed.path) if parsed.scheme else unquote(raw.split("?", 1)[0])
    filename = PurePath(path.replace("\\", "/")).name
    return filename or raw


def _format_publication_date(value: Any) -> str | None:
    """Format a trustworthy document date as a human-readable publication month."""
    if isinstance(value, datetime):
        return value.strftime("%B %Y")
    if isinstance(value, date):
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
    """Extract source filename and publication date without using DWH timestamps.

    Explicit document publication/update metadata is preferred. If absent, a month/year
    is parsed from the filename. Generic creation timestamps are deliberately ignored.
    """
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
    """Serialize retrieved chunks with explicit, provenance-labelled metadata."""
    blocks = []
    for index, doc in enumerate(retrieved_docs, start=1):
        filename, publication_date = document_source_metadata(doc)
        blocks.append(
            f"--- DOCUMENT {index} ---\n"
            f"Source Filename: {filename}\n"
            f"Document Publication Date: {publication_date}\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def query_or_respond(state: MessagesState, llm: Any, retrieve_tool: Any):
    """Generate tool call for retrieval, or respond directly."""
    llm_with_tools = llm.bind_tools([retrieve_tool], tool_choice="retrieve_bound")
    messages = [SystemMessage(RETRIEVAL_SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def create_bound_retrieve_tool(vector_store):
    """Create a retrieval tool for CCS Commercial Intelligence documents."""

    @tool(response_format="content_and_artifact")
    def retrieve_bound(query: str):
        """Retrieve relevant CCS Commercial Intelligence knowledge base content."""
        retrieved_docs = vector_store.similarity_search(query, k=8)
        return format_retrieved_documents(retrieved_docs), retrieved_docs

    return retrieve_bound


def _turn_messages_from_latest_human(messages):
    """Return the current turn, including its human question.

    Including the human message is essential: synthesis must still see the question when
    retrieval returns no chunks and SQL is the only available context.
    """
    last_human_index = -1
    for idx, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            last_human_index = idx
    return messages[last_human_index:] if last_human_index >= 0 else messages


def _extract_current_turn_sources(tool_messages):
    source_names = []
    for message in tool_messages:
        artifact = getattr(message, "artifact", None)
        if not artifact:
            continue
        for doc in artifact:
            if isinstance(doc, Document):
                filename, _ = document_source_metadata(doc)
                if filename != "Unknown source":
                    source_names.append(filename)
    return list(dict.fromkeys(source_names))


def generate(state: MessagesState, llm: Any):
    """Generate an answer from current-turn documents and structured SQL context."""
    current_turn_messages = _turn_messages_from_latest_human(state["messages"])
    recent_tool_messages = [
        message for message in current_turn_messages if getattr(message, "type", None) == "tool"
    ]
    source_names = _extract_current_turn_sources(recent_tool_messages)
    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
    safe_context = sanitise_retrieved_content(docs_content)

    sql_context_str = "No structured SQL database data returned for this query."
    for msg in reversed(state["messages"]):
        content = getattr(msg, "content", "")
        if getattr(msg, "type", None) == "system" and "=== RETRIEVED STRUCTURED SQL DATA ===" in content:
            sql_context_str = content
            break

    system_message_content = (
        "You are an assistant that must synthesize structured SQL data and unstructured document context.\n\n"
        "DIRECT ANSWER FIRST (MANDATORY): Open every response with one sentence that directly answers and summarizes "
        "the user's request. For a SQL total, state the exact total in that opening sentence before any details or table.\n"
        "SQL-ONLY ANSWERS (MANDATORY): If SQL data is provided in <sql_database_context>, you MUST use the SQL numbers "
        "to answer the user's question directly in text format, even when <document_context> is empty or no RAG document "
        "chunks were retrieved. Never output 'The user has not asked a question yet', 'Please provide a question', or "
        "any similar placeholder when SQL data is present. The SQL table shown elsewhere in the UI does not replace the "
        "required textual answer.\n"
        "QUESTION DECOMPOSITION (MANDATORY): Identify and explicitly answer every sub-question in the user's request. "
        "Never omit a sub-question because one context source contains more detail than another. If asked whether a "
        "document exists and when it was updated, explicitly confirm whether it exists and state its document date.\n"
        "METRIC PROVENANCE: Structured SQL data in <sql_database_context> is the source of truth for exact company "
        "metrics, totals, spend figures, and counts. Present relevant SQL metrics clearly.\n"
        "DATE PROVENANCE (MANDATORY): 'Document Publication Date' in <document_context> comes from the retrieved "
        "document filename or document metadata. Database record creation/ingestion/ETL dates in SQL are not document "
        "publication or update dates. When asked when a report, fact sheet, or other document was published or last "
        "updated, use the Document Publication Date and never substitute a SQL database timestamp. If the document date "
        "is 'Not provided', say that the update date could not be established from document metadata.\n"
        "DOCUMENT EXISTENCE: A relevant retrieved document block is evidence that the document exists; identify it by "
        "its Source Filename. Do not claim a document exists if no retrieved block supports that claim.\n"
        "CONTRADICTIONS: If document metrics contradict SQL metrics, report SQL metrics; this rule does not apply to "
        "document publication dates, which must come from document context.\n"
        "Use a concise bullet list after the opening sentence when it helps cover multiple parts; otherwise use concise prose.\n\n"
        "<sql_database_context>\n"
        f"{sql_context_str}\n"
        "</sql_database_context>\n\n"
        "<document_context>\n"
        f"{safe_context}\n"
        "</document_context>"
    )
    conversation_messages = [
        message for message in current_turn_messages
        if message.type in ("human", "ai") and not getattr(message, "tool_calls", None)
    ]
    response = llm.invoke([SystemMessage(system_message_content), *conversation_messages])
    response.additional_kwargs["source_names"] = source_names
    return {"messages": [response]}


def stream_turn(graph, user_input: str, thread_id: str = "abc123", stream_mode: str = "values") -> Iterator[Dict[str, Any]]:
    """Stream a single user turn through the graph."""
    config = {"configurable": {"thread_id": thread_id}}
    yield from graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=stream_mode,
        config=config,
    )


def answer_once(graph, user_input: str, thread_id: str = "abc123", state_pre_updated: bool = False):
    """Run one turn and return the final AI answer and retrieved context."""
    last_ai_content = ""
    final_messages = []
    _ = None if state_pre_updated else user_input
    for step in stream_turn(graph, user_input, thread_id):
        messages = step.get("messages", [])
        if messages:
            final_messages = messages
            msg = messages[-1]
            last_ai_content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", last_ai_content)

    def _mtype(message):
        if hasattr(message, "type"):
            return message.type
        return message.get("type") or message.get("role") if isinstance(message, dict) else None

    source_names, source_contents = [], []
    for message in reversed(final_messages):
        if _mtype(message) == "tool":
            artifact = getattr(message, "artifact", None)
            if artifact:
                for doc in artifact:
                    if isinstance(doc, Document):
                        filename, _ = document_source_metadata(doc)
                        if filename != "Unknown source":
                            source_names.append(filename)
                        source_contents.append(doc.page_content)
            break
    return {
        "answer": last_ai_content,
        "source_names": list(dict.fromkeys(source_names)),
        "source_contents": source_contents,
    }


def build_graph(llm, vector_store, checkpointer):
    retrieve_bound = create_bound_retrieve_tool(vector_store)
    query_node = partial(query_or_respond, llm=llm, retrieve_tool=retrieve_bound)
    generate_node = partial(generate, llm=llm)
    tool_node = ToolNode([retrieve_bound])

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile(checkpointer=checkpointer)


def format_sources(source_names, CI_docs_URLs):
    """Format source documents into links and create expander content."""
    if not source_names:
        return None
    unique_sources = list(dict.fromkeys(source_names))
    source_links = []
    for source_name in unique_sources:
        doc_row = CI_docs_URLs[CI_docs_URLs["File Name"] == source_name]
        if doc_row.shape[0] > 0:
            doc_URL = doc_row.iloc[0, :]["File URL"]
            source_links.append(f"[{source_name}]({doc_URL})")
        else:
            source_links.append(source_name)
    sources_content = f"**Most Relevant Document:**\n- {source_links[0]}"
    if len(source_links) > 1:
        sources_content += "\n\n**Other Related Documents:**\n" + "\n".join(f"- {link}" for link in source_links[1:])
    return sources_content
