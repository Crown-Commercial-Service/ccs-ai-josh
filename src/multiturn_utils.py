from typing import Any, Iterator, Dict
from functools import partial
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.documents.base import Document
from langgraph.graph import MessagesState, StateGraph, END
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
    "Use three sentences maximum and keep the answer concise."
)


def query_or_respond(state: MessagesState, llm: Any, retrieve_tool: Any):
    """Generate tool call for retrieval, or respond directly."""
    llm_with_tools = llm.bind_tools([retrieve_tool], tool_choice="retrieve_bound")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def create_bound_retrieve_tool(vector_store):
    """Create a retrieval tool for CCS Commercial Intelligence documents, including policy guidance, commercial strategy, contract management, operational procedures, document summaries, and supporting reference materials."""

    @tool(response_format="content_and_artifact")
    def retrieve_bound(query: str):
        """Retrieve relevant CCS Commercial Intelligence knowledge base content: policy guidance, commercial strategy, contract management, operational procedures, document summaries, and supporting reference material."""
        retrieved_docs = vector_store.similarity_search(query, k=8)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_bound


def _turn_messages_after_latest_human(messages):
    last_human_index = -1
    for idx, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            last_human_index = idx
    return messages[last_human_index + 1 :] if last_human_index >= 0 else messages


def _extract_current_turn_sources(tool_messages):
    source_names = []
    for message in tool_messages:
        artifact = getattr(message, "artifact", None)
        if not artifact:
            continue
        for doc in artifact:
            if hasattr(doc, "metadata"):
                title = doc.metadata.get("title") or doc.metadata.get("source")
                if title:
                    source_names.append(title)
    return list(dict.fromkeys(source_names))


def generate(state: MessagesState, llm: Any):
    """Generate answer using both SQL system context and retrieved document context from the current turn only."""
    current_turn_messages = _turn_messages_after_latest_human(state["messages"])

    recent_tool_messages = [
        message for message in current_turn_messages if getattr(message, "type", None) == "tool"
    ]
    source_names = _extract_current_turn_sources(recent_tool_messages)
    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
    safe_context = sanitise_retrieved_content(docs_content)

    sql_context_str = "No structured SQL database data returned for this query."
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "system" and "=== RETRIEVED STRUCTURED SQL DATA ===" in getattr(msg, "content", ""):
            sql_context_str = msg.content
            break

    system_message_content = (
        "You are an assistant for question-answering tasks that must synthesize BOTH structured SQL data "
        "and unstructured document context.\n\n"
        "GROUND TRUTH: Structured SQL data in <sql_database_context> is the absolute source of truth for all exact metrics, totals, spend figures, and counts.\n"
        "COMPLEMENTARY RAG: Use information from <document_context> to provide narrative context, policy guidance, or qualitative details that support or explain the SQL results.\n"
        "CONTRADICTION HANDLING: If any retrieved document figures or totals directly contradict the SQL data, ignore the contradicting numbers in the documents and strictly report the SQL figures.\n"
        "When quantitative metrics are present, lead with the SQL answer first, then follow with non-contradictory document context.\n"
        "Blend both context streams into one cohesive, concise answer.\n"
        "Use three sentences maximum unless a concise bullet list is clearly more useful.\n\n"
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
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    response.additional_kwargs["source_names"] = source_names
    return {"messages": [response]}


def stream_turn(
    graph,
    user_input: str,
    thread_id: str = "abc123",
    stream_mode: str = "values",
) -> Iterator[Dict[str, Any]]:
    """
    Stream a single user turn through the graph, yielding step values.

    Yields the values dicts produced by graph.stream(...).
    """
    config = {"configurable": {"thread_id": thread_id}}
    yield from graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=stream_mode,
        config=config,
    )


def answer_once(
    graph,
    user_input: str,
    thread_id: str = "abc123",
    state_pre_updated: bool = False
):
    """
    Run one turn and return both the final AI answer and the retrieved context.

    Returns:
        dict with keys:
          - "answer": str
          - "context": str (concatenated content of the most recent tool messages)
    """
    last_ai_content = ""
    final_messages = []
    _ = None if state_pre_updated else user_input
    for step in stream_turn(graph, user_input, thread_id):
        messages = step.get("messages", [])
        if messages:
            final_messages = messages
            msg = messages[-1]
            if hasattr(msg, "content"):
                last_ai_content = msg.content
            elif isinstance(msg, dict):
                last_ai_content = msg.get("content", last_ai_content)

    def _mtype(m):
        if hasattr(m, "type"):
            return m.type
        if isinstance(m, dict):
            return m.get("type") or m.get("role")
        return None

    source_names = []
    source_contents = []
    i = len(final_messages) - 1
    last_tool_message_found = False
    while i >= 0:
        message = final_messages[i]
        if last_tool_message_found:
            break
        elif _mtype(message) == "tool":
            last_tool_message_found = True
            artifact = getattr(message, "artifact", None)
            if not artifact:
                source_names = []
                source_contents = []
            else:
                for doc in artifact:
                    if isinstance(doc, Document):
                        title = doc.metadata.get("title") or doc.metadata.get("source")
                        if title:
                            source_names.append(title)
                        source_contents.append(doc.page_content)
        i -= 1
    response = {
        "answer": last_ai_content,
        "source_names": list(dict.fromkeys(source_names)),
        "source_contents": source_contents,
    }
    return response


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
    graph_builder.add_conditional_edges(
        "query_or_respond", tools_condition, {END: END, "tools": "tools"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph


def format_sources(source_names, CI_docs_URLs):
    """Format source documents into links and create expander content"""
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
        sources_content += "\n\n**Other Related Documents:**\n" + "\n".join(
            f"- {link}" for link in source_links[1:]
        )

    return sources_content
