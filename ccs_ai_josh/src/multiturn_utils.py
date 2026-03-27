from typing import Any, Iterator, Dict
from functools import partial, wraps, WRAPPER_ASSIGNMENTS
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.documents.base import Document
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.prompt_sanitiser import sanitise_input

def query_or_respond(state: MessagesState, llm: Any, retrieve_tool: Any):
    "Generate tool call for retrieval, or respond directly"
    llm_with_tools = llm.bind_tools([retrieve_tool])
    response = llm_with_tools.invoke(state["messages"])
    # the response will contain the most recent response and the previous responses
    return {"messages": [response]}

def create_bound_retrieve_tool(vector_store):
    """Create a properly decorated retrieve tool bound to a specific vector store"""
    @tool(response_format="content_and_artifact")
    def retrieve_bound(query: str):
        """Retrieve information related to a query"""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    return retrieve_bound

def generate(state: MessagesState, llm: Any):
    """Generate answer"""
    # capture the most recent tool messages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    # put the recent tool messages into their original order
    tool_messages = recent_tool_messages[::-1]
    # format chat exchange and results of tool calls into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep the"
        "answer concise."
        "\n\n"
        "--- BEGIN RETRIEVED DOCUMENTS ---\n"
        f"{docs_content}"
        "\n--- END RETRIEVED DOCUMENTS ---"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

def stream_turn(
    graph,
    user_input: str,
    thread_id: str = "abc123",
    stream_mode: str = "values",
) -> Iterator[Dict[str, Any]]:
    """
    Stream a single user turn through the graph, yielding step values.

    The raw user input is passed through :func:`~src.prompt_sanitiser.sanitise_input`
    before being forwarded to the conversation graph.  A :class:`ValueError` is
    raised (and should be caught by the caller) if the input contains a detected
    prompt-injection or SQL-injection attempt.

    Yields the values dicts produced by graph.stream(...).
    """
    sanitised = sanitise_input(user_input)
    config = {"configurable": {"thread_id": thread_id}}
    yield from graph.stream(
        {"messages": [{"role": "user", "content": sanitised}]},
        stream_mode=stream_mode,
        config=config,
    )

def answer_once(
    graph,
    user_input: str,
    thread_id: str = "abc123",
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

    for step in stream_turn(graph, user_input, thread_id):
        # in case there have been no messages yet, use `get` to pass a default value (empty list)
        messages = step.get("messages", [])
        if messages:
            final_messages = messages
            msg = messages[-1]
            # extract content, handling cases where messages are either dicts or object attributes
            if hasattr(msg, "content"):
                last_ai_content = msg.content
            elif isinstance(msg, dict):
                last_ai_content = msg.get("content", last_ai_content)

    # Helpers to read message fields across LangChain objects/dicts
    def _mtype(m):
        if hasattr(m, "type"):
            return m.type
        if isinstance(m, dict):
            # some serialisations use "type", others "role"
            return m.get("type") or m.get("role")
        return None

    # here we check if the model used the retrieval tool, and if so we collect its output
    source_names = []
    source_contents = []
    # Set start position to most recent message
    i = len(final_messages) - 1
    # run through messages until the most recent tool message is found, and grab its result
    last_tool_message_found = False
    while i >= 0:
        message = final_messages[i]
        if last_tool_message_found:
            # We've reached the most recent non-tool message after the last tool message, so exit
            break
        elif _mtype(message) == "tool":
            # we've found the most recent tool message, so now we need to extract the relevant info
            last_tool_message_found = True
            # check if the tool has returned an artifact
            artifact = getattr(message, "artifact", None)
            if not artifact:
                # no artifact attached — treat as no retrieval
                source_names = []
                source_contents = []
            else:
                # loop through all of the chunks that the message has retrieved
                for doc in artifact:
                    # Check if the artifact is a langchain_core.documents.base.Document object (retrieval did occur), or a dict (retrieval didn't occur)
                    if isinstance(doc, Document):
                        # retrieval did occur, so return the doc names and contents
                        source_names.append(doc.metadata['title'])
                        source_contents.append(doc.page_content)
                    # Skip non-Document objects without clearing existing sources
        else:
            # this isn't a tool message, so keep looking
            pass
        i -= 1
    response = {
        "answer": last_ai_content,
        "source_names": source_names,
        "source_contents": source_contents
    }
    return response

def build_graph(llm, vector_store):
    # create a properly decorated tool bound to the vector store
    retrieve_bound = create_bound_retrieve_tool(vector_store)
    
    # bind llm and retrieve_tool into the nodes that need them
    query_node = partial(query_or_respond, llm=llm, retrieve_tool=retrieve_bound)
    generate_node = partial(generate, llm=llm)
    tool_node = ToolNode([retrieve_bound])

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("query_or_respond", query_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("generate", generate_node)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"}
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver() # for in-memory state handling
    graph = graph_builder.compile(checkpointer=memory)
    return graph

def format_sources(source_names, CI_docs_URLs):
    """Format source documents into links and create expander content"""
    if not source_names:
        return None

    # Remove duplicates while preserving order
    unique_sources = list(dict.fromkeys(source_names))
    source_links = []

    for source_name in unique_sources:
        # Convert file name to links to docs
        doc_row = CI_docs_URLs[CI_docs_URLs['File Name']==source_name]
        # if the file is in the CI Docs URLs table, add a hyperlink
        if doc_row.shape[0] > 0:
            doc_URL = doc_row.iloc[0,:]['File URL']
            source_links.append(f"[{source_name}]({doc_URL})")
        # if the file is missing from the CI Docs URLs table, just add a name
        else:
            source_links.append(source_name)

    # Create formatted source block for expander
    sources_content = f"**Most Relevant Document:**\n- {source_links[0]}"
    if len(source_links) > 1:
        sources_content += f"\n\n**Other Related Documents:**\n" + "\n".join(f"- {link}" for link in source_links[1:])

    return sources_content
