import os
from dotenv import load_dotenv
from typing import Optional, Iterator, Dict, Any
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.llm_utils import check_index_naming

load_dotenv()

# before connecting to anything, check that the vector store fields are compatible with langchain
index_client = SearchIndexClient(os.getenv("VECTOR_STORE_ENDPOINT"), AzureKeyCredential(os.getenv("VECTOR_STORE_KEY")))
vector_store_name_status = check_index_naming(index_client=index_client, index_name=os.getenv("VECTOR_STORE_INDEX"))
if vector_store_name_status:
    print("Vector store is named correctly")
else:
    raise Exception("Vector store naming is not compatible with LangChain")

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_MODEL_KEY"),
)
print("Embedding model connected")

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.getenv("VECTOR_STORE_ENDPOINT"),
    azure_search_key=os.getenv("VECTOR_STORE_KEY"),
    index_name=os.getenv("VECTOR_STORE_INDEX"),
    embedding_function=embeddings.embed_query,
    content_key="chunk"
)
print("Vector store connected")

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)
print("LLM connected")

#################
## CHAIN STEPS ##
#################

# 1. Generate an AI Message that may be a tool call or a direct answer
def query_or_respond(state: MessagesState):
    "Generate tool call for retrieval, or respond directly"
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # the response will contain the most recent response and the previous responses
    return {"messages": [response]}

# 2. Execute retrieval
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query"""
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs)
    )
    return serialized, retrieved_docs
tools = ToolNode([retrieve])

# 3. Generate a response using the retrieved context
def generate(state: MessagesState):
    """Generate answer"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    # format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep the"
        "answer concise."
        "\n\n"
        f"{docs_content}"
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

######################
## GRAPH DEFINITION ##
######################

def build_graph():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

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

graph = build_graph()

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
        messages = step.get("messages", [])
        if messages:
            final_messages = messages
            msg = messages[-1]
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

    def _tool_calls(m):
        if hasattr(m, "tool_calls"):
            return m.tool_calls
        if isinstance(m, dict):
            return m.get("tool_calls")
        return None

    def _content(m):
        if hasattr(m, "content"):
            c = m.content
        elif isinstance(m, dict):
            c = m.get("content", "")
        else:
            c = ""
        if isinstance(c, list):
            # Flatten list content (e.g. blocks) into text
            c = "\n".join(str(part) for part in c if part)
        return c if isinstance(c, str) else str(c)

    # Skip trailing final assistant message(s) without tool_calls
    i = len(final_messages) - 1
    while i >= 0 and _mtype(final_messages[i]) == "ai" and not _tool_calls(final_messages[i]):
        i -= 1

    # Collect the last consecutive block of tool messages
    context_parts = []
    while i >= 0 and _mtype(final_messages[i]) == "tool":
        context_parts.append(_content(final_messages[i]))
        i -= 1

    context_text = "\n\n".join(reversed([p for p in context_parts if p]))

    return {"answer": last_ai_content, "context": context_text}

while 1<2:
    user_input = input("What do you want to know?\n\n")
    # for step in graph.stream(
    #     {"messages": [{"role": "user", "content": user_input}]},
    #     stream_mode="values",
    #     config=config
    # ):
    #     step["messages"][-1].pretty_print()
    #     print(step["messages"][-1].keys())
    response = answer_once(graph, user_input)
    print(response['context'])
    print(response['answer'])

