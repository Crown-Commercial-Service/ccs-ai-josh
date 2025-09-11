from typing import Any, Iterator, Dict
from functools import partial, wraps, WRAPPER_ASSIGNMENTS
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.documents.base import Document
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

def query_or_respond(state: MessagesState, llm: Any):
    "Generate tool call for retrieval, or respond directly"
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # the response will contain the most recent response and the previous responses
    return {"messages": [response]}

@tool(response_format="content_and_artifact")
def retrieve(query: str, vector_store: Any):
    """Retrieve information related to a query"""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

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
                        source_contents.append(doc.metadata['chunk'])
                    else:
                        # retrieval didn't occur, so don't return any citations
                        source_names = []
                        source_contents = []
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
    # bind llm into the nodes that need it
    query_node = partial(query_or_respond, llm=llm)
    generate_node = partial(generate, llm=llm)

    # bind vector_store into the retrieve tool
    # use a small wrapper so any tool metadata on the decorator is preserved
    def retrieve_bound(query: str):
        """Run the retrieve function on the vector store"""
        return retrieve(query, vector_store=vector_store)
    # copy all the default assigned attributes except __annotations__
    assigned_without_annotations = tuple(
        name for name in WRAPPER_ASSIGNMENTS if name != "__annotations__"
    )
    # copy metadata (including __dict__) from the original decorated function
    retrieve_bound = wraps(retrieve, assigned=assigned_without_annotations)(retrieve_bound)
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

from src.llm_utils import check_index_naming
import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

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

graph = build_graph(llm=llm, vector_store=vector_store)

while True:
    user_input = input("What do you want to know?\n")
    response = answer_once(graph, user_input)
    for i in range(len(response['source_contents'])):
        print(f"###### Chunk {i+1} ######")
        print(response['source_contents'][i])
    print(response['source_names'])
    print(response['answer'])
