import os
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

def check_index_naming(index_client, index_name:str) -> bool:
    """Checks if the naming of columns in an index is compatible with LangChain expectations.
    Args:
        index_client: An instance of SearchIndexClient (or a mock).
        index_name: the name of the index
    
    Returns:
        bool: True if both 'content_vector' and 'content' fields exist, False otherwise.
    """
    index = index_client.get_index(index_name)

    vector_name = False
    content_name = False
    for field in index.fields:
        if field.name == 'content_vector':
            vector_name = True
        elif field.name == 'content':
            content_name = True
    if vector_name and content_name:
        return True
    else:
        return False

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

# Define application steps
def retrieve(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def generate_response(question:str, vector_store, llm) -> dict:
    """Invoke the LLM graph to generate a response to a question.
    """
    # wrapper functions used here, because:
    # we need to pass two arguments into the retrieval and generation steps 
    # BUT langgraph's add_sequence only accepts functions with one arg (state) 
    def retrieve_with_store(state: State):
        return retrieve(state, vector_store)
    def generate_with_llm(state: State):
        return generate(state, llm)
    graph_builder = StateGraph(State).add_sequence([
        retrieve_with_store,
        generate_with_llm
    ])
    graph_builder.add_edge(START, "retrieve_with_store")
    graph = graph_builder.compile()
    response = graph.invoke({"question": question})
    source_names = [doc.metadata["title"] for doc in response["context"]]
    source_contents = [doc.metadata["chunk"] for doc in response["context"]]
    output = {'answer':response["answer"], 'source_names':source_names, 'source_contents':source_contents}
    return(output)