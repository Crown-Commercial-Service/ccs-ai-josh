from src.llm_utils import check_index_naming
import os
# Azure vector store holds the vectors in a field called "text_vector", not "content_vector" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
# Azure vector store holds the document contents in a field called "chunk", not "content" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from src.multiturn_utils import build_graph, answer_once

load_dotenv()

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
