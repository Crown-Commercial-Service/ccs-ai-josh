import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from typing_extensions import List, TypedDict
from src.llm_utils import check_index_naming, generate_response

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

while 1<2:
    print("\n")
    user_input = input("What do you want to know?\n\n")
    print(generate_response(question=user_input, vector_store=vector_store, llm=llm))
