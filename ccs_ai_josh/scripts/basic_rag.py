import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# specify the embedding model
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_MODEL_KEY"),
)
print("Embedding model connected")
# specify the vector store
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.getenv("VECTOR_STORE_ENDPOINT"),
    azure_search_key=os.getenv("VECTOR_STORE_KEY"),
    index_name=os.getenv("VECTOR_STORE_INDEX"),
    embedding_function=embeddings.embed_query,
    fields=["document_id", "content", "url", "filename"]
)
print("Vector store connected")

# take input from command line as a surrogate for input from the front-end
user_input = input("What do you want to know?\n")
# Perform a similarity search
# Note: this breaks at the moment because the vector store doesn't actually contain any embeddings, just keywords
docs = vector_store.similarity_search(
    query=user_input,
    k=3,
    search_type="similarity",
)
print(docs[0].page_content)