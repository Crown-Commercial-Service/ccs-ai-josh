import os
# Azure vector store holds the vectors in a field called "text_vector", not "content_vector" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
# Azure vector store holds the document contents in a field called "chunk", not "content" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.identity import DefaultAzureCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from src.multiturn_utils import build_graph, answer_once, format_sources
from azure.storage.blob import BlobServiceClient
import io # Import the io module

st.set_page_config(layout="wide", page_title="AI Josh")

st.title("AI Josh")
st.write("An AI system to answer questions about Commercial Intelligence documents.")

load_dotenv()

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_MODEL_KEY"),
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.getenv("VECTOR_STORE_ENDPOINT"),
    azure_search_key=os.getenv("VECTOR_STORE_KEY"),
    index_name=os.getenv("VECTOR_STORE_INDEX"),
    embedding_function=embeddings.embed_query,
    content_key="chunk"
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# Initialize graph in session state to preserve conversation context
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(llm=llm, vector_store=vector_store)

# Read CI_document_URLs.csv from Azure Blob Storage
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url=os.getenv('BLOB_URL'), credential=credential)
container_client = blob_service_client.get_container_client(os.getenv("BLOB_CONFIG_CONTAINER"))
blob_client = container_client.get_blob_client("CI_document_URLs.csv")

# Download the blob content and read it into a pandas DataFrame
blob_data = blob_client.download_blob()
CI_docs_URLs = pd.read_csv(io.BytesIO(blob_data.readall()))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Add sources expander for assistant messages that have sources
        if message["role"] == "assistant" and message.get("sources"):
            sources_content = format_sources(message["sources"], CI_docs_URLs)
            if sources_content:
                with st.expander("🔗 View Sources", expanded=False):
                    st.markdown(sources_content)

if user_input := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = answer_once(st.session_state.graph, user_input)
    output = response["answer"]

    # Store message with sources if retrieval occurred
    message_data = {"role": "assistant", "content": output}
    if len(response['source_names']) > 0:
        message_data["sources"] = response['source_names']

    st.session_state.messages.append(message_data)

    with st.chat_message("assistant"):
        st.markdown(output)
        # Add sources in an expander if retrieval occurred
        if len(response['source_names']) > 0:
            sources_content = format_sources(response['source_names'], CI_docs_URLs)
            if sources_content:
                with st.expander("🔗 View Sources", expanded=False):
                    st.markdown(sources_content)

# Add fixed disclaimer at the bottom
st.markdown(
    """
    <style>
        .fixed-disclaimer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            colour: #333;
            text-align: centre;
            padding: 10px 0;
            font-size: 0.9rem;
            z-index: 9999;
        }
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none !important;}
        .stAppDeployButton {display:none !important;}
        [data-testid="stToolbar"] {display:none !important;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
    <div class="fixed-disclaimer">
         Disclaimer: AI-generated content may not always be accurate or up-to-date. Please verify critical information independently.
    </div>
    """,
    unsafe_allow_html=True
)