import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from src.llm_utils import check_index_naming
from src.multiturn_utils import build_graph, answer_once

st.set_page_config(layout="wide")

st.title("AI Josh")
st.write("An AI system to answer questions about Commercial Intelligence documents.")

load_dotenv()

# before connecting to anything, check that the vector store fields are compatible with langchain
index_client = SearchIndexClient(os.getenv("VECTOR_STORE_ENDPOINT"), AzureKeyCredential(os.getenv("VECTOR_STORE_KEY")))
vector_store_name_status = check_index_naming(index_client=index_client, index_name=os.getenv("VECTOR_STORE_INDEX"))
if vector_store_name_status == False:
    raise Exception("Vector store naming is not compatible with LangChain")

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

graph = build_graph(llm=llm, vector_store=vector_store)

CI_docs_URLs = pd.read_csv("data/CI_document_URLs.csv")

if user_input := st.chat_input("How can I help?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    response = answer_once(graph, user_input)
    # only displaying link to document that holds most relevant chunk
    relevant_doc = response['source_names'][0]
    # convert file name to links to docs
    doc_URL = CI_docs_URLs[CI_docs_URLs['File Name']==relevant_doc].iloc[0,:]['File URL']
    doc_link = f"\n\n[{relevant_doc}]({doc_URL})"
    output = response["answer"] + doc_link
    with st.chat_message("assistant"):
        st.markdown(output)

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
    </style>
    <div class="fixed-disclaimer">
         Disclaimer: AI-generated content may not always be accurate or up-to-date. Please verify critical information independently.
    </div>
    """,
    unsafe_allow_html=True
)