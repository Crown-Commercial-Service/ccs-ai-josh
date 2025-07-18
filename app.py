import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from typing_extensions import List, TypedDict
from src.llm_utils import check_index_naming, generate_response

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

CI_docs_URLs = pd.read_csv("data/CI_document_URLs.csv")

if user_input := st.chat_input("How can I help?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    response = generate_response(question=user_input, vector_store=vector_store, llm=llm)
    # collapsing to unique doc in case of multiple chunks from same doc
    docs = list(set(response['sources']))
    # convert file names to links to docs
    doc_links = []
    for i in docs:
        print(i)
        document_URL = CI_docs_URLs[CI_docs_URLs['File Name']==i].iloc[0,:]['File URL']
        print(document_URL)
        doc_link = f"[{i}]({document_URL})"
        print(doc_link)
        doc_links.append(doc_link)
    sources_formatted = f"\n\nCitations:"
    for i in doc_links:
        sources_formatted += f"\n\n* {i}"
    output = response["answer"] + sources_formatted
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