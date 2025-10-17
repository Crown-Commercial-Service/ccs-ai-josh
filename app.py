import os
import io
import uuid
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from src.multiturn_utils import build_graph, answer_once, format_sources

# --- INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()

# Set environment variables for LangChain Azure Search integration
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"

# Initialize Flask App
app = Flask(__name__)
# A secret key is required for Flask session management
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# In-memory storage for user-specific langgraph instances
# This preserves the state of each user's conversation across requests
graphs = {}

# --- AZURE & LANGCHAIN SETUP (runs once on startup) ---

# Configure Embeddings Model
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_MODEL_KEY"),
)

# Configure Vector Store
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.getenv("VECTOR_STORE_ENDPOINT"),
    azure_search_key=os.getenv("VECTOR_STORE_KEY"),
    index_name=os.getenv("VECTOR_STORE_INDEX"),
    embedding_function=embeddings.embed_query,
    content_key="chunk"
)

# Configure LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# Read CI document URLs from Azure Blob Storage
try:
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=os.getenv('BLOB_URL'), credential=credential)
    container_client = blob_service_client.get_container_client(os.getenv("BLOB_CONFIG_CONTAINER"))
    blob_client = container_client.get_blob_client("CI_document_URLs.csv")
    blob_data = blob_client.download_blob()
    CI_docs_URLs = pd.read_csv(io.BytesIO(blob_data.readall()))
except Exception as e:
    print(f"Error loading CSV from Blob Storage: {e}")
    CI_docs_URLs = pd.DataFrame() # Create an empty DataFrame on failure

# --- FLASK ROUTE ---

@app.route('/', methods=['GET', 'POST'])
def home():
    """Handles both displaying the chat and processing new messages."""
    
    # Create a unique session ID for the user if it doesn't exist
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['messages'] = []

    user_id = session['user_id']

    # POST: This block runs when the user sends a message
    if request.method == 'POST':
        user_input = request.form.get('message')
        if user_input:
            # Add user message to session history
            session['messages'].append({"role": "user", "content": user_input})

            # Get or create the langgraph for the current user
            if user_id not in graphs:
                graphs[user_id] = build_graph(llm=llm, vector_store=vector_store)
            graph = graphs[user_id]
            
            # Get response from the langgraph
            response = answer_once(graph, user_input)
            output = response["answer"]
            
            # Format sources if they exist
            sources_content = ""
            if response.get('source_names') and not CI_docs_URLs.empty:
                sources_content = format_sources(response['source_names'], CI_docs_URLs)

            # Add assistant response to session history
            assistant_message = {"role": "assistant", "content": output, "sources": sources_content}
            session['messages'].append(assistant_message)
            
            # Ensure the session is saved
            session.modified = True
        
        # Redirect to the same page with a GET request to show the updated chat
        return redirect(url_for('home'))

    # GET: This block runs on initial page load or after the redirect
    # It simply renders the page with the full message history
    return render_template('index.html', messages=session.get('messages', []))

if __name__ == '__main__':
    app.run(debug=True)