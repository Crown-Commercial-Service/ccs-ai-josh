import os

from gto.api import history

# Set environment variables for LangChain Azure Search integration
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
import io
import uuid
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from markdown_it import MarkdownIt
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from src.multiturn_utils import build_graph, answer_once, format_sources
from src.sanitise import sanitise_user_input, PromptInjectionError
from Feedback.feedback_mechanism import FeedbackMechanism
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from src.text_to_sql import train_text_to_sql
from src.query_correction_engine import spell_correct_user_query, harden_vanna_sql

# --- INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
# A secret key is required for Flask session management
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# In-memory storage for user-specific langgraph instances
# This preserves the state of each user's conversation across requests
graphs = {}

# --- AZURE & LANGCHAIN SETUP (runs once on startup) ---

# feedback mechanism
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME")

fbm = FeedbackMechanism(
    storage_connection_string=STORAGE_CONNECTION_STRING, table_name=TABLE_NAME
)

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
    content_key="chunk",
)

# Configure LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)


# Initialize the converter to convert source doc markdown to html
md = MarkdownIt()

COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")
checkpointer = CosmosDBSaver(
    database_name=COSMOS_DB_NAME, container_name=COSMOS_CONTAINER_NAME
)

# --- FLASK ROUTE ---
azure_env = os.getenv("USE_AZURE", False)
dummy_env = os.getenv("USE_DUMMY", True)
using_azure = str(azure_env).strip().lower() == "true"
using_dummy = str(dummy_env).strip().lower() == "true"

model = train_text_to_sql(use_azure= using_azure, use_dummy=using_dummy)


@app.route('/', methods=['GET', 'POST'])
def home():
    """Handles both displaying the chat and processing new messages via LangGraph."""
    #  Read CI document URLs from Azure Blob Storage to match links with names
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=os.getenv("BLOB_URL"), credential=credential
        )
        container_client = blob_service_client.get_container_client(
            os.getenv("BLOB_CONFIG_CONTAINER")
        )
        blob_client = container_client.get_blob_client("CI_document_URLs.csv")
        blob_data = blob_client.download_blob()
        CI_docs_URLs = pd.read_csv(io.BytesIO(blob_data.readall()))
        CI_docs_URLs = CI_docs_URLs.rename(
            columns={"FileName": "File Name", "AzureURL": "File URL"}
        )
    except Exception as e:
        print(f"Error loading CSV from Blob Storage: {e}")
        CI_docs_URLs = pd.DataFrame()

    # Identify the user (Thread/sessopm ID) - NO more 'messages' in session which caused last error
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    user_id = session['user_id']
    config = {"configurable": {"thread_id": user_id}}

    #  POST: This block runs when the user sends a message
    if request.method == 'POST':
        user_input = request.form.get('message')
        if user_input:
            # Sanitise input before passing to the LLM
            try:
                user_input = sanitise_user_input(user_input)
            except PromptInjectionError:
                session['sanitisation_error'] = (
                    "Your message could not be processed because it appears to "
                    "contain a prompt injection attempt. Please rephrase your question."
                )
                session.modified = True
                return redirect(url_for("home"))
            except ValueError:
                session['sanitisation_error'] = (
                    "Your message could not be processed. Please ensure it is "
                    "non-empty and within the allowed length."
                )
                session.modified = True
                return redirect(url_for("home"))

            # Get or create the langgraph for the current user
            if user_id not in graphs:
                graphs[user_id] = build_graph(llm=llm, vector_store=vector_store, checkpointer=checkpointer)

            graph = graphs[user_id]
            #  get context for vanna ai
            state = graph.get_state(config)
            graph_messages = state.values.get("messages", [])
            # Filter down to true conversation items only (skipping our system overrides)
            clean_conversation = [
                msg for msg in graph_messages
                if msg.type in ("human", "ai") and not getattr(msg, "tool_calls", None)
            ]
            history_context = ""
            for msg in clean_conversation[-6:]:  # Safely inspect the last 6 clean turns
                role_label = "user" if msg.type == "human" else "assistant"
                history_context += f"{role_label}: {msg.content}\n"
            user_input_sql = spell_correct_user_query(user_input=user_input, llm=llm) # TODO update json file name once you get real data
            compiled_query = f"Context:\n{history_context}Current Request: {user_input_sql}"
            print(compiled_query)
            #  Get data warehouse tables using Vanna SQL
            db_context = ""
            raw_ui_data = []
            df_results = None
            try:
                generated_sql = model.generate_sql(compiled_query)
                generated_sql = harden_vanna_sql(generated_sql)
                print(f"Hardened Execution SQL Sent to DB: {generated_sql}")
                df_results = model.run_sql(generated_sql)
                if df_results is not None and not df_results.empty:
                    raw_ui_data = df_results.to_dict(orient="records")

                    if len(df_results) > 30:
                        print("lots of rows more than 30")
                        total_rows = len(df_results)
                        numeric_summary = ""
                        for col in df_results.select_dtypes(include=['number']).columns:
                            numeric_summary += f"- Total Sum of {col}: {df_results[col].sum():,.2f}\n"
                            numeric_summary += f"- Average of {col}: {df_results[col].mean():,.2f}\n"

                        sample_df = df_results.head(5)

                        db_context = (
                            f"--- DATA OVERVIEW (COMPLETE REFRESH: {total_rows} ROWS) ---\n"
                            f"Calculated Aggregates across all rows:\n{numeric_summary}\n"
                            f"--- STRUCTURE SAMPLE (FIRST 5 ROWS) ---\n"
                            f"{sample_df.to_string(index=False)}"
                        )
                    else:
                        db_context = df_results.to_string(index=False)
                else:
                    db_context = "The query executed successfully but returned 0 rows matching these parameters."
                    raw_ui_data = []

            except Exception as db_err:
                db_context = f"No structured database matching fields: {str(db_err)}"
                raw_ui_data = []

            # INJECTION: Update LangGraph state thread with the SQL results before execution
            if df_results is not None and not df_results.empty:
                graph.update_state(
                    config,
                    {
                        "messages": [

                            {
                                "role": "system",
                                "content": f"--- RETRIEVED DATABASE TABLE DATA ---\n{db_context}\n"
                                           f"Blend these metrics into your final analysis answer where appropriate."
                            }
                        ]
                    }
                )
                # Get response from the langgraph - messages saves history to CosmoDB Saver automatically
                response = answer_once(graph, user_input, thread_id=user_id)
            else:
                response = answer_once(graph, user_input, thread_id=user_id)


            print(f"Graph processed response: {response}")
            updated_state = graph.get_state(config)
            live_messages = updated_state.values.get("messages", [])
            if live_messages and live_messages[-1].type == "ai":
                # Inject the data array into the fresh message's additional_kwargs backpack frame
                live_messages[-1].additional_kwargs["full_table_data"] = raw_ui_data
                graph.update_state(config, {"messages": live_messages})


        # Redirect to the same page with a GET request to show the updated chat
        return redirect(url_for("home"))

    #  Pull history from LangGraph CosmoDBSaver
    formatted_history = []
    if user_id in graphs:
        graph = graphs[user_id]
        state = graph.get_state(config)
        graph_messages = state.values.get("messages", [])

        for msg in graph_messages:
            role = "user" if msg.type == "human" else "assistant"


            if msg.content and msg.type in ("human", "ai"):
                # Initialize message data for the UI
                msg_data = {
                    "role": role,
                    "content": md.render(msg.content),
                    "sources": "",  # Default empty
                    "full_table": msg.additional_kwargs.get("full_table_data", None)
                }

                # If it's the assistant, check the "Backpack" for sources
                if role == "assistant":
                    # Looks for source_names saved in your generate node
                    source_names = msg.additional_kwargs.get("source_names", [])

                    if source_names and not CI_docs_URLs.empty:
                        # Format and render sources to HTML
                        raw_sources = format_sources(source_names, CI_docs_URLs)
                        msg_data["sources"] = md.render(raw_sources)

                formatted_history.append(msg_data)

    return render_template('index.html', messages=formatted_history, sanitisation_error=session.pop('sanitisation_error', None))


@app.route("/feedback", methods=["POST"])
def log_feedback():
    """Receives feedback data from the client-side JavaScript."""

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()

    # The four required values are now in the 'data' dictionary:
    thumbs_up_selected = data.get("thumbs_up_selected")
    assistant_content = data.get("assistant_content")
    user_content = data.get("user_content")
    feedback_text = data.get("feedback_text")  # This will be "no feedback" as requested
    project_name = "AI-Josh"
    ai_model = os.getenv("DEPLOYMENT_NAME")

    # --- LOGGING/STORAGE LOGIC GOES HERE ---
    fbm.store_feedback(
        project_name=project_name,
        ai_model=ai_model,
        ai_response=assistant_content,
        user_query=user_content,
        feedback_about_response=feedback_text,
        thumbs=thumbs_up_selected,
    )

    # Return a success message to the JavaScript
    return jsonify(
        {
            "status": "success",
            "message": "Feedback logged",
            "data_received": {
                "thumbs_up": thumbs_up_selected,
                "ai_content": assistant_content,
                "user_content": user_content,
                "text": feedback_text,
            },
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=True)
