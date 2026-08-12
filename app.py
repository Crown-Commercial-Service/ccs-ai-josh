import io
import json
import os
# Azure vector store holds the vectors in a field called "text_vector", not "content_vector" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
# Azure vector store holds the document contents in a field called "chunk", not "content" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
import uuid

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from markdown_it import MarkdownIt

from Feedback.feedback_mechanism import FeedbackMechanism
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from src.multiturn_utils import answer_once, build_graph, format_sources
from src.query_correction_engine import harden_vanna_sql, spell_correct_user_query
from src.sanitise import PromptInjectionError, sanitise_user_input
from src.text_to_sql import train_text_to_sql

# --- INITIALIZATION ---

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

graphs = {}

STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME")

fbm = FeedbackMechanism(
    storage_connection_string=STORAGE_CONNECTION_STRING, table_name=TABLE_NAME
)

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
    content_key="chunk",
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)

md = MarkdownIt()

COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")
checkpointer = CosmosDBSaver(
    database_name=COSMOS_DB_NAME, container_name=COSMOS_CONTAINER_NAME
)

azure_env = os.getenv("USE_AZURE", "true")
dummy_env = os.getenv("USE_DUMMY", "false")
retrain_vanna_env = os.getenv("RETRAIN_VANNA_MODEL", "false")

retrain_vanna = str(retrain_vanna_env).strip().lower() == "true"

model = train_text_to_sql(
    retrain_vanna=retrain_vanna
)


def _get_blob_service_client() -> BlobServiceClient:
    """Create the shared passwordless Blob Storage client."""
    blob_url = os.getenv("BLOB_URL")
    if not blob_url:
        raise ValueError("BLOB_URL environment variable is required.")
    return BlobServiceClient(
        account_url=blob_url,
        credential=DefaultAzureCredential(),
    )


def load_real_entities() -> dict | None:
    """Load and validate real_entities.json from Azure Blob Storage.

    Returning ``None`` preserves the query correction engine's existing local
    catalog fallback when Blob Storage is temporarily unavailable.
    """
    try:
        container_name = os.getenv("REAL_ENTITIES_CONTAINER")
        if not container_name:
            raise ValueError("REAL_ENTITIES_CONTAINER environment variable is required.")

        blob_client = _get_blob_service_client().get_blob_client(
            container=container_name,
            blob="real_entities.json",
        )
        entities = json.loads(blob_client.download_blob().readall())

        if not isinstance(entities, dict):
            raise ValueError("The entity catalog must be a JSON object.")
        if not isinstance(entities.get("suppliers"), list):
            raise ValueError("The entity catalog must contain a suppliers list.")
        if not isinstance(entities.get("frameworks"), list):
            raise ValueError("The entity catalog must contain a frameworks list.")

        return entities
    except Exception as exc:
        print(
            "Error loading real_entities.json from Blob Storage; "
            f"using the existing local catalog fallback: {exc}"
        )
        return None


def load_ci_docs_urls() -> pd.DataFrame:
    """Load CI document URLs from Azure Blob Storage."""
    try:
        container_client = _get_blob_service_client().get_container_client(
            os.getenv("BLOB_CONFIG_CONTAINER")
        )
        blob_client = container_client.get_blob_client("CI_document_URLs.csv")
        blob_data = blob_client.download_blob()
        ci_docs_urls = pd.read_csv(io.BytesIO(blob_data.readall()))
        return ci_docs_urls.rename(
            columns={"FileName": "File Name", "AzureURL": "File URL"}
        )
    except Exception as e:
        print(f"Error loading CSV from Blob Storage: {e}")
        return pd.DataFrame()


def get_user_id() -> str:
    """Ensure a stable per-session user ID."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


def get_or_create_graph(user_id: str):
    """Return the cached graph for a user, creating it on demand."""
    if user_id not in graphs:
        graphs[user_id] = build_graph(
            llm=llm, vector_store=vector_store, checkpointer=checkpointer
        )
    return graphs[user_id]


def build_history_context(graph_messages) -> str:
    """Create a compact text history context from recent human/ai messages."""
    clean_conversation = [
        msg
        for msg in graph_messages
        if msg.type in ("human", "ai") and not getattr(msg, "tool_calls", None)
    ]

    history_context = ""
    for msg in clean_conversation[-6:]:
        role_label = "user" if msg.type == "human" else "assistant"
        history_context += f"{role_label}: {msg.content}\n"
    return history_context


def run_sql_pipeline(compiled_query: str, catalog_data: dict | None = None):
    """Generate, harden, and execute SQL, returning dataframe and UI payload."""
    db_context = ""
    raw_ui_data = []
    df_results = None

    try:
        try:
            generated_sql = model.generate_sql(compiled_query)
            print(f"DEBUG: Vanna Raw SQL Output -> '{generated_sql}'")
        except Exception as vanna_err:
            print(f"❌ EXACT VANNA FAILURE CAUSE: {type(vanna_err).__name__}: {vanna_err}")
            generated_sql = None
        if not generated_sql or str(generated_sql).strip() == "":
            print("⚠️ Vanna returned an empty query string.")
            return None, [], "There is no structured SQL data available for this query."
        generated_sql = harden_vanna_sql(
            generated_sql,
            catalog_data=catalog_data,
        )
        print(f"Hardened Execution SQL Sent to DB: {generated_sql}")
        df_results = model.run_sql(generated_sql)

        if df_results is not None and not df_results.empty:
            raw_ui_data = df_results.head(20).to_dict(orient="records")
            if len(df_results) > 30:
                print("lots of rows more than 30")
                total_rows = len(df_results)
                numeric_summary = ""
                for col in df_results.select_dtypes(include=["number"]).columns:
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
            db_context = (
                "The query executed successfully but returned 0 rows matching these parameters."
            )
            raw_ui_data = []
    except Exception as db_err:
        db_context = f"No structured database matching fields: {str(db_err)}"
        raw_ui_data = []

    return df_results, raw_ui_data, db_context


def inject_results_into_graph(graph, config, db_context: str, df_results):
    """Store database results in graph state for the assistant response."""
    if df_results is not None and not df_results.empty:
        graph.update_state(
            config,
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "=== RETRIEVED STRUCTURED SQL DATA ===\n"
                            f"{db_context}\n"
                            "Blend these metrics into your final analysis answer where appropriate."
                        ),
                    }
                ]
            },
        )


def attach_table_data_to_latest_ai_message(graph, config, raw_ui_data):
    """Attach tabular data to the latest assistant message if present."""
    updated_state = graph.get_state(config)
    live_messages = updated_state.values.get("messages", [])
    if live_messages and live_messages[-1].type == "ai":
        live_messages[-1].additional_kwargs["full_table_data"] = raw_ui_data
        graph.update_state(config, {"messages": live_messages})


def format_chat_history(graph_messages, ci_docs_urls: pd.DataFrame):
    """Render LangGraph messages into template-friendly history objects."""
    formatted_history = []
    for msg in graph_messages:
        role = "user" if msg.type == "human" else "assistant"
        if msg.content and msg.type in ("human", "ai"):
            msg_data = {
                "role": role,
                "content": md.render(msg.content),
                "sources": "",
                "full_table": msg.additional_kwargs.get("full_table_data", None),
            }
            if role == "assistant":
                source_names = msg.additional_kwargs.get("source_names", [])
                if source_names and not ci_docs_urls.empty:
                    raw_sources = format_sources(source_names, ci_docs_urls)
                    msg_data["sources"] = md.render(raw_sources)
            formatted_history.append(msg_data)
    return formatted_history


@app.route("/", methods=["GET", "POST"])
def home():
    """Handles both displaying the chat and processing new messages via LangGraph."""
    ci_docs_urls = load_ci_docs_urls()
    user_id = get_user_id()
    config = {"configurable": {"thread_id": user_id}}

    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            try:
                user_input = sanitise_user_input(user_input)
            except PromptInjectionError:
                session["sanitisation_error"] = (
                    "Your message could not be processed because it appears to "
                    "contain a prompt injection attempt. Please rephrase your question."
                )
                session.modified = True
                return redirect(url_for("home"))
            except ValueError:
                session["sanitisation_error"] = (
                    "Your message could not be processed. Please ensure it is "
                    "non-empty and within the allowed length."
                )
                session.modified = True
                return redirect(url_for("home"))

            graph = get_or_create_graph(user_id)
            state = graph.get_state(config)
            graph_messages = state.values.get("messages", [])
            history_context = build_history_context(graph_messages)

            # Use the same Blob-hosted catalog for both query correction and
            # SQL hardening so the two stages cannot disagree about entities.
            real_entities = load_real_entities()
            user_input_sql = spell_correct_user_query(
                user_input=user_input,
                llm=llm,
                catalog_data=real_entities,
            )
            compiled_query = f"Context:\n{history_context}Current Request: {user_input_sql}"
            print(compiled_query)

            df_results, raw_ui_data, db_context = run_sql_pipeline(
                compiled_query,
                catalog_data=real_entities,
            )
            inject_results_into_graph(graph, config, db_context, df_results)

            response = answer_once(graph, user_input, thread_id=user_id)
            print(f"Graph processed response: {response}")
            attach_table_data_to_latest_ai_message(graph, config, raw_ui_data)

        return redirect(url_for("home"))

    formatted_history = []
    if user_id in graphs:
        graph = graphs[user_id]
        state = graph.get_state(config)
        graph_messages = state.values.get("messages", [])
        formatted_history = format_chat_history(graph_messages, ci_docs_urls)

    return render_template(
        "index.html",
        messages=formatted_history,
        sanitisation_error=session.pop("sanitisation_error", None),
    )


@app.route("/feedback", methods=["POST"])
def log_feedback():
    """Receives feedback data from the client-side JavaScript."""

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    thumbs_up_selected = data.get("thumbs_up_selected")
    assistant_content = data.get("assistant_content")
    user_content = data.get("user_content")
    feedback_text = data.get("feedback_text")
    project_name = "AI-Josh"
    ai_model = os.getenv("DEPLOYMENT_NAME")

    fbm.store_feedback(
        project_name=project_name,
        ai_model=ai_model,
        ai_response=assistant_content,
        user_query=user_content,
        feedback_about_response=feedback_text,
        thumbs=thumbs_up_selected,
    )

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
    app.run(debug=True, use_reloader=False)
