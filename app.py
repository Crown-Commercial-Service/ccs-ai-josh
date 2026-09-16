import io
import json
import logging
import os
import uuid
from pathlib import Path

# These must be set before importing the Azure Search integration.
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from markupsafe import escape

from Feedback.feedback_mechanism import FeedbackMechanism
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from src.data_asset_queries import (
    QueryRoute,
    answer_data_asset_query,
    classify_query_route,
)
from src.markdown_rendering import render_markdown_content
from src.multiturn_utils import answer_once, build_graph, format_sources
from src.query_correction_engine import harden_vanna_sql, spell_correct_user_query
from src.sanitise import PromptInjectionError, sanitise_user_input
from src.text_to_sql import train_text_to_sql

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
graphs = {}
DATA_ASSETS_PATH = Path(__file__).resolve().parent / "data_assets.json"

STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME")
fbm = FeedbackMechanism(
    storage_connection_string=STORAGE_CONNECTION_STRING, table_name=TABLE_NAME
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_MODEL_KEY"),
)
vector_store = AzureSearch(
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
checkpointer = CosmosDBSaver(
    database_name=os.getenv("COSMOS_DB_NAME"),
    container_name=os.getenv("COSMOS_CONTAINER_NAME"),
)
retrain_vanna = os.getenv("RETRAIN_VANNA_MODEL", "false").strip().lower() == "true"
model = train_text_to_sql(retrain_vanna=retrain_vanna)


def _get_blob_service_client() -> BlobServiceClient:
    blob_url = os.getenv("BLOB_URL")
    if not blob_url:
        raise ValueError("BLOB_URL environment variable is required.")
    return BlobServiceClient(account_url=blob_url, credential=DefaultAzureCredential())


def load_real_entities() -> dict | None:
    try:
        container_name = os.getenv("REAL_ENTITIES_CONTAINER")
        if not container_name:
            raise ValueError("REAL_ENTITIES_CONTAINER environment variable is required.")
        blob_client = _get_blob_service_client().get_blob_client(
            container=container_name, blob="real_entities.json"
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
        logger.exception(
            "REAL_ENTITY_CATALOG_LOAD_FAILED; using local fallback; error_type=%s",
            type(exc).__name__,
        )
        return None


def load_ci_docs_urls() -> pd.DataFrame:
    try:
        container = _get_blob_service_client().get_container_client(
            os.getenv("BLOB_CONFIG_CONTAINER")
        )
        data = container.get_blob_client("CI_document_URLs.csv").download_blob().readall()
        return pd.read_csv(io.BytesIO(data)).rename(
            columns={"FileName": "File Name", "AzureURL": "File URL"}
        )
    except Exception as exc:
        logger.exception("CI_DOCUMENT_URL_LOAD_FAILED error_type=%s", type(exc).__name__)
        return pd.DataFrame()


def get_user_id() -> str:
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


def get_or_create_graph(user_id: str):
    if user_id not in graphs:
        graphs[user_id] = build_graph(
            llm=llm, vector_store=vector_store, checkpointer=checkpointer
        )
    return graphs[user_id]


def build_history_context(graph_messages) -> str:
    clean = [
        msg
        for msg in graph_messages
        if msg.type in ("human", "ai") and not getattr(msg, "tool_calls", None)
    ]
    return "".join(
        f"{'user' if msg.type == 'human' else 'assistant'}: {msg.content}\n"
        for msg in clean[-6:]
    )


def run_sql_pipeline(compiled_query: str, catalog_data: dict | None = None):
    db_context, raw_ui_data, df_results = "", [], None
    try:
        try:
            generated_sql = model.generate_sql(compiled_query)
            logger.info(
                "VANNA_GENERATION_COMPLETED has_sql=%s sql_length=%d",
                bool(generated_sql and str(generated_sql).strip()),
                len(str(generated_sql or "")),
            )
        except Exception as exc:
            logger.exception("VANNA_GENERATION_FAILED error_type=%s", type(exc).__name__)
            generated_sql = None
        if not generated_sql or not str(generated_sql).strip():
            logger.warning("VANNA_GENERATION_EMPTY")
            return None, [], "There is no structured SQL data available for this query."

        generated_sql = harden_vanna_sql(generated_sql, catalog_data=catalog_data)
        logger.info("SQL_HARDENING_COMPLETED sql_length=%d", len(generated_sql))
        df_results = model.run_sql(generated_sql)
        if df_results is not None and not df_results.empty:
            raw_ui_data = df_results.head(20).to_dict(orient="records")
            if len(df_results) > 30:
                summary = ""
                for col in df_results.select_dtypes(include=["number"]).columns:
                    summary += f"- Total Sum of {col}: {df_results[col].sum():,.2f}\n"
                    summary += f"- Average of {col}: {df_results[col].mean():,.2f}\n"
                db_context = (
                    f"--- DATA OVERVIEW (COMPLETE REFRESH: {len(df_results)} ROWS) ---\n"
                    f"Calculated Aggregates across all rows:\n{summary}\n"
                    "--- STRUCTURE SAMPLE (FIRST 5 ROWS) ---\n"
                    f"{df_results.head(5).to_string(index=False)}"
                )
            else:
                db_context = df_results.to_string(index=False)
            logger.info("SQL_EXECUTION_COMPLETED row_count=%d", len(df_results))
        else:
            db_context = "The query executed successfully but returned 0 rows matching these parameters."
            logger.info("SQL_EXECUTION_COMPLETED row_count=0")
    except Exception as exc:
        logger.exception("SQL_PIPELINE_FAILED error_type=%s", type(exc).__name__)
        db_context = "No structured database matching fields."
    return df_results, raw_ui_data, db_context


def inject_results_into_graph(graph, config, db_context: str, df_results):
    if df_results is not None and not df_results.empty:
        graph.update_state(
            config,
            {"messages": [{"role": "system", "content": (
                "=== RETRIEVED STRUCTURED SQL DATA ===\n"
                f"{db_context}\n"
                "Use this data as authoritative for company metrics, totals, spend and counts. "
                "Any creation, ingestion or ETL date here is a Database Record Creation Date, "
                "not a document publication/update date. For questions about when a report or "
                "fact sheet was updated, use Document Publication Date from retrieved document "
                "context instead. Address every part of a multi-part user question."
            )}]},
        )


def attach_table_data_to_latest_ai_message(graph, config, raw_ui_data):
    state = graph.get_state(config)
    messages = state.values.get("messages", [])
    if messages and messages[-1].type == "ai":
        messages[-1].additional_kwargs["full_table_data"] = raw_ui_data
        graph.update_state(config, {"messages": messages})


def store_direct_answer(graph, config, user_input: str, answer: str):
    """Persist the exact user string and answer while bypassing Vanna/RAG."""
    graph.update_state(
        config,
        {"messages": [HumanMessage(content=user_input), AIMessage(content=answer)]},
    )


def format_chat_history(graph_messages, ci_docs_urls: pd.DataFrame):
    formatted = []
    for msg in graph_messages:
        role = "user" if msg.type == "human" else "assistant"
        if msg.content and msg.type in ("human", "ai"):
            rendered_content = (
                escape(msg.content)
                if role == "user"
                else render_markdown_content(msg.content)
            )
            item = {
                "role": role,
                "content": rendered_content,
                "raw_content": msg.content if role == "user" else None,
                "sources": "",
                "full_table": msg.additional_kwargs.get("full_table_data", None),
            }
            if role == "assistant":
                names = msg.additional_kwargs.get("source_names", [])
                if names and not ci_docs_urls.empty:
                    item["sources"] = render_markdown_content(
                        format_sources(names, ci_docs_urls)
                    )
            formatted.append(item)
    return formatted


@app.route("/", methods=["GET", "POST"])
def home():
    ci_docs_urls = load_ci_docs_urls()
    user_id = get_user_id()
    config = {"configurable": {"thread_id": user_id}}

    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            try:
                sanitised_input = sanitise_user_input(user_input)
            except PromptInjectionError:
                session["sanitisation_error"] = (
                    "Your message could not be processed because it appears to contain "
                    "a prompt injection attempt. Please rephrase your question."
                )
                return redirect(url_for("home"))
            except (TypeError, ValueError):
                session["sanitisation_error"] = (
                    "Your message could not be processed. Please ensure it is non-empty "
                    "and within the allowed length."
                )
                return redirect(url_for("home"))

            graph = get_or_create_graph(user_id)
            route = classify_query_route(user_input)
            logger.info(
                "REQUEST_ROUTED route=%s user_id=%s user_input=%r",
                route.value,
                user_id,
                user_input,
            )

            if route is QueryRoute.DATA_ASSET_CATALOG:
                answer = answer_data_asset_query(user_input, DATA_ASSETS_PATH)
                store_direct_answer(graph, config, user_input, answer)
                return redirect(url_for("home"))

            state = graph.get_state(config)
            history = build_history_context(state.values.get("messages", []))
            real_entities = load_real_entities()
            corrected = spell_correct_user_query(
                user_input=sanitised_input, llm=llm, catalog_data=real_entities
            )
            compiled_query = f"Context:\n{history}Current Request: {corrected}"
            df_results, table_data, db_context = run_sql_pipeline(
                compiled_query, catalog_data=real_entities
            )
            inject_results_into_graph(graph, config, db_context, df_results)
            answer_once(graph, user_input, thread_id=user_id)
            attach_table_data_to_latest_ai_message(graph, config, table_data)
        return redirect(url_for("home"))

    formatted_history = []
    if user_id in graphs:
        state = graphs[user_id].get_state(config)
        formatted_history = format_chat_history(
            state.values.get("messages", []), ci_docs_urls
        )
    return render_template(
        "index.html",
        messages=formatted_history,
        sanitisation_error=session.pop("sanitisation_error", None),
    )


@app.route("/feedback", methods=["POST"])
def log_feedback():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    fbm.store_feedback(
        project_name="AI-Josh",
        ai_model=os.getenv("DEPLOYMENT_NAME"),
        ai_response=data.get("assistant_content"),
        user_query=data.get("user_content"),
        feedback_about_response=data.get("feedback_text"),
        thumbs=data.get("thumbs_up_selected"),
    )
    return jsonify({"status": "success", "message": "Feedback logged"}), 200


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", use_reloader=False)
