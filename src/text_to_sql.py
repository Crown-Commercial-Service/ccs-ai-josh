import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path

import pandas as pd
import pyodbc
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from vanna.legacy.azuresearch import AzureAISearch_VectorStore
from vanna.legacy.openai import OpenAI_Chat

load_dotenv()
logger = logging.getLogger("app.text_to_sql")


def _build_credential() -> DefaultAzureCredential:
    """Use the configured user-assigned identity in Azure and the local chain locally."""
    client_id = os.getenv("AZURE_CLIENT_ID")
    logger.info(
        "event=credential_created managed_identity_client_id_set=%s client_id_suffix=%s",
        bool(client_id),
        f"***{client_id[-6:]}" if client_id else "not-set",
    )
    return DefaultAzureCredential(
        managed_identity_client_id=client_id or None,
        # Deployed apps should not silently fall through to developer credentials.
        exclude_shared_token_cache_credential=bool(os.getenv("WEBSITE_HOSTNAME")),
        exclude_visual_studio_code_credential=bool(os.getenv("WEBSITE_HOSTNAME")),
        exclude_cli_credential=bool(os.getenv("WEBSITE_HOSTNAME")),
        exclude_powershell_credential=bool(os.getenv("WEBSITE_HOSTNAME")),
        exclude_developer_cli_credential=bool(os.getenv("WEBSITE_HOSTNAME")),
    )


azure_credential = _build_credential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)
client = AzureOpenAI(
    api_version=os.getenv("VANNA_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("VANNA_AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
)


class AzureTextToSQL(AzureAISearch_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        if not config:
            raise ValueError("Config dictionary is required for AzureTextToSQL initialization.")

        OpenAI_Chat.__init__(self, client=client, config=config)
        self.openai_client = client
        endpoint = config.get("azure_search_endpoint")
        index_name = config.get("index_name")
        if not endpoint or not index_name:
            raise ValueError("Both 'azure_search_endpoint' and 'index_name' must be provided in config.")

        logger.info(
            "event=vanna_search_client_initializing endpoint_set=%s index=%s",
            bool(endpoint),
            index_name,
        )
        self.search_index_client = SearchIndexClient(endpoint=endpoint, credential=azure_credential)
        self.search_client = SearchClient(
            endpoint=endpoint, index_name=index_name, credential=azure_credential
        )

    def generate_embedding(self, text: str):
        started = time.monotonic()
        deployment = os.getenv(
            "VANNA_AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )
        try:
            response = self.openai_client.embeddings.create(
                input=[text], model=deployment
            )
            logger.info(
                "event=vanna_embedding_success deployment=%s input_chars=%d duration_ms=%d",
                deployment,
                len(text),
                int((time.monotonic() - started) * 1000),
            )
            return response.data[0].embedding
        except Exception:
            logger.exception(
                "event=vanna_embedding_failure deployment=%s input_chars=%d duration_ms=%d",
                deployment,
                len(text),
                int((time.monotonic() - started) * 1000),
            )
            raise

    def _upload_to_azure_search(self, content: str, training_type: str, context: str = ""):
        import uuid

        vector = self.generate_embedding(content if not context else f"{context}\n{content}")
        document = {
            "id": str(uuid.uuid4()),
            "content": content,
            "text_vector": vector,
            "metadata": json.dumps({"type": training_type, "context": context}),
        }
        try:
            result = self.search_client.upload_documents(documents=[document])
            logger.info("event=vanna_training_upload_success type=%s", training_type)
            return result
        except Exception:
            logger.exception("event=vanna_training_upload_failure type=%s", training_type)
            raise

    def add_ddl(self, ddl: str, **kwargs) -> bool:
        return self._upload_to_azure_search(ddl, "ddl")

    def add_documentation(self, documentation: str, **kwargs) -> bool:
        return self._upload_to_azure_search(documentation, "documentation")

    def add_question_sql(self, question: str, sql: str, **kwargs) -> bool:
        return self._upload_to_azure_search(sql, "sql", question)

    def _search_azure_index(self, question: str, training_type: str, limit: int = 5) -> list:
        from azure.search.documents.models import VectorizedQuery

        started = time.monotonic()
        try:
            vector = self.generate_embedding(question)
            vector_query = VectorizedQuery(
                vector=vector, k_nearest_neighbors=limit, fields="text_vector"
            )
            results = self.search_client.search(
                search_text=None, vector_queries=[vector_query], top=limit
            )
            extracted_data = []
            scanned = 0
            for doc in results:
                scanned += 1
                try:
                    meta = json.loads(doc.get("metadata", "{}"))
                except (TypeError, json.JSONDecodeError):
                    meta = {}
                if meta.get("type") == training_type:
                    extracted_data.append(
                        {"content": doc.get("content"), "context": meta.get("context", "")}
                    )
            logger.info(
                "event=vanna_search_success type=%s scanned=%d matched=%d duration_ms=%d",
                training_type,
                scanned,
                len(extracted_data),
                int((time.monotonic() - started) * 1000),
            )
            return extracted_data
        except Exception:
            logger.exception(
                "event=vanna_search_failure type=%s duration_ms=%d",
                training_type,
                int((time.monotonic() - started) * 1000),
            )
            raise

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        results = self._search_azure_index(question, "sql")
        return [{"question": r["context"], "sql": r["content"]} for r in results]

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return [r["content"] for r in self._search_azure_index(question, "ddl")]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return [r["content"] for r in self._search_azure_index(question, "documentation")]


def initialise_agent():
    logger.info(
        "event=vanna_agent_initializing openai_endpoint_set=%s search_endpoint_set=%s index_set=%s",
        bool(os.getenv("VANNA_AZURE_OPENAI_ENDPOINT")),
        bool(os.getenv("VANNA_VECTOR_STORE_ENDPOINT")),
        bool(os.getenv("VANNA_INDEX_NAME")),
    )
    return AzureTextToSQL(
        client=client,
        config={
            "model": os.getenv("VANNA_MODEL", "gpt-5.4-mini"),
            "azure_search_endpoint": os.getenv("VANNA_VECTOR_STORE_ENDPOINT"),
            "index_name": os.getenv("VANNA_INDEX_NAME"),
        },
    )


def load_row_snippet():
    context_doc = Path(__file__).resolve().parent / "vanna_context_documents" / "row_snippet.md"
    return context_doc.read_text(encoding="utf-8") if context_doc.exists() else None


def train_text_to_sql(retrain_vanna=True):
    model = initialise_agent()
    sql_credential = _build_credential()
    db_server = os.getenv("PROD_DB_SERVER")
    db_name = os.getenv("PROD_DB_NAME")
    logger.info(
        "event=sql_configuration server_set=%s database_set=%s driver=ODBC_Driver_18",
        bool(db_server),
        bool(db_name),
    )
    if not db_server or not db_name:
        logger.error("event=sql_configuration_invalid missing=%s", ",".join(
            name for name, value in (("PROD_DB_SERVER", db_server), ("PROD_DB_NAME", db_name)) if not value
        ))

    connection_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={db_server};Database={db_name};Encrypt=yes;"
        "TrustServerCertificate=no;Connection Timeout=30;"
    )

    def custom_run_sql(sql: str) -> pd.DataFrame:
        operation_id = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:12]
        started = time.monotonic()
        logger.info(
            "event=sql_execution_started operation_id=%s sql_chars=%d server_set=%s database_set=%s",
            operation_id, len(sql), bool(db_server), bool(db_name),
        )
        try:
            token_started = time.monotonic()
            token_obj = sql_credential.get_token("https://database.windows.net/.default")
            logger.info(
                "event=sql_token_success operation_id=%s duration_ms=%d",
                operation_id, int((time.monotonic() - token_started) * 1000),
            )
            token_bytes = token_obj.token.encode("utf-16-le")
            token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

            with pyodbc.connect(connection_str, attrs_before={1256: token_struct}) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                if cursor.description is None:
                    logger.info("event=sql_execution_success operation_id=%s rows=0 duration_ms=%d",
                                operation_id, int((time.monotonic() - started) * 1000))
                    return pd.DataFrame()
                columns = [column[0] for column in cursor.description]
                frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
                logger.info("event=sql_execution_success operation_id=%s rows=%d columns=%d duration_ms=%d",
                            operation_id, len(frame), len(columns), int((time.monotonic() - started) * 1000))
                return frame
        except ClientAuthenticationError:
            logger.exception(
                "event=sql_identity_authentication_failure operation_id=%s hint=verify_AZURE_CLIENT_ID_and_UAMI_assignment",
                operation_id,
            )
            raise
        except pyodbc.Error:
            logger.exception(
                "event=sql_odbc_failure operation_id=%s hint=verify_ODBC_driver_SQL_firewall_private_DNS_and_database_user",
                operation_id,
            )
            raise
        except Exception:
            logger.exception("event=sql_execution_failure operation_id=%s", operation_id)
            raise

    model.run_sql = custom_run_sql
    model.run_sql_is_set = True

    if retrain_vanna:
        table_name = os.getenv("PROD_DB_TABLE_NAME")
        logger.info("event=vanna_retraining_started table_set=%s", bool(table_name))
        columns_query = f"""
            SELECT TABLE_CATALOG as [database], TABLE_SCHEMA as [table_schema],
                   TABLE_NAME as [table_name], COLUMN_NAME as [column_name],
                   DATA_TYPE as [data_type]
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
        """
        try:
            df_table_schema = model.run_sql(columns_query)
            model.train(plan=model.get_training_plan_generic(df_table_schema))
            row_snippet = load_row_snippet()
            if row_snippet is not None:
                model.add_documentation(row_snippet)
            logger.info("event=vanna_retraining_success")
        except Exception:
            logger.exception("event=vanna_retraining_failure")
    else:
        logger.info("event=vanna_retraining_skipped")

    return model
