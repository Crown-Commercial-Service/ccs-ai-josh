import json
import os
import re
import struct
from pathlib import Path

import pandas as pd
import pyodbc
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from vanna.legacy.azuresearch import AzureAISearch_VectorStore
from vanna.legacy.openai import OpenAI_Chat

load_dotenv()
azure_credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version=os.getenv("VANNA_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("VANNA_AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
)

_CONTEXT_DOCUMENTS_DIR = Path(__file__).resolve().parent / "vanna_context_documents"
_EXAMPLE_PAIR_PATTERN = re.compile(
    r"\*\s*\*\*User Query:\*\*\s*[\"“](?P<question>.*?)[\"”]\s*"
    r"(?:\r?\n)+\s*```sql\s*(?P<sql>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_TIME_PERIOD_SQL_RULE = (
    "MANDATORY TIME-PERIOD SQL RULE: When a request specifies a financial year, "
    "relative year (including 'this year' or 'last year'), or date range and the SQL "
    "filters on that period, include the corresponding period column in SELECT beside "
    "every aggregate. For example use SELECT FinancialYear, SUM(EvidencedSpend) AS "
    "TotalEvidencedSpend ... GROUP BY FinancialYear. Never return a naked aggregate "
    "without the requested period identifier."
)


class AzureTextToSQL(AzureAISearch_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        if not config:
            raise ValueError("Config dictionary is required for AzureTextToSQL initialization.")

        # Skip AzureAISearch_VectorStore.__init__ to preserve passwordless setup.
        OpenAI_Chat.__init__(self, client=client, config=config)
        self.openai_client = client

        endpoint = config.get("azure_search_endpoint")
        index_name = config.get("index_name")
        if not endpoint or not index_name:
            raise ValueError(
                "Both 'azure_search_endpoint' and 'index_name' must be provided in config."
            )

        self.search_index_client = SearchIndexClient(
            endpoint=endpoint, credential=azure_credential
        )
        self.search_client = SearchClient(
            endpoint=endpoint, index_name=index_name, credential=azure_credential
        )

    def generate_embedding(self, text: str):
        response = self.openai_client.embeddings.create(
            input=[text],
            model=os.getenv(
                "VANNA_AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
            ),
        )
        return response.data[0].embedding

    def _upload_to_azure_search(
        self, content: str, training_type: str, context: str = ""
    ):
        import uuid

        vector = self.generate_embedding(content if not context else f"{context}\n{content}")
        document = {
            "id": str(uuid.uuid4()),
            "content": content,
            "text_vector": vector,
            "metadata": json.dumps({"type": training_type, "context": context}),
        }
        self.search_client.upload_documents(documents=[document])
        return True

    def add_ddl(self, ddl: str, **kwargs) -> bool:
        return self._upload_to_azure_search(content=ddl, training_type="ddl")

    def add_documentation(self, documentation: str, **kwargs) -> bool:
        return self._upload_to_azure_search(
            content=documentation, training_type="documentation"
        )

    def add_question_sql(self, question: str, sql: str, **kwargs) -> bool:
        return self._upload_to_azure_search(
            content=sql, training_type="sql", context=question
        )

    def _search_azure_index(
        self, question: str, training_type: str, limit: int = 5
    ) -> list:
        """Search a broad vector pool, then filter by our JSON metadata type."""
        from azure.search.documents.models import VectorizedQuery

        if limit <= 0:
            return []

        candidate_pool_size = max(limit * 4, 20)
        vector = self.generate_embedding(question)
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=candidate_pool_size,
            fields="text_vector",
        )
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=None,
            top=candidate_pool_size,
        )

        extracted_data = []
        for doc in results:
            try:
                metadata = json.loads(doc.get("metadata", "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}

            if metadata.get("type") != training_type:
                continue

            extracted_data.append(
                {
                    "content": doc.get("content"),
                    "context": metadata.get("context", ""),
                }
            )
            if len(extracted_data) >= limit:
                break

        return extracted_data

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        results = self._search_azure_index(question, "sql")
        return [{"question": item["context"], "sql": item["content"]} for item in results]

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return [item["content"] for item in self._search_azure_index(question, "ddl")]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """Return retrieved guidance plus invariant runtime SQL-generation rules."""
        documentation = [
            item["content"]
            for item in self._search_azure_index(question, "documentation")
        ]
        # Inject this at generation time so correctness does not depend on retraining
        # or on the time-period guidance being among the nearest vector neighbours.
        documentation.append(_TIME_PERIOD_SQL_RULE)
        return documentation


def initialise_agent():
    return AzureTextToSQL(
        client=client,
        config={
            "model": "gpt-5.4-mini",
            "azure_search_endpoint": os.getenv("VANNA_VECTOR_STORE_ENDPOINT"),
            "index_name": os.getenv("VANNA_INDEX_NAME"),
        },
    )


def _load_context_document(filename: str):
    context_doc = _CONTEXT_DOCUMENTS_DIR / filename
    return context_doc.read_text(encoding="utf-8") if context_doc.exists() else None


def load_row_snippet():
    return _load_context_document("row_snippet.md")


def load_examples_document():
    return _load_context_document("examples.md")


def parse_question_sql_examples(documentation: str) -> list[tuple[str, str]]:
    """Extract Markdown ``User Query`` and fenced SQL pairs in source order."""
    if not documentation:
        return []
    return [
        (match.group("question").strip(), match.group("sql").strip())
        for match in _EXAMPLE_PAIR_PATTERN.finditer(documentation)
    ]


def add_context_training_documents(model) -> int:
    """Upload local guidance and examples, returning the example-pair count."""
    row_snippet = load_row_snippet()
    if row_snippet:
        model.add_documentation(row_snippet)

    examples_document = load_examples_document()
    if not examples_document:
        return 0

    model.add_documentation(examples_document)
    examples = parse_question_sql_examples(examples_document)
    for question, sql in examples:
        model.add_question_sql(question=question, sql=sql)
    return len(examples)


def train_text_to_sql(retrain_vanna=True):
    model = initialise_agent()

    client_id = os.getenv("AZURE_CLIENT_ID")
    sql_credential = (
        DefaultAzureCredential(managed_identity_client_id=client_id)
        if client_id
        else DefaultAzureCredential()
    )
    db_server = os.getenv("PROD_DB_SERVER")
    db_name = os.getenv("PROD_DB_NAME")
    connection_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={db_server};Database={db_name};Encrypt=yes;TrustServerCertificate=yes;"
    )

    def custom_run_sql(sql: str) -> pd.DataFrame:
        token_obj = sql_credential.get_token("https://database.windows.net/.default")
        token_bytes = token_obj.token.encode("utf-16-le")
        token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

        with pyodbc.connect(connection_str, attrs_before={1256: token_struct}) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            if cursor.description is None:
                return pd.DataFrame()
            columns = [column[0] for column in cursor.description]
            return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)

    model.run_sql = custom_run_sql
    model.run_sql_is_set = True

    if retrain_vanna is True:
        print(
            f"🏋️‍♂️ Starting Retraining. Extracting schema for "
            f"{os.getenv('PROD_DB_TABLE_NAME')}..."
        )
        columns_query = f"""
            SELECT
                TABLE_CATALOG as [database],
                TABLE_SCHEMA as [table_schema],
                TABLE_NAME as [table_name],
                COLUMN_NAME as [column_name],
                DATA_TYPE as [data_type]
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'dbo'
              AND TABLE_NAME = '{os.getenv("PROD_DB_TABLE_NAME")}'
        """

        try:
            df_table_schema = model.run_sql(columns_query)
            plan = model.get_training_plan_generic(df_table_schema)
            model.train(plan=plan)
            example_count = add_context_training_documents(model)
            print(
                "🚀 Training pass completed successfully; "
                f"uploaded {example_count} question-to-SQL examples."
            )
        except Exception as exc:
            print(f"⚠️ Error during automated schema training: {exc}")
    else:
        print("🚀 No retraining commanded! Skipping training step.")

    return model
