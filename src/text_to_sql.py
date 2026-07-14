import os
from openai import AzureOpenAI
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.azuresearch import AzureAISearch_VectorStore
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
import pyodbc
import pandas as pd
from dotenv import load_dotenv
import struct

load_dotenv()
azure_credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_version=os.getenv("VANNA_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("VANNA_AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider
)

retrain_vanna = os.getenv("USE_AZURE", False)
retrain_vanna = str(retrain_vanna).strip().lower() == "true"



class TextToSQL(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)


class AzureTextToSQL(AzureAISearch_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        if not config:
            raise ValueError("Config dictionary is required for AzureTextToSQL initialization.")

        # 1. Safely skip AzureAISearch_VectorStore.__init__ to protect your passwordless setup
        OpenAI_Chat.__init__(self, client=client, config=config)
        self.openai_client = client

        # 2. Extract configuration fields
        endpoint = config.get('azure_search_endpoint')
        index_name = config.get('index_name')

        if not endpoint or not index_name:
            raise ValueError("Both 'azure_search_endpoint' and 'index_name' must be provided in config.")

        # Manually initialize clients using your Entra credentials
        self.search_index_client = SearchIndexClient(endpoint=endpoint, credential=azure_credential)
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=azure_credential)

    # 3. Handle embedding creation using Azure OpenAI
    def generate_embedding(self, text: str):
        response = self.openai_client.embeddings.create(
            input=[text],
            model=os.getenv("VANNA_AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        )
        return response.data[0].embedding

    # Helper to format and upload documents strictly mapping to YOUR custom Azure AI Search schema
    def _upload_to_azure_search(self, content: str, training_type: str, context: str = ""):
        import uuid
        import json

        # Calculate the vector embedding
        vector = self.generate_embedding(content if not context else f"{context}\n{content}")

        # Exact field mapping for YOUR index: id, content, metadata, text_vector
        document = {
            "id": str(uuid.uuid4()),
            "content": content,
            "text_vector": vector,  # Mapped to your text_vector field
            "metadata": json.dumps({
                "type": training_type,
                "context": context
            })
        }

        self.search_client.upload_documents(documents=[document])
        return True

    # 4. TRAINING INTERCEPTORS: Maps Vanna's training pipeline directly to your custom index keys
    def add_ddl(self, ddl: str, **kwargs) -> bool:
        return self._upload_to_azure_search(content=ddl, training_type="ddl")

    def add_documentation(self, documentation: str, **kwargs) -> bool:
        return self._upload_to_azure_search(content=documentation, training_type="documentation")

    def add_question_sql(self, question: str, sql: str, **kwargs) -> bool:
        return self._upload_to_azure_search(content=sql, training_type="sql", context=question)

    # 5. RETRIEVAL OVERRIDES: Tells Vanna how to pull and read data from your custom schema fields when querying
    def _search_azure_index(self, question: str, training_type: str, limit: int = 5) -> list:
        import json
        from azure.search.documents.models import VectorizedQuery

        vector = self.generate_embedding(question)
        vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=limit, fields="text_vector")

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=f"metadata/any(m: search.in(m, '{training_type}'))" if False else None,  # Simplified layout fallback
            top=limit
        )

        extracted_data = []
        for doc in results:
            try:
                meta = json.loads(doc.get("metadata", "{}"))
            except Exception:
                meta = {}

            if meta.get("type") == training_type:
                extracted_data.append({
                    "content": doc.get("content"),
                    "context": meta.get("context", "")
                })
        return extracted_data

    # Map Vanna's semantic search triggers to your tailored retrieval methods
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        # Expects a list of dicts with 'question' and 'sql' keys
        results = self._search_azure_index(question, "sql")
        return [{"question": r["context"], "sql": r["content"]} for r in results]

    def get_related_ddl(self, question: str, **kwargs) -> list:
        # Expects a list of strings
        results = self._search_azure_index(question, "ddl")
        return [r["content"] for r in results]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        # Expects a list of strings
        results = self._search_azure_index(question, "documentation")
        return [r["content"] for r in results]

def initialise_agent(use_azure=False):
    if use_azure:
        return AzureTextToSQL(
            client=client,
            config={
                'model': 'gpt-5.4-mini',
                'azure_search_endpoint': os.getenv("VANNA_VECTOR_STORE_ENDPOINT"),
                'index_name': 'data-agent-db-knowledge',
            }
        )
    else:
        return TextToSQL(
            client=client,
            config={
                'model': 'gpt-5.4-mini',
                'path': './azd_local_chroma'  # Completely free local storage
            }
        )


def train_text_to_sql(use_dummy=False, use_azure=False):
    model = initialise_agent(use_azure=use_azure)

    if use_dummy:
        db_file = "company_store.db"
        if not os.path.exists(db_file):
            from src.dummy_db import build_dummy_database
            build_dummy_database()

        model.connect_to_sqlite(db_file)

        # Check if Vanna already has training data stored locally
        existing_training = model.get_training_data()

        # Only train if the local vector storage is completely empty
        if existing_training.empty:
            print("🏋️‍♂️ No training data found. Training Vanna model...")
            df_ddl = model.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
            for ddl in df_ddl['sql'].to_list():
                model.train(ddl=ddl)
        else:
            print("🚀 Local training cache found! Skipping training step.")

        return model

    else:
        # 1. Fetch Entra ID token & build passwordless Azure SQL connection using pyodbc
        token_obj = azure_credential.get_token("https://database.windows.net/.default")
        token_bytes = token_obj.token.encode("utf-16-le")

        token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

        db_server = os.getenv("PROD_DB_SERVER")
        db_name = os.getenv("PROD_DB_NAME")

        connection_str = (
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server={db_server};"
            f"Database={db_name};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;"
        )

        # 2. re-write the custom SQL execution function to use pyodbc and inject the token
        def custom_run_sql(sql: str) -> pd.DataFrame:
            with pyodbc.connect(connection_str, attrs_before={1256: token_struct}) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)

                if cursor.description is None:
                    return pd.DataFrame()

                columns = [column[0] for column in cursor.description]
                return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)

        model.run_sql = custom_run_sql
        model.run_sql_is_set = True

        # # 3. Check for existing training data in Azure vector index
        # try:
        #     existing_training = model.get_training_data()
        # except Exception:
        #     existing_training = pd.DataFrame()

        # 4. Extract schema dynamically and train Vanna (No complex loops, no hardcoding)
        if retrain_vanna is True:
            print(f"🏋️‍♂️ Azure Search index is empty. Extracting schema for {os.getenv('PROD_DB_TABLE_NAME')}...")

            # Using exact aliases Vanna requires to map the database structure seamlessly
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
                # Fetch target table schema
                df_table_schema = model.run_sql(columns_query)

                # Generate the plan dynamically using Vanna's built-in generator
                plan = model.get_training_plan_generic(df_table_schema)

                # Train the model in one shot
                model.train(plan=plan)
                print("🚀 Production Azure Search training pass completed successfully!")
            except Exception as e:
                print(f"⚠️ Error during automated schema training: {e}")
        else:
            print("🚀 Production Azure Search training cache found! Skipping training step.")

        return model

model = train_text_to_sql(use_azure=True, use_dummy=False)

from langchain_openai import  AzureChatOpenAI
from dotenv import load_dotenv
from src.query_correction_engine import spell_correct_user_query, harden_vanna_sql
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)

cleaned_input = spell_correct_user_query(user_input="show me all the data first 5 rows", llm=llm)

sql_query = model.generate_sql(cleaned_input)
sql_query = harden_vanna_sql(sql_query)
print("________SQL QUERY _________")
print(sql_query)
print("_________OUTPUT_________")
result = model.run_sql(sql_query)
print(result.to_string(index=False))
