import os
from openai import AzureOpenAI
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.azuresearch import AzureAISearch_VectorStore
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv

load_dotenv()
azure_credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_version=os.getenv("VANNA_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("VANNA_AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider
)

class TextToSQL(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)

class AzureTextToSQL(AzureAISearch_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        if not config:
            raise ValueError("Config dictionary is required for AzureTextToSQL initialization.")

            # 1. Skip AzureAISearch_VectorStore.__init__ to avoid the mandatory API Key check
            # Instead, manually initialize the OpenAI Chat layout mixin base class
        OpenAI_Chat.__init__(self, client=client, config=config)

        # 2. Extract configuration fields
        endpoint = config.get('azure_search_endpoint')
        index_name = config.get('index_name')

        if not endpoint or not index_name:
            raise ValueError("Both 'azure_search_endpoint' and 'index_name' must be provided in config.")


        # Vanna's internal search methods specifically require these exact variable names
        self.search_index_client = SearchIndexClient(endpoint=endpoint, credential=azure_credential)
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=azure_credential)

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
            from  src.dummy_db import build_dummy_database
            build_dummy_database()


        model.connect_to_sqlite(db_file)

        # 💡 Check if Vanna already has training data stored locally
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
        model.connect_to_mssql(
            server=os.getenv("PROD_DB_SERVER"),
            database=os.getenv("PROD_DB_NAME"),
            username=os.getenv("PROD_DB_USER"),
            password=os.getenv("PROD_DB_PASSWORD")
        )
        try:
            existing_training = model.get_training_data()
        except Exception:
            import pandas as pd
            existing_training = pd.DataFrame()

            # 🏋️‍♂️ TRAIN PRODUCTION SCHEMA INTO AZURE VECTOR INDEX
        if existing_training.empty:
            print("🏋️‍♂️ Azure Search index is empty. Extracting active tables and schemas...")

            # Query the catalog to find user tables (Example uses SQL Server/Postgres catalog tables)
            schema_query = """
                        SELECT TABLE_NAME 
                        FROM INFORMATION_SCHEMA.TABLES 
                        WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo'
                    """
            df_tables = model.run_sql(schema_query)

            print(f"Found {len(df_tables)} production tables. Generating embeddings for Azure Search...")
            for table_name in df_tables['TABLE_NAME'].to_list():
                # Extract clean string definitions using Vanna's built-in schema dialect helper
                # and automatically save the structural block up into 'data-agent-db-knowledge'
                try:
                    ddl_text = model.get_create_table_statement(table_name)
                    model.train(ddl=ddl_text)
                    print(f" Successfully indexed schema for table: {table_name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not generate automated DDL for {table_name}: {e}")

            print("🚀 Production Azure Search training pass completed successfully!")
        else:
            print("🚀 Production Azure Search training cache found! Skipping training step.")

        return model

# model = train_text_to_sql(use_azure=False, use_dummy=True)
#
# from langchain_openai import  AzureChatOpenAI
# from dotenv import load_dotenv
# from src.query_correction_engine import spell_correct_user_query, harden_vanna_sql
# load_dotenv()
#
# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     azure_deployment=os.getenv("DEPLOYMENT_NAME"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     temperature=0.0,
# )
#
# cleaned_input = spell_correct_user_query(user_input="who ceo of microsoft and tell me the total spend of that company in the year 2024", llm=llm)
#
# sql_query = model.generate_sql(cleaned_input)
# sql_query = harden_vanna_sql(sql_query)
# print("________SQL QUERY _________")
# print(sql_query)
# print("_________OUTPUT_________")
# result = model.run_sql(sql_query)
# print(result.to_string(index=False))
