import os
import uuid
from datetime import datetime
from azure.data.tables import TableClient
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME")

def db_to_pandas(db):
    rows = db.query_entities(query_filter="")
    return pd.DataFrame(list(rows))

def get_table_client():
    """Initializes and returns the TableClient."""
    try:
        table_client = TableClient.from_connection_string(
            conn_str=STORAGE_CONNECTION_STRING,
            table_name=TABLE_NAME
        )
        # df = db_to_pandas(table_client)
        return table_client
    except Exception as e:
        print(f" Error connecting to Azure Table: {e}")
        return None

def store_feedback( project_name: str,
    ai_model: str,
    thumbs: bool,
    user_query: str,
    feedback_about_response: str,
    ai_response: str):

    table_client = get_table_client()
    # partition key is needed
    partition_key = project_name
    timestamp_part = datetime.now().strftime("%Y%m%d%H%M%S")
    uuid_part = str(uuid.uuid4())[:8]
    row_key = f"{timestamp_part}-{uuid_part}"

    inserted_data = {
        'PartitionKey': partition_key,
        'RowKey': row_key,

        # Your Custom Columns
        'AI_Model': ai_model,
        'Project_Name': project_name,  # Stored redundantly for clarity/query flexibility
        'Thumbs': str(thumbs),  # Stored as True/False
        'User_Query': user_query,
        'User_Feedback': feedback_about_response,
        'AI_Response': ai_response

    }

    try:

        # Check if table exists, if so then just insert
        try:
            table_client.create_table()
        except Exception as e:
            print(f"Errror : {e}")

        table_client.upsert_entity(entity=inserted_data)

        print(f"✅ Success! Feedback logged for project '{project_name}'. RowKey: {row_key}")

    except Exception as e:
        print(f"❌ Failed to store data in Cosmos DB: {e}")


store_feedback(
    project_name="feedbackmechanism",
    ai_model=os.getenv("DEPLOYMENT_NAME"),
    thumbs=True,
    user_query="what is 1+1",
    feedback_about_response="correct",
    ai_response="2"

)