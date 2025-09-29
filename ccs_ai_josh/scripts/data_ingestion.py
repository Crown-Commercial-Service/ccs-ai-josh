import requests
import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def start_session(endpoint, user_email, password):
    """
    Starts a Kahootz API session and returns the session token.

    Args:
        endpoint (str): The user's endpoint URL
        user_email (str): The user's email address.
        password (str): The user's password.

    Returns:
        dict: A dictionary containing the API response, or None if the request fails.
    """
    url = endpoint
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    payload = {
        "params": json.dumps({
            "apiFunction": "startSession",
            "apiParams": {
                "userEmail": user_email,
                "password": password
            }
        })
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error starting session: {e}")
        return None

def get_doc_names(endpoint, group, token):
    """
    Retrieves a list of document names.

    Args:
        endpoint (str): The user's endpoint URL
        group (str): The user's group name.
        token (str): The user's token for this session (retrieved with start_session()).

    Returns:
        doc_names: A dictionary containing the API response, or None if the request fails.
    """
    url = endpoint
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "params": json.dumps({
            "apiFunction": "listDocuments",
            "apiToken": token,
            "apiGroupEmail": group,
            "apiParams": {
                'maxRows': '1000'
            }
        })
    }
    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error starting session: {e}")
        return None


# list docs and write details to file
if __name__ == "__main__":
    session_response = start_session(
        endpoint=os.getenv("KAHOOTZ_ENDPOINT"),
        user_email=os.getenv("KAHOOTZ_EMAIL"),
        password=os.getenv("KAHOOTZ_KEY")
    )
    if session_response:
        print("Session started successfully!")
        if session_response.get("status") == 9001:
            kahootz_token = session_response.get("tokenid")
            print(f"Session token retrieved")
        else:
            print(f"API returned status: {session_response.get('status')} - {session_response.get('status text')}")
    else:
        print("Failed to start session.")
    
    doc_list = get_doc_names(
        endpoint=os.getenv("KAHOOTZ_ENDPOINT"),
        group=os.getenv("KAHOOTZ_GROUP"), 
        token=kahootz_token
    )
    if doc_list:
        print("Session connected successfully!")
        if doc_list.get("status") == 9001:
            print(f"Document list retrieved")
            doc_names = []
            doc_urls = []
            doc_mod_dates = []
            for i in doc_list['querydata']['data']:
                # create a dict so that values for each entry can be extracted using the accompanying naming schema
                # this *should* future-proof against naming/ordering changes
                entry_dict = dict(zip(doc_list['querydata']['columns'], i))
                doc_names.append(entry_dict['name'])
                doc_urls.append(entry_dict['fullpath'])
                doc_mod_dates.append(entry_dict['modifydate'])
            doc_df = pd.DataFrame({
                'Name': doc_names,
                'URL': doc_urls,
                'Modification Date': doc_mod_dates
            })
            print(f"Details retrieved for {len(doc_df)} documents")
            print("First 5 documents:")
            print(doc_df.head())
        else:
            print(f"API returned status: {doc_list.get('status')} - {doc_list.get('status text')}")
    else:
        print("Failed to connect to session.")