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

def list_docs(endpoint:str, group:str, token:str) -> dict:
    """
    Retrieves a dict of document names and metadata.

    Args:
        endpoint: The user's endpoint URL
        group: The user's group name.
        token: The user's token for this session (retrieved with start_session()).

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

def doc_list_to_df(doc_data:dict) -> pd.DataFrame:
    """
    Reformats the API return of listDocuments to a pandas dataframe.

    Args:
        doc_data: the contents of the return from a call to listDocuments, as a dict (as returned by list_docs())

    Returns:
        doc_df: a df with the name, path and modification date for each file
    """
    doc_names = []
    doc_urls = []
    doc_mod_dates = []
    for i in doc_data['querydata']['data']:
        # create a dict so that values for each entry can be extracted using the accompanying naming schema
        # this *should* future-proof against naming/ordering changes
        entry_dict = dict(zip(doc_list['querydata']['columns'], i))
        doc_names.append(entry_dict['name'])
        doc_urls.append(entry_dict['getfileurl'])
        doc_mod_dates.append(entry_dict['modifydate'])
    doc_df = pd.DataFrame({
        'Name': doc_names,
        'URL': doc_urls,
        'Modification Date': doc_mod_dates
    })
    return doc_df

def download_file(url:str, site:str, token:str) -> None:
    """
    Downloads a document.

    Args:
        url: the url pointing at the file to download
        group: The user's site index.
        token: The user's token for this session (retrieved with start_session()).

    Returns:
        None
    """
    local_filename = 'downloaded_file.pdf'
    # 1. Define the custom headers dictionary
    headers = {
        'APIToken': token
    }
    download_url = site + url
    try:
        # 2. Send an HTTP GET request with the defined headers
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # 3. Save the file content
            print(f"Downloading {url} to {local_filename}")
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"Successfully downloaded {local_filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download (check URL, token, and permissions): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# list docs and write details to file
if __name__ == "__main__":
    session_response = start_session(
        endpoint=os.getenv("KAHOOTZ_ENDPOINT"),
        user_email=os.getenv("KAHOOTZ_EMAIL"),
        password=os.getenv("KAHOOTZ_KEY")
    )
    print(session_response)
    if session_response:
        print("Session started successfully!")
        if session_response.get("status") == 9001:
            kahootz_site = session_response.get("siteurl")
            kahootz_token = session_response.get("tokenid")
            print(f"Session token retrieved")
        else:
            print(f"API returned status: {session_response.get('status')} - {session_response.get('status text')}")
    else:
        print("Failed to start session.")
    
    doc_list = list_docs(
        endpoint=os.getenv("KAHOOTZ_ENDPOINT"),
        group=os.getenv("KAHOOTZ_GROUP"), 
        token=kahootz_token
    )
    if doc_list:
        if doc_list.get("status") == 9001:
            print(f"Document list retrieved")
            doc_df = doc_list_to_df(doc_list)
            print(f"Details retrieved for {len(doc_df)} documents")
            print("First 5 documents:")
            print(doc_df.head())
        else:
            print(f"API returned status: {doc_list.get('status')} - {doc_list.get('status text')}")
    else:
        print("Failed to connect to session.")

    for i in len(doc_df):
        download_file(
            url=doc_df.iloc[:,i]['URL'],
            site=kahootz_site,
            token=kahootz_token,
            name=doc_df.iloc[:,i]['Name']
        )