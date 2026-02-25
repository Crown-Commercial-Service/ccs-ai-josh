import requests
import json
import os
import re
import pandas as pd
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pathlib import Path

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
        "params": json.dumps(
            {
                "apiFunction": "startSession",
                "apiParams": {"userEmail": user_email, "password": password},
            }
        )
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error starting session: {e}")
        return None


def list_docs(endpoint: str, group: str, token: str) -> dict:
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
        "params": json.dumps(
            {
                "apiFunction": "listDocuments",
                "apiToken": token,
                "apiGroupEmail": group,
                "apiParams": {"maxRows": "1000"},
            }
        )
    }
    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error starting session: {e}")
        return None


def doc_list_to_df(doc_data: dict) -> pd.DataFrame:
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
    for i in doc_data["querydata"]["data"]:
        # create a dict so that values for each entry can be extracted using the accompanying naming schema
        # this *should* future-proof against naming/ordering changes
        entry_dict = dict(zip(doc_data["querydata"]["columns"], i))
        doc_names.append(entry_dict["name"])
        doc_urls.append(entry_dict["getfileurl"])
        doc_mod_dates.append(entry_dict["modifydate"])
    doc_df = pd.DataFrame(
        {"Name": doc_names, "URL": doc_urls, "Modification Date": doc_mod_dates}
    )
    # remove documents that are not needed in AI Josh
    mask = ~(
        (doc_df["Name"].str.startswith("CCS"))
        | (doc_df["Name"].str.contains("Market Strategy", na=False))
        | (doc_df["Name"].str.contains("Market Summary", na=False))
    )
    return doc_df[mask]


def download_file(url: str, site: str, token: str, local_filepath: str) -> None:
    """
    Downloads a document to local storage.

    Args:
        name: the name of the file once it is downloaded
        url: the url pointing at the file to download
        group: The user's site index.
        token: The user's token for this session (retrieved with start_session()).
        local_filepath: The path on disk to write the file to

    Returns:
        None
    """
    # 1. Define the custom headers dictionary
    headers = {"APIToken": token}
    download_url = site + url
    try:
        # 2. Send an HTTP GET request with the defined headers
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # 3. Save the file content
            with open(local_filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(
            f"An error occurred during download (check URL, token, and permissions): {e}"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def retrieve_kahootz_file_info(
    endpoint: str, user_email: str, password: str, group: str
) -> pd.DataFrame:
    """
    Retrieves information on the files present in Kahootz.

    Args:
        endpoint (str): The user's endpoint URL
        user_email (str): The user's email address.
        password (str): The user's password.
        group: The user's group name.
    """
    session_response = start_session(
        endpoint=endpoint, user_email=user_email, password=password
    )
    print(session_response)
    if session_response:
        print("Session started successfully!")
        if session_response.get("status") == 9001:
            kahootz_site = session_response.get("siteurl")
            kahootz_token = session_response.get("tokenid")
            print("Session token retrieved")
        else:
            raise Exception(
                f"API returned status: {session_response.get('status')} - {session_response.get('status text')}"
            )
    else:
        raise Exception("Failed to start session.")

    doc_list = list_docs(endpoint=endpoint, group=group, token=kahootz_token)
    if doc_list:
        if doc_list.get("status") == 9001:
            print("Document list retrieved")
            doc_df = doc_list_to_df(doc_list)
            print(f"Details retrieved for {len(doc_df)} documents")
        else:
            raise Exception(
                f"API returned status: {doc_list.get('status')} - {doc_list.get('status text')}"
            )
    else:
        raise Exception("Failed to connect to session.")
    kahootz_file_info = {
        "kahootz_site": kahootz_site,
        "kahootz_token": kahootz_token,
        "doc_df": doc_df,
    }
    return kahootz_file_info


def list_blob_files(blob_url, container_name):
    """
    List the files in a blob storage container

    Args:
        blob_url: the URL of the blob storage
        container_name: the name of the container holding the files

    Returns:
        file_names: the names of the files in the container
    """
    # 1. Authenticate using DefaultAzureCredential
    # This automatically checks environment variables, Azure CLI, managed identity, etc.
    credential = DefaultAzureCredential()

    # 2. Create the BlobServiceClient
    blob_service_client = BlobServiceClient(blob_url, credential=credential)

    # 3. Get the container client
    container_client = blob_service_client.get_container_client(container_name)

    # 4. List the contents
    print(f"Listing blobs in container: {container_name}")
    for blob in container_client.list_blobs():
        print(blob.name)


def list_local_files(filepath: str) -> list:
    """
    Lists the filenames in a given directory on disk.

    Args:
        filepath: The path to the directory.

    Returns:
        A list of filenames and directory names, or an empty list if
        the path is invalid.
    """
    try:
        filenames = os.listdir(filepath)
        return filenames
    except FileNotFoundError:
        print(f"Error: The path '{filepath}' does not exist.")
        return []
    except NotADirectoryError:
        print(f"Error: The path '{filepath}' is not a directory.")
        return []


# list docs and write details to file
if __name__ == "__main__":
    # 1. Get Kahootz file info
    kahootz_file_info = retrieve_kahootz_file_info(
        endpoint=os.getenv("KAHOOTZ_ENDPOINT"),
        user_email=os.getenv("KAHOOTZ_EMAIL"),
        password=os.getenv("KAHOOTZ_KEY"),
        group=os.getenv("KAHOOTZ_GROUP"),
    )
    doc_df = kahootz_file_info["doc_df"]
    kahootz_site = kahootz_file_info["kahootz_site"]
    kahootz_token = kahootz_file_info["kahootz_token"]

    # 2. Get local storage file info
    # Note: change this to blob when possible
    local_files = list_local_files(filepath="./raw_docs")

    # 3. Identify files that are on Kahootz but not blob storage
    new_files = []
    for i in doc_df["Name"]:
        if i + ".pdf" not in local_files:
            new_files.append(i)
    new_files.sort()
    print("The following files will be downloaded:")
    for i in new_files:
        print(i)
    user_confirmation = input("Do you want to download them? [yes/no]: ").lower()
    if user_confirmation == "yes":
        doc_to_download_df = doc_df[doc_df["Name"].isin(new_files)]

        # 4. Download files that are on Kahootz but not blob storage
        # Note: while waiting for Azure permissions fix, we download all files
        for i in range(len(doc_to_download_df)):
            filetype = (
                doc_to_download_df.iloc[i, :]["URL"].split("/")[-1].split(".")[-1]
            )
            # remove any slashes in name to avoid disk write errors
            local_filename = (
                re.sub("/", "", doc_to_download_df.iloc[i, :]["Name"]) + "." + filetype
            )
            download_file(
                url=doc_to_download_df.iloc[i, :]["URL"],
                site=kahootz_site,
                token=kahootz_token,
                local_filepath=Path("raw_docs", local_filename),
            )
            if (i + 1) % 10 == 0:
                print(f"Downloaded {i+1} documents")
        print("File download complete")

    # 5. Delete files that are on blob storage but not Kahootz
