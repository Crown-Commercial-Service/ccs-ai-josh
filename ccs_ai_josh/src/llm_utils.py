import os
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

def check_index_naming(index_client, index_name:str) -> bool:
    """Checks if the naming of columns in an index is compatible with LangChain expectations.
    Args:
        index_client: An instance of SearchIndexClient (or a mock).
        index_name: the name of the index
    
    Returns:
        bool: True if both 'content_vector' and 'content' fields exist, False otherwise.
    """
    index = index_client.get_index(index_name)

    vector_name = False
    content_name = False
    for field in index.fields:
        if field.name == 'content_vector':
            vector_name = True
        elif field.name == 'content':
            content_name = True
    if vector_name and content_name:
        return True
    else:
        return False