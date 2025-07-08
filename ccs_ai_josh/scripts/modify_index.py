# this script should be run straight after creation of the vector store through the Azure AI Search wizard, before any searches
# at this point the vector store holds the vectors in a field called "text_vector", not "content_vector" as langchain expects
# it also holds the document contents in a field called "chunk", not "content" as langchain expects
# we will use the Azure SDK to copy the contents of the "text_vector" and "chunk" fields to new fields with appropriate names

import os
from dotenv import load_dotenv
load_dotenv()

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from copy import deepcopy

# 1. Fetch the Index Definition
index_client = SearchIndexClient(os.getenv("VECTOR_STORE_ENDPOINT"), AzureKeyCredential(os.getenv("VECTOR_STORE_KEY")))
index = index_client.get_index(os.getenv("VECTOR_STORE_INDEX"))

# 2. Update the Index Schema

# Find the definition of text_vector
for field in index.fields:
    if field.name == 'text_vector':
        text_vector_field = field
    elif field.name == 'chunk':
        text_contents_field = field

# Create a new field for content_vector with the same configuration
content_vector_field = deepcopy(text_vector_field)
content_vector_field.name = 'content_vector'
# Create a new field for content with the same configuration
content_field = deepcopy(text_contents_field)
content_field.name = 'content'

# Append new fields to index
index.fields.append(content_vector_field)
index.fields.append(content_field)

# 3. Update the Index
index_client.create_or_update_index(index)

# Copy Data from text_vector to content_vector
search_client = SearchClient(os.getenv("VECTOR_STORE_ENDPOINT"), os.getenv("VECTOR_STORE_INDEX"), AzureKeyCredential(os.getenv("VECTOR_STORE_KEY")))

documents = []
results = search_client.search("*", select="text_vector, chunk_id, parent_id, chunk, title", top=1000)
for doc in results:
    doc["content_vector"] = doc["text_vector"]
    doc["content"] = doc["chunk"]
    documents.append(doc)

# Now upload the updated docs (this will upsert them)
search_client.upload_documents(documents=documents)
