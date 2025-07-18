# AI Josh
An AI-powered assistant to answer questions about the contents of documents.

## Installation
1. Make sure you have poetry installed on your system. If not, follow [these](https://python-poetry.org/docs/#installation) instructions to install it.
2. Clone the repo and move into it
```
git clone https://github.com/Crown-Commercial-Service/ccs-ai-josh
cd ccs-ai-josh
```
3. Build the environment using poetry
```
poetry install
```

## Contribution Guidelines
When contributing to this codebase, please follow these guidelines:
1. The codebase uses a feature branch model, where new work is done on new feature branches created off the `develop` branch, and a pull request is submitted to merge the branch back into `develop`
2. Please name feature branches using the convention TICKET-ID-short-description-of-work e.g. AI-91-basic-rag-system
3. After creating a feature branch, make sure that your environment is up-to-date by running `poetry install`.
4. If you need to install a new package for your contribution, please add it to the environment using `poetry add` and commit your new `poetry.lock` file along with your code.
5. Under **no** circumstances should **any** code be pushed directly to the `main` branch.

## Document Embeddings
This system uses retrieval-augmented generation (RAG) to provide responses based on the content of relevant documents. These contents are stored in a vector store, which can be recreated by using the Azure AI Search Wizard. Broadly, this follows the standard process detailed [here](https://learn.microsoft.com/en-us/azure/search/search-get-started-portal-import-vectors?tabs=sample-data-storage%2Cmodel-aoai%2Cconnect-data-storage#supported-data-sources), but specific details are provided in the internal documentation.
Note: after creating the vector store, run `ccs_ai_josh/scripts/modify_index.py` to reformat the index to be compatible for use with LangChain.

## Document URIs
To link to the documents that an answer is based on, we use a table of file names and URIs that is updated regularly to reflect the latest documents that are available. For full instructions on how to create this table, see the internal documentation.