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
4. Run the unit tests. If any fail, you may have an issue with your installation.
    ```
    poetry run pytest ccs_ai_josh/tests/
    ```
Note that these tests require an internet connection to run, because they call an LLM.


## Contribution Guidelines
When contributing to this codebase, please follow these guidelines:
1. The codebase uses a feature branch model, where new work is done on new feature branches created off the `develop` branch, and a pull request is submitted to merge the branch back into `develop`
2. Please name feature branches using the convention TICKET-ID-short-description-of-work e.g. AI-91-basic-rag-system
3. After creating a feature branch, make sure that your environment is up-to-date by running `poetry install`.
4. If you need to install a new package for your contribution, please add it to the environment using `poetry add` and commit your new `poetry.lock` file along with your code.
5. Under **no** circumstances should **any** code be pushed directly to the `main` branch.

## Building the AI Josh System
### Document Embeddings
AI Josh uses retrieval-augmented generation (RAG) to provide responses based on the content of relevant documents. These contents are stored in a vector store, which can be recreated by using the Azure AI Search Wizard. Broadly, this follows the standard process detailed [here](https://learn.microsoft.com/en-us/azure/search/search-get-started-portal-import-vectors?tabs=sample-data-storage%2Cmodel-aoai%2Cconnect-data-storage#supported-data-sources), but specific details are provided in the internal documentation.

### Document URLs
To link to the documents that an answer is based on, we use a file called `CI_document_URLs.csv`, which is a table of file names and URLs. For full instructions on how to create this table, see the internal documentation.

## Running the User Interface Locally
If you want to run the user interface locally for development or testing purposes, run:
```
poetry run streamlit run app.py
```

## Running Model Evaluation
If you want to evaluate the performance of the system against the truthset:
1. Download the file `AI Josh Truthset.xlsx` from Google Drive
2. Place it in `ccs_ai_josh/data/`
3. Run the evaluation pipeline:
    ```
    dvc repro
    ```
Note that the full evaluation pipeline submits a total of 664 queries to an LLM each time it is run, so plan your evaluation experiments carefully.

## Current Performance
As of 13/08/2025, AI Josh has the following performance characteristics:
* 100% document retrieval accuracy for positive control questions
* High correctness of response for positive control questions on fact sheets (9.2/10) and marketing reports (8.6/10)
* Moderate correctness of response for negative control questions on fact sheets (7.9/10)
* Low correctness of response for negative control questions on fact sheets (4.2/10)
