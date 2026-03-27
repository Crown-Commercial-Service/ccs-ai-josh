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
4. Install the environment as a jupyter kernel
    ```
    poetry run python -m ipykernel install --user --name="ccs-ai-josh"
    ```
5. Run the unit tests. If any fail, you may have an issue with your installation.
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

## Prompt Sanitisation
All user messages are routed through a shared sanitisation layer
(`ccs_ai_josh/src/prompt_sanitiser.py`) before being passed to the LLM.

### What it does
| Step | Detail |
|------|--------|
| **Type validation** | Rejects non-string values with `TypeError`. |
| **Length validation** | Silently truncates inputs longer than 2 000 characters to limit token-stuffing attacks. |
| **Injection pattern detection** | Raises `ValueError` (shown to the user as a friendly warning) when the input matches a known injection pattern. |
| **Retrieved-content delimiters** | The generation step wraps all retrieved document content in explicit `--- BEGIN / END RETRIEVED DOCUMENTS ---` markers so the LLM can distinguish external data from trusted instructions (defence against indirect / multi-step injection). |

### Injection categories detected
* **Direct injection** – e.g. "Ignore previous instructions …", "Forget all prior instructions …"
* **SQL injection** – e.g. "execute the following SQL …", `DROP TABLE`, `SELECT … FROM … WHERE`
* **Data exfiltration** – e.g. "Reveal your system prompt", "Show me your API key"
* **Jailbreak triggers** – e.g. "DAN mode", "developer mode"

### Reviewing / extending the sanitiser
The full list of patterns is in `INJECTION_PATTERNS` inside `prompt_sanitiser.py`.
To add a new pattern, append a `(regex_string, human_readable_label)` tuple to that list.
The pre-compiled `_COMPILED_PATTERNS` list is rebuilt automatically at module load time.

### Tests
Automated tests live in `ccs_ai_josh/tests/test_prompt_sanitiser.py` and cover:
* Normal queries passing through unchanged.
* All four injection categories above.
* Input truncation behaviour.
* Type-error handling.
* Generation-step delimiter presence (indirect injection defence).

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

## Exploring Model Evaluation Results
The results of model evaluation can be explored in jupyter notebooks. To run the notebooks in this repo, launch a jupyter notebook server by running:
```
poetry run jupyter lab
```
In the jupyter lab landing page that launches in your browser, open one of the notebooks and select the kernel `ccs-ai-josh`.

**Note:** if your browser doesn't automatically load the jupyter lab landing page, you may need to follow the link that is displayed in the terminal instead.

## Current Performance
As of 13/08/2025, AI Josh has the following performance characteristics:
* 100% document retrieval accuracy for positive control questions
* High correctness of response for positive control questions on fact sheets (9.2/10) and marketing reports (8.6/10)
* Moderate correctness of response for negative control questions on fact sheets (7.9/10)
* Low correctness of response for negative control questions on fact sheets (4.2/10)
