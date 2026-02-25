import os

# Azure vector store holds the vectors in a field called "text_vector", not "content_vector" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"
# Azure vector store holds the document contents in a field called "chunk", not "content" as langchain expects
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from multiturn_utils import build_graph, answer_once
from eval_utils import evaluate_response

load_dotenv()

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_MODEL_KEY"),
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.getenv("VECTOR_STORE_ENDPOINT"),
    azure_search_key=os.getenv("VECTOR_STORE_KEY"),
    index_name=os.getenv("VECTOR_STORE_INDEX"),
    embedding_function=embeddings.embed_query,
    content_key="chunk",
)

# Connect to model on Azure
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)

# truthset needs to be downloaded from Google Drive and placed in `data` folder
truthset_file_path = os.path.join("data", "AI Josh Truthset.xlsx")
truthset_sheet_name = "Questions"

# Read the sheet into a pandas DataFrame
truthset = pd.read_excel(truthset_file_path, sheet_name=truthset_sheet_name)
# drop any blank columns that were inserted by parsing the spreadsheet
truthset = truthset.dropna(axis=1, how="all").reset_index(drop=True)
# send each question to the model and store the response in the truthset df
responses = []
for i in truthset["Question"]:
    # build the graph each time, to clear context
    graph = build_graph(llm=llm, vector_store=vector_store)
    response = answer_once(graph, i)
    responses.append(response)
    if len(responses) % 10 == 0:
        print(f"Responses generated for {len(responses)} questions")
truthset["Answer"] = [i["answer"] for i in responses]
truthset["Retrieved Files"] = [i["source_names"] for i in responses]
truthset["Retrieved Contents"] = [i["source_contents"] for i in responses]
print("Responses generated for all questions")
# evaluate each response
correctness = []
retrieval = []
groundedness = []
document_match = []
for i in range(len(truthset)):
    # reformat the retrieved doc names to match the naming convention in the truthset
    retrieved_docnames = [
        re.sub(r".pdf$", "", j, flags=re.IGNORECASE)
        for j in truthset["Retrieved Files"][i]
    ]
    retrieved_docnames = [re.sub(r"One Page ", "", k) for k in retrieved_docnames]
    retrieved_docnames = [docname.strip() for docname in retrieved_docnames]
    score = evaluate_response(
        llm=llm,
        question=truthset["Question"][i],
        answer=truthset["Answer"][i],
        context=truthset["Retrieved Contents"][i],
        retrieved_docs=retrieved_docnames,
        ref_answer=truthset["True Answer"][i],
        ref_doc=truthset["File"][i].strip(),
    )
    correctness.append(score["correctness"])
    retrieval.append(score["retrieval"])
    groundedness.append(score["groundedness"])
    document_match.append(score["document_match"])
    if len(correctness) % 10 == 0:
        print(f"Responses evaluated for {len(correctness)} questions")
print("Responses evaluated for all questions")
# store the scores in the truthset df
truthset["Correctness"] = correctness
truthset["Retrieval"] = retrieval
truthset["Groundedness"] = groundedness
truthset["Document Match"] = document_match
# rearrange columns to put context columns on right, because these have loads of content
columns = list(truthset.columns)
cols_to_move = ["Retrieved Files", "Retrieved Contents"]
for col in cols_to_move:
    columns.remove(col)
columns.extend(cols_to_move)
truthset = truthset[columns]

# write results to file, including all columns in the truthset file
# deliberately not erasing the original truthset file
truthset.to_csv(os.path.join("data", "evaluation_results.csv"), index=False)
