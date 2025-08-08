import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores.azuresearch import AzureSearch
from src.llm_utils import check_index_naming, generate_response
from src.eval_utils import evaluate_response

load_dotenv()

# before connecting to anything, check that the vector store fields are compatible with langchain
index_client = SearchIndexClient(os.getenv("VECTOR_STORE_ENDPOINT"), AzureKeyCredential(os.getenv("VECTOR_STORE_KEY")))
vector_store_name_status = check_index_naming(index_client=index_client, index_name=os.getenv("VECTOR_STORE_INDEX"))
if vector_store_name_status == False:
    raise Exception("Vector store naming is not compatible with LangChain")

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
    content_key="chunk"
)

# Connect to model on Azure
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# truthset needs to be downloaded from Google Drive and placed in `data` folder
truthset_file_path = os.path.join('data', 'AI Josh Truthset.xlsx')
truthset_sheet_name = 'Questions'

# Read the sheet into a pandas DataFrame
truthset = pd.read_excel(truthset_file_path, sheet_name=truthset_sheet_name)
# drop any blank columns that were inserted by parsing the spreadsheet
truthset = truthset.dropna(axis=1, how='all').reset_index()
# send each question to the model and store the response in the truthset df
responses = []
for i in truthset['Question']:
    response = generate_response(question=i, vector_store=vector_store, llm=llm)
    responses.append(response)
    if len(responses) % 10 == 0:
        print(f"Responses generated for {len(responses)} questions")
truthset["Answer"] = [i['answer'] for i in responses]
truthset["Retrieved Files"] = [i['source_names'] for i in responses]
truthset["Retrieved Contents"] = [i['source_contents'] for i in responses]
print("Responses generated for all questions")
# evaluate each response
correctness = []
retrieval = []
groundedness = []
document_match = []
for i in range(len(truthset)):
    score = evaluate_response(
        llm=llm,
        question=truthset['Question'][i],
        answer=truthset['Answer'][i],
        context=truthset['Retrieved Contents'][i],
        # strip filetype suffix from each filename before testing match
        retrieved_docs=[re.sub(r'.pdf$', '', i, flags=re.IGNORECASE) for i in truthset['Retrieved Files'][i]],
        ref_answer=truthset['True Answer'][i],
        ref_doc=truthset['File'][i]
    )
    correctness.append(score['correctness'])
    retrieval.append(score['retrieval'])
    groundedness.append(score['groundedness'])
    document_match.append(score['document_match'])
    if len(correctness) % 10 == 0:
        print(f"Responses evaluated for {len(correctness)} questions")
print("Responses evaluated for all questions")
# store the scores in the truthset df
truthset['Correctness'] = correctness
truthset['Retrieval'] = retrieval
truthset['Groundedness'] = groundedness
truthset['Document Match'] = document_match
# rearrange columns to put context columns on right, because these have loads of content
columns = list(truthset.columns)
cols_to_move = ['Retrieved Files', 'Retrieved Contents']
for col in cols_to_move:
    columns.remove(col)
columns.extend(cols_to_move)
truthset = truthset[columns]

# write results to file, including all columns in the truthset file
# deliberately not erasing the original truthset file
truthset.to_csv(os.path.join('data', 'evaluation_results.csv'), index=False)