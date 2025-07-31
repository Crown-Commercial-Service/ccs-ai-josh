import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.eval_utils import evaluate_response

load_dotenv()

# Connect to model on Azure
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# example data
q = "What is the capital of France?"
ref_ans = "The capital of France is Paris."
gen_ans_correct = "Paris is the capital of France."
gen_ans_incorrect = "Paris is the capital of Spain"
chunk1 = "European capitals: UK=London, France=Paris, Spain=Madrid, Germany=Berlin"
chunk2 = "French cities: Marseille, Paris (capital), Lyon, Versailles"
chunk3 = "Spanish cities: Barcelona, Madrid, Seville"

results = evaluate_response(
    llm=llm,
    question=q,
    response=gen_ans_correct,
    context=[chunk1, chunk2],
    ref_answer=ref_ans
)
print(f"Correctness = {results["correctness"]}")
print(f"Retrieval = {results["retrieval"]}")
print(f"Groundedness = {results["groundedness"]}")