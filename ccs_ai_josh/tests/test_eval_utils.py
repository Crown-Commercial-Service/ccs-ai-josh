import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.eval_utils import evaluate_response

load_dotenv()

# Connect to model on Azure to run the unit tests for LLM-as-a-judge eval functions
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0
)

# dummy data
q = "What is the capital of France?"
ref_ans = "The capital of France is Paris."
gen_ans_correct = "Paris is the capital of France."
gen_ans_incorrect = "Marseille is the capital of France"
chunk1 = "European capitals: UK=London, France=Paris, Spain=Madrid, Germany=Berlin"
chunk2 = "French cities: Marseille, Paris (capital), Lyon, Versailles"
chunk3 = "Spanish cities: Barcelona, Madrid, Seville"

def test_perfect_response():
    results = evaluate_response(
        llm=llm,
        question=q,
        response=gen_ans_correct,
        context=[chunk1, chunk2],
        ref_answer=ref_ans
    )
    assert results["correctness"] == 10
    assert results["retrieval"] == 10
    assert results["groundedness"] == 10

def test_incorrect_response():
    results = evaluate_response(
        llm=llm,
        question=q,
        response=gen_ans_incorrect,
        context=[chunk1, chunk2],
        ref_answer=ref_ans
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 10
    assert results["groundedness"] == 1