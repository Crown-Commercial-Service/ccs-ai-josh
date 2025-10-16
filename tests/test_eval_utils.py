import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from eval_utils import evaluate_response

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
gen_ans_incorrect_1 = "Marseille is the capital of France"
gen_ans_incorrect_2 = "Madrid is the capital of Spain"
chunk1 = "European capitals: UK=London, France=Paris, Spain=Madrid, Germany=Berlin"
chunk2 = "French cities: Marseille, Paris (capital), Lyon, Versailles"
chunk3 = "Spanish cities: Barcelona, Madrid (capital), Seville"
doc_names = ['european_capitals.txt', 'french_cities.txt', 'spanish_cities.txt']

def test_correct():
    """
    The retrieved context is highly relevant,
    and the model gets the answer right.
    """
    results = evaluate_response(
        llm=llm,
        question=q,
        answer=gen_ans_correct,
        context=[chunk1, chunk2],
        retrieved_docs=doc_names[0:1],
        ref_answer=ref_ans,
        ref_doc=doc_names[0]
    )
    assert results["correctness"] == 10
    assert results["retrieval"] == 10
    assert results["groundedness"] == 10
    assert results["document_match"] == True

def test_correct_low_groundedness():
    """
    The retrieved context is irrelevant,
    the model ignores it and gets the answer right.
    """
    results = evaluate_response(
        llm=llm,
        question=q,
        answer=gen_ans_correct,
        context=[chunk3],
        retrieved_docs=[doc_names[2]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0]
    )
    assert results["correctness"] == 10
    assert results["retrieval"] == 1
    assert results["groundedness"] == 1
    assert results["document_match"] == False

def test_incorrect_low_groundedness():
    """
    The retrieved context is highly relevant,
    the model ignores it and gets the answer wrong.
    """
    results = evaluate_response(
        llm=llm,
        question=q,
        answer=gen_ans_incorrect_1,
        context=[chunk1, chunk2],
        retrieved_docs=[doc_names[0],doc_names[1]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0]
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 10
    assert results["groundedness"] == 1
    assert results["document_match"] == True

def test_incorrect_low_retrieval():
    """
    The retrieved context is irrelevant,
    the model uses it and gets the answer wrong.
    """
    results = evaluate_response(
        llm=llm,
        question=q,
        answer=gen_ans_incorrect_2,
        context=[chunk3],
        retrieved_docs=[doc_names[2],doc_names[0]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0]
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 1
    assert results["groundedness"] == 10
    assert results["document_match"] == False

def test_incorrect_low_groundedness_low_retrieval():
    """
    The retrieved context is irrelevant,
    the model ignores it but gets the answer wrong.
    """
    results = evaluate_response(
        llm=llm,
        question=q,
        answer=gen_ans_incorrect_1,
        context=[chunk3],
        retrieved_docs=[doc_names[2],doc_names[0]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0]
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 1
    assert results["groundedness"] == 1
    assert results["document_match"] == False