import eval_utils
from eval_utils import evaluate_response

# dummy data
q = "What is the capital of France?"
ref_ans = "The capital of France is Paris."
gen_ans_correct = "Paris is the capital of France."
gen_ans_incorrect_1 = "Marseille is the capital of France"
gen_ans_incorrect_2 = "Madrid is the capital of Spain"
chunk1 = "European capitals: UK=London, France=Paris, Spain=Madrid, Germany=Berlin"
chunk2 = "French cities: Marseille, Paris (capital), Lyon, Versailles"
chunk3 = "Spanish cities: Barcelona, Madrid (capital), Seville"
doc_names = ["european_capitals.txt", "french_cities.txt", "spanish_cities.txt"]


def _patch_scores(monkeypatch, *, correctness, retrieval, groundedness, document_match):
    monkeypatch.setattr(eval_utils, "score_correctness", lambda **kwargs: correctness)
    monkeypatch.setattr(eval_utils, "score_retrieval", lambda **kwargs: retrieval)
    monkeypatch.setattr(eval_utils, "score_groundedness", lambda **kwargs: groundedness)
    monkeypatch.setattr(eval_utils, "test_doc_match", lambda **kwargs: document_match)


def test_correct(monkeypatch):
    _patch_scores(
        monkeypatch, correctness=10, retrieval=10, groundedness=10, document_match=True
    )

    results = evaluate_response(
        llm=object(),
        question=q,
        answer=gen_ans_correct,
        context=[chunk1, chunk2],
        retrieved_docs=doc_names[0:1],
        ref_answer=ref_ans,
        ref_doc=doc_names[0],
    )
    assert results["correctness"] == 10
    assert results["retrieval"] == 10
    assert results["groundedness"] == 10
    assert results["document_match"] is True


def test_correct_low_groundedness(monkeypatch):
    _patch_scores(
        monkeypatch, correctness=10, retrieval=1, groundedness=1, document_match=False
    )

    results = evaluate_response(
        llm=object(),
        question=q,
        answer=gen_ans_correct,
        context=[chunk3],
        retrieved_docs=[doc_names[2]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0],
    )
    assert results["correctness"] == 10
    assert results["retrieval"] == 1
    assert results["groundedness"] == 1
    assert results["document_match"] is False


def test_incorrect_low_groundedness(monkeypatch):
    _patch_scores(
        monkeypatch, correctness=1, retrieval=10, groundedness=1, document_match=True
    )

    results = evaluate_response(
        llm=object(),
        question=q,
        answer=gen_ans_incorrect_1,
        context=[chunk1, chunk2],
        retrieved_docs=[doc_names[0], doc_names[1]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0],
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 10
    assert results["groundedness"] == 1
    assert results["document_match"] is True


def test_incorrect_low_retrieval(monkeypatch):
    _patch_scores(
        monkeypatch, correctness=1, retrieval=1, groundedness=10, document_match=False
    )

    results = evaluate_response(
        llm=object(),
        question=q,
        answer=gen_ans_incorrect_2,
        context=[chunk3],
        retrieved_docs=[doc_names[2], doc_names[0]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0],
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 1
    assert results["groundedness"] == 10
    assert results["document_match"] is False


def test_incorrect_low_groundedness_low_retrieval(monkeypatch):
    _patch_scores(
        monkeypatch, correctness=1, retrieval=1, groundedness=1, document_match=False
    )

    results = evaluate_response(
        llm=object(),
        question=q,
        answer=gen_ans_incorrect_1,
        context=[chunk3],
        retrieved_docs=[doc_names[2], doc_names[0]],
        ref_answer=ref_ans,
        ref_doc=doc_names[0],
    )
    assert results["correctness"] == 1
    assert results["retrieval"] == 1
    assert results["groundedness"] == 1
    assert results["document_match"] is False
