from src import eval_utils


def test_evaluate_response_with_retrieval(monkeypatch):
    llm = object()
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."
    context = ["European capitals: France=Paris"]
    retrieved_docs = ["european_capitals.txt"]
    ref_answer = "The capital of France is Paris."
    ref_doc = "european_capitals.txt"

    monkeypatch.setattr(eval_utils, "score_correctness", lambda **kwargs: 10)
    monkeypatch.setattr(eval_utils, "score_retrieval", lambda **kwargs: 9)
    monkeypatch.setattr(eval_utils, "score_groundedness", lambda **kwargs: 8)
    monkeypatch.setattr(eval_utils, "test_doc_match", lambda **kwargs: True)

    results = eval_utils.evaluate_response(
        llm=llm,
        question=question,
        answer=answer,
        context=context,
        retrieved_docs=retrieved_docs,
        ref_answer=ref_answer,
        ref_doc=ref_doc,
    )

    assert results == {
        "correctness": 10,
        "retrieval": 9,
        "groundedness": 8,
        "document_match": True,
    }


def test_evaluate_response_without_retrieval(monkeypatch):
    llm = object()

    monkeypatch.setattr(eval_utils, "score_correctness", lambda **kwargs: 7)

    def should_not_be_called(**kwargs):
        raise AssertionError("Retrieval and groundedness scoring should be skipped.")

    monkeypatch.setattr(eval_utils, "score_retrieval", should_not_be_called)
    monkeypatch.setattr(eval_utils, "score_groundedness", should_not_be_called)
    monkeypatch.setattr(eval_utils, "test_doc_match", should_not_be_called)

    results = eval_utils.evaluate_response(
        llm=llm,
        question="Question",
        answer="Answer",
        context=[],
        retrieved_docs=[],
        ref_answer="Reference answer",
        ref_doc="reference_doc.txt",
    )

    assert results == {
        "correctness": 7,
        "retrieval": 0,
        "groundedness": 0,
        "document_match": False,
    }
