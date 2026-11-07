"""The interface contract with the rest of the system. If these fail, integration breaks."""
import inspect

import pytest

import text
from text import classify_text, classify_text_detailed
from text.data import generate_sample_dataset as gen


def test_signature_matches_agreed_contract():
    params = list(inspect.signature(classify_text).parameters)
    assert params == ["file_bytes", "filename"]


def test_returns_exactly_category_and_confidence(fresh_docs):
    out = classify_text(fresh_docs["invoice"][0].encode(), "a.txt")
    assert set(out) == {"category", "confidence"}
    assert isinstance(out["category"], str)
    assert isinstance(out["confidence"], float) and 0.0 <= out["confidence"] <= 1.0


def test_output_is_json_serialisable(fresh_docs):
    import json
    json.dumps(classify_text(fresh_docs["resume"][0].encode(), "a.txt"))
    json.dumps(classify_text_detailed(fresh_docs["resume"][0].encode(), "a.txt"))


@pytest.mark.parametrize("category", gen.CATEGORIES)
def test_unseen_documents_get_their_category(fresh_docs, category):
    # NOTE: synthetic data, so this checks the plumbing, not real-world accuracy.
    hits = sum(classify_text(d.encode(), "x.txt")["category"] == category for d in fresh_docs[category])
    assert hits >= 4, f"{category}: only {hits}/5 correct"


def test_pdf_and_docx_inputs_work_end_to_end(fresh_docs, make_pdf, make_docx):
    doc = fresh_docs["legal_contract"][0]
    assert classify_text(make_docx(doc), "c.docx")["category"] == "legal_contract"
    assert classify_text(make_pdf(doc[:1500]), "c.pdf")["category"] == "legal_contract"


def test_too_little_text_is_unknown_with_zero_confidence():
    out = classify_text_detailed(b"hi", "tiny.txt")
    assert out["category"] == "unknown" and out["confidence"] == 0.0 and out["reason"] == "too_little_text"


def test_unfamiliar_document_is_not_forced_into_a_category(monkeypatch):
    """The text branch should be able to say 'unknown' (the Known/Unknown design)."""
    import random
    monkeypatch.setenv("TEXT_UNKNOWN_THRESHOLD", "0.8")
    rng = random.Random(5)
    rejected = sum(classify_text(gen.make_document("recipe", rng).encode(), "r.txt")["category"] == "unknown" for _ in range(20))
    assert rejected >= 14  # measured ~95% at 0.8 on synthetic data; loose bound on purpose


def test_threshold_zero_switches_rejection_off(monkeypatch):
    monkeypatch.setenv("TEXT_UNKNOWN_THRESHOLD", "0")
    out = classify_text_detailed(b"a completely unrelated paragraph about gardening tomatoes and soil in spring weather", "g.txt")
    assert out["category"] != "unknown" and out["category"] == out["best_guess"]


def test_detailed_has_fields_the_api_returns(fresh_docs):
    d = classify_text_detailed(fresh_docs["invoice"][0].encode(), "a.txt")
    for k in ("category", "confidence", "best_guess", "top_k", "reason", "source", "extractor", "file_type", "text_chars", "model_version"):
        assert k in d
    assert d["source"] == "svm" and len(d["top_k"]) == 3


def test_model_info_reports_data_source_honestly():
    info = text.model_info()
    assert info["source"] == "local" and "invoice" in info["categories"]
    assert info["data_source"]  # recorded at training time ('synthetic-demo' for the shipped model)


def test_missing_model_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="text.train"):
        classify_text(b"some reasonably long document text for classification purposes", "a.txt")
