"""Training pipeline: metrics, leakage prevention, metadata, threshold tuning, dataset checker."""
import json
import random

import pytest

from text import config
from text.data import check_dataset
from text.data import generate_sample_dataset as gen
from text.train import (DEFAULT_THRESHOLD, THRESHOLD_GRID, _is_real, group_near_duplicates, grouped_split,
                        pick_threshold, threshold_table, train)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("trainingdata")
    gen.generate(d / "real", gen.CATEGORIES[:4], per_class=30, seed=11)
    gen.generate(d / "ood", gen.OOD_CATEGORIES, per_class=12, seed=12)
    return d


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    meta = train(data / "real", out, cv=3, ood_dir=data / "ood", data_source="synthetic-test")
    return meta, out


# ---------------------------------------------------------------- metrics
def test_all_required_metrics_are_recorded(trained):
    m = trained[0]["metrics"]
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "cv_f1_macro_mean"):
        assert 0.0 <= m[k] <= 1.0, k
    assert set(m["per_class"]) == set(gen.CATEGORIES[:4])
    for r in m["per_class"].values():
        assert set(r) == {"precision", "recall", "f1-score", "support"}
    cm = m["confusion_matrix"]
    assert cm["labels"] == sorted(gen.CATEGORIES[:4])
    assert sum(map(sum, cm["matrix"])) == m["n"]                      # every test doc is in the matrix
    assert "validation" in m and m["validation"]["n"] > 0


# ---------------------------------------------------------------- metadata
def test_metadata_records_dataset_features_and_libraries(trained):
    meta, out = trained
    saved = json.loads((out / "model_meta.json").read_text())
    assert saved["data_source"] == "synthetic-test"
    ds = saved["dataset"]
    assert ds["n_documents"] == 120 and sum(ds["per_category"].values()) == 120
    assert set(ds["split"]) >= {"train", "validation", "test", "seed"}
    assert ds["out_of_domain"]["n_validation"] + ds["out_of_domain"]["n_test"] == 24
    f = saved["features"]
    assert f["tfidf"]["ngram_range"] == [1, 2] and f["tfidf"]["vocabulary_size"] > 100
    assert f["classifier"]["type"].startswith("LinearSVC")             # still a Linear SVM
    libs = saved["libraries"]
    assert {"python", "scikit-learn", "numpy"} <= set(libs)
    assert saved["sklearn_version"] == libs["scikit-learn"]


def test_synthetic_data_is_flagged_as_demo(trained):
    meta = trained[0]
    assert meta["real_data_validated"] is False
    assert "DEMO MODEL" in meta["evaluation_note"]


@pytest.mark.parametrize("source,expected", [
    ("synthetic-demo", False), ("synthetic", False), ("my-demo-set", False), ("unspecified", False), ("", False),
    ("drdo-real", True), ("internal-corpus-v1", True),
])
def test_real_data_flag(source, expected):
    assert _is_real(source) is expected


# ---------------------------------------------------------------- leakage
def test_near_duplicates_are_grouped():
    base = "the quarterly invoice total amount due payment terms net thirty days " * 4
    docs = [base, base + " extra", "completely different words about turbine maintenance schedules and lubrication " * 4]
    g = group_near_duplicates(docs, 0.9)
    assert g[0] == g[1] and g[0] != g[2]


def test_groups_never_cross_splits(tmp_path):
    """Versions of the same document must land on the same side of every split."""
    import numpy as np
    rng = random.Random(3)
    base_docs = {c: [gen.make_document(c, rng) for _ in range(15)] for c in gen.CATEGORIES[:3]}
    for cat, docs in base_docs.items():                         # each document saved with 2 near-identical versions
        (tmp_path / cat).mkdir()
        for i, d in enumerate(docs):
            (tmp_path / cat / f"{i}_v1.txt").write_text(d)
            (tmp_path / cat / f"{i}_v2.txt").write_text(d + " Revised.")
    from text.train import load_folder
    texts, labels, _ = load_folder(tmp_path)
    groups = group_near_duplicates(texts, 0.9)
    labels_arr = np.array(labels)
    rest, test = grouped_split(labels_arr, groups, 0.2, seed=1)
    assert not (set(groups[rest]) & set(groups[test]))
    # and the full trainer (which asserts this internally) completes
    train(tmp_path, tmp_path / "out", cv=2, data_source="synthetic-test")


def test_exact_duplicates_dropped_and_conflicting_labels_reported(tmp_path, capsys):
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
    same = "an identical paragraph that appears in two different category folders by mistake " * 2
    (tmp_path / "a" / "1.txt").write_text(same)
    (tmp_path / "b" / "1.txt").write_text(same)
    from text.train import load_folder
    texts, labels, stats = load_folder(tmp_path)
    assert len(texts) == 1 and stats["duplicates"] == 1 and stats["conflicting_labels"] == 1
    assert "WARNING" in capsys.readouterr().err


def test_too_few_documents_refuses_to_train(tmp_path):
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
        for i in range(5):
            (tmp_path / c / f"{i}.txt").write_text(f"document number {i} about topic {c} " * 6 + c * i)
    with pytest.raises(SystemExit, match="at least"):
        train(tmp_path, tmp_path / "out", cv=2)


# ---------------------------------------------------------------- unknown threshold
def test_threshold_needs_ood_data(data, tmp_path):
    with pytest.raises(SystemExit, match="--ood-dir"):
        train(data / "real", tmp_path / "o", tune_threshold=True)


def test_threshold_not_tuned_unless_asked(trained):
    assert trained[0]["unknown_threshold"] == DEFAULT_THRESHOLD and trained[0]["unknown_threshold_tuned"] is False


def test_tuning_uses_validation_and_reports_test(data, tmp_path):
    meta = train(data / "real", tmp_path / "t", cv=3, ood_dir=data / "ood", data_source="synthetic-test", tune_threshold=True)
    assert meta["unknown_threshold_tuned"] is True
    assert meta["unknown_threshold"] in THRESHOLD_GRID
    assert meta["unknown_threshold"] == pick_threshold(meta["threshold_table_validation"])   # chosen on validation only
    assert meta["threshold_table_test"] != meta["threshold_table_validation"]                # reported on different documents


def test_threshold_table_math():
    class P:  # stand-in model: confidence is encoded in the text
        classes_ = ["x", "y"]

        def predict_proba(self, texts):
            import numpy as np
            return np.array([[float(t), 1 - float(t)] for t in texts])
    # in-domain: confident+correct (0.9), confident+correct (0.8), unsure (0.55); OOD: 0.6 and 0.95 (dangerous)
    rows = threshold_table(P(), ["0.9", "0.8", "0.55"], ["x", "x", "x"], ["0.6", "0.95"], grid=[0.5, 0.7, 0.99])
    r50, r70, r99 = rows
    assert r50["coverage"] == 1.0 and r50["unknown_recall"] == 0.0           # nothing rejected at 0.5
    assert r70["coverage"] == pytest.approx(2 / 3, abs=1e-3)
    assert r70["unknown_recall"] == 0.5                                       # 0.6 rejected, 0.95 NOT rejected
    assert r70["unknown_precision"] == pytest.approx(0.5, abs=1e-3)           # 1 correct rejection vs 1 real doc rejected
    assert r99["coverage"] == 0.0 and r99["unknown_recall"] == 1.0 and r99["selective_accuracy"] is None


def test_confidence_rejection_can_miss_confident_ood_documents():
    """The documented limitation: a high-confidence unfamiliar document is NOT rejected."""
    class P:
        classes_ = ["x", "y"]

        def predict_proba(self, texts):
            import numpy as np
            return np.array([[0.97, 0.03] for _ in texts])
    row = threshold_table(P(), ["a"], ["x"], ["unfamiliar"], grid=[0.9])[0]
    assert row["unknown_recall"] == 0.0


# ---------------------------------------------------------------- dataset checker
def test_checker_accepts_good_dataset(data, capsys):
    assert check_dataset.check(data / "real") == 0
    assert "READY to train" in capsys.readouterr().out


def test_checker_rejects_small_and_missing(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for i in range(3):
        (tmp_path / "a" / f"{i}.txt").write_text("some readable document text for the small category " * 3 + str(i))
        (tmp_path / "b" / f"{i}.txt").write_text("other readable words for the second category here " * 3 + str(i))
    assert check_dataset.check(tmp_path) == 1
    assert "PROBLEM" in capsys.readouterr().out
    assert check_dataset.check(tmp_path / "nope") == 1


# ---------------------------------------------------------------- calibration
class _FixedModel:
    """Stand-in model that gives every document the same probability vector."""
    classes_ = ["x", "y"]

    def __init__(self, p_x):
        self.p_x = p_x

    def predict_proba(self, texts):
        import numpy as np
        return np.array([[self.p_x, 1 - self.p_x]] * len(texts))


def test_calibration_is_zero_when_confidence_equals_hit_rate():
    from text.train import calibration_report
    # every document scored 0.8 for 'x'; truth is 'x' for 8 of 10 -> accuracy 0.8 == confidence 0.8
    r = calibration_report(_FixedModel(0.8), ["t"] * 10, ["x"] * 8 + ["y"] * 2)
    assert r["accuracy"] == 0.8 and r["mean_confidence"] == 0.8 and r["ece"] == 0.0


def test_calibration_detects_overconfidence():
    from text.train import calibration_report
    # always 99 % sure of 'x' but right only half the time -> ECE about 0.49
    r = calibration_report(_FixedModel(0.99), ["t"] * 10, ["x"] * 5 + ["y"] * 5)
    assert r["ece"] == pytest.approx(0.49, abs=0.01) and r["brier"] > 0.5


def test_calibration_is_recorded_in_metadata(trained):
    c = trained[0]["metrics"]["calibration"]
    for split in ("test", "validation"):
        assert 0.0 <= c[split]["ece"] <= 1.0 and 0.0 <= c[split]["brier"] <= 2.0 and c[split]["n"] > 0
        assert sum(b["n"] for b in c[split]["bins"]) == c[split]["n"]            # every document falls in exactly one bin
