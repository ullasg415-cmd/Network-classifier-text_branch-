"""Train the text classifier: documents -> Tika/local extraction -> TF-IDF -> Linear SVM.

    python -m text.train --data-dir data/real --data-source drdo-real
    python -m text.train --data-dir data/real --ood-dir data/ood --tune-threshold --data-source drdo-real
    python -m text.train --data-dir data/real --register          # also log + register in MLflow

Data layout: one folder per category, any mix of .pdf / .docx / .txt inside.
Extraction goes through text.extract, the same code used at inference time.

How leakage is prevented
  1. exact duplicate documents are dropped (one copy kept);
  2. near-duplicates (TF-IDF cosine similarity >= --near-dup-threshold) are put in the same
     *group*, and a group is never split across train / validation / test;
  3. splits are stratified by category;
  4. the unknown-threshold is chosen on the validation split + validation half of the
     out-of-domain documents, and reported on the untouched test split + test half.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from . import config
from .extract import ALLOWED_EXTENSIONS, TextExtractionError, extract_text
from .model import build_pipeline

THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 ... 0.95
DEFAULT_THRESHOLD = 0.5   # used when no out-of-domain data is available to tune against
MIN_DOCS_PER_CATEGORY = 12


# --------------------------------------------------------------------------- data
def load_folder(data_dir: Path, min_chars: int | None = None):
    """Read folder-per-category data. Returns (texts, labels, stats)."""
    min_chars = config.min_chars() if min_chars is None else min_chars
    texts, labels, seen = [], [], {}
    stats = {"files": 0, "failed": 0, "too_short": 0, "duplicates": 0, "conflicting_labels": 0}
    cats = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not cats:
        raise SystemExit(f"{data_dir} has no category sub-folders")
    for cat_dir in cats:
        for f in sorted(cat_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            stats["files"] += 1
            try:
                text = extract_text(f.read_bytes(), f.name).text
            except TextExtractionError as e:
                stats["failed"] += 1
                print(f"  skip {f}: {e}", file=sys.stderr)
                continue
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen:  # identical documents in train and test would inflate scores
                stats["duplicates"] += 1
                if seen[h] != cat_dir.name:
                    stats["conflicting_labels"] += 1
                    print(f"  WARNING: identical document filed under '{seen[h]}' and '{cat_dir.name}': {f}", file=sys.stderr)
                continue
            seen[h] = cat_dir.name
            texts.append(text)
            labels.append(cat_dir.name)
    return texts, labels, stats


def group_near_duplicates(texts, threshold: float = 0.9):
    """Return an int group id per document; documents with cosine similarity >= threshold
    (TF-IDF, English stop-words removed) end up in the same group (connected components)."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    if n > 20_000:
        print(f"  note: near-duplicate grouping is O(n^2); {n} documents may take a while", file=sys.stderr)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)  # rows are L2-normalised
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    chunk = 256
    for start in range(0, n, chunk):
        sims = (X[start:start + chunk] @ X.T).toarray()
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i != j:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def grouped_split(labels, groups, fraction: float, seed: int):
    """Stratified, group-aware split. Returns (rest_idx, held_idx); held is ~fraction of the data."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1 / fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    return rest, held


# ---------------------------------------------------------------------- calibration
def calibration_report(pipe, texts, labels, bins: int = 10) -> dict:
    """How far the reported confidence is from the actual hit rate.

    Takes the model's top-class probability as 'confidence' and asks: of the documents scored ~0.9,
    did ~90 % turn out correct?
      ece   = expected calibration error (0 = perfectly calibrated; weighted mean |accuracy - confidence| per bin)
      brier = multi-class Brier score (0 = perfect; lower is better)
    With few documents (tens) these are noisy; with a near-perfect classifier ECE mostly measures
    under-confidence. Always read them together with `n`.
    """
    import numpy as np
    proba = pipe.predict_proba(texts)
    classes = np.array(pipe.classes_)
    conf = proba.max(axis=1)
    y = np.array(labels)
    correct = classes[proba.argmax(axis=1)] == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi if i == bins - 1 else conf < hi)
        if m.any():
            gap = abs(float(correct[m].mean()) - float(conf[m].mean()))
            ece += float(m.mean()) * gap
            rows.append({"range": [round(float(lo), 2), round(float(hi), 2)], "n": int(m.sum()),
                         "mean_confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(correct[m].mean()), 4)})
    onehot = (classes[None, :] == y[:, None]).astype(float)
    return {"n": int(len(y)), "mean_confidence": round(float(conf.mean()), 4), "accuracy": round(float(correct.mean()), 4),
            "ece": round(ece, 4), "brier": round(float(((proba - onehot) ** 2).sum(axis=1).mean()), 4), "bins": rows}


# ----------------------------------------------------------------- unknown threshold
def threshold_table(pipe, in_texts, in_labels, ood_texts, grid=THRESHOLD_GRID):
    """Precision/recall trade-off of the 'unknown' rule at each threshold.

    in-domain documents should be ACCEPTED (and correctly labelled); out-of-domain documents
    should be REJECTED. With 'unknown' as the positive class:
      unknown_recall    = fraction of out-of-domain documents rejected
      unknown_precision = rejected out-of-domain / all rejected
      coverage          = fraction of in-domain documents accepted
      selective_accuracy= accuracy on the in-domain documents that were accepted
    """
    import numpy as np
    pi = pipe.predict_proba(in_texts)
    in_conf = pi.max(axis=1)
    in_correct = np.array(pipe.classes_)[pi.argmax(axis=1)] == np.array(in_labels)
    ood_conf = pipe.predict_proba(ood_texts).max(axis=1) if len(ood_texts) else np.array([])
    rows = []
    for t in grid:
        accepted = in_conf >= t
        tp = int((ood_conf < t).sum())        # unfamiliar documents correctly rejected
        fp = int((~accepted).sum())           # real documents wrongly rejected
        fn = int((ood_conf >= t).sum())       # unfamiliar documents wrongly accepted
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "threshold": t,
            "coverage": round(float(accepted.mean()), 4),
            "selective_accuracy": round(float(in_correct[accepted].mean()), 4) if accepted.any() else None,
            "unknown_precision": round(prec, 4),
            "unknown_recall": round(rec, 4),
            "unknown_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        })
    return rows


def pick_threshold(table):
    """Balance between keeping real documents (coverage) and rejecting unfamiliar ones (recall)."""
    best = max(table, key=lambda r: (r["coverage"] + r["unknown_recall"], -r["threshold"]))
    return best["threshold"]


# ----------------------------------------------------------------------------- misc
def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("scikit-learn", "numpy", "scipy", "joblib", "pypdf", "python-docx", "mlflow"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def feature_config(pipe) -> dict:
    tf, svm = pipe.named_steps["tfidf"], pipe.named_steps["svm"]
    base = svm.estimator
    return {
        "tfidf": {"lowercase": tf.lowercase, "stop_words": tf.stop_words, "ngram_range": list(tf.ngram_range),
                  "sublinear_tf": tf.sublinear_tf, "min_df": tf.min_df, "max_features": tf.max_features,
                  "vocabulary_size": len(tf.vocabulary_)},
        "classifier": {"type": "LinearSVC (probability-calibrated)", "C": base.C, "class_weight": base.class_weight,
                       "calibration_cv": svm.cv, "calibration_method": svm.method},
    }


def _is_real(data_source: str) -> bool:
    s = (data_source or "").lower()
    return bool(s) and s != "unspecified" and not s.startswith("synthetic") and "demo" not in s


def _counts(labels):
    return dict(sorted(Counter(labels).items()))


# --------------------------------------------------------------------------- train
def train(data_dir: Path, out_dir: Path, test_size=0.2, val_size=0.2, seed=42, cv=5, ood_dir: Path | None = None,
          data_source="unspecified", register: str | None = None, tune_threshold: bool = False,
          near_dup_threshold: float = 0.9) -> dict:
    import joblib
    import numpy as np
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

    if tune_threshold and not ood_dir:
        raise SystemExit("--tune-threshold needs --ood-dir (documents from categories NOT in the training set)")

    texts, labels, stats = load_folder(data_dir)
    categories = sorted(set(labels))
    print(f"Loaded {len(texts)} documents in {len(categories)} categories {stats}")
    if len(categories) < 2:
        raise SystemExit("Need at least 2 categories to train")
    smallest = min(labels.count(c) for c in categories)
    if smallest < MIN_DOCS_PER_CATEGORY:
        raise SystemExit(f"A category has only {smallest} documents; need at least {MIN_DOCS_PER_CATEGORY} per category")

    groups = group_near_duplicates(texts, near_dup_threshold)
    n_groups = int(groups.max()) + 1
    print(f"Near-duplicate grouping (cosine >= {near_dup_threshold}): {len(texts)} documents -> {n_groups} groups")

    labels_arr = np.array(labels)
    rest_idx, test_idx = grouped_split(labels_arr, groups, test_size, seed)
    rel_val = val_size / (1 - test_size)
    sub_tr, sub_val = grouped_split(labels_arr[rest_idx], groups[rest_idx], rel_val, seed)
    train_idx, val_idx = rest_idx[sub_tr], rest_idx[sub_val]
    pick = lambda idx: ([texts[i] for i in idx], [labels[i] for i in idx])  # noqa: E731
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = pick(train_idx), pick(val_idx), pick(test_idx)
    # leakage guard: no group may appear in two splits
    g_tr, g_val, g_te = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    assert not (g_tr & g_val or g_tr & g_te or g_val & g_te), "near-duplicate group crosses splits"
    for name, y in (("train", y_tr), ("validation", y_val), ("test", y_te)):
        missing = set(categories) - set(y)
        if missing:
            raise SystemExit(f"{name} split has no documents for {sorted(missing)}; add more documents")
    print(f"Split (grouped, stratified): train={len(X_tr)} validation={len(X_val)} test={len(X_te)}")

    pipe = build_pipeline()
    folds = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, groups=groups[train_idx], cv=folds, scoring="f1_macro")
    pipe.fit(X_tr, y_tr)

    def evaluate(X, y):
        pred = pipe.predict(X)
        rep = classification_report(y, pred, labels=categories, output_dict=True, zero_division=0)
        return {
            "n": len(y),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision_macro": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
            "per_class": {c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score", "support")}
                          for c in categories},
            "confusion_matrix": {"labels": categories, "matrix": confusion_matrix(y, pred, labels=categories).tolist()},
        }

    test_m, val_m = evaluate(X_te, y_te), evaluate(X_val, y_val)
    metrics = {**test_m, "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
               "cv_f1_macro_std": round(float(cv_scores.std()), 4), "validation": val_m,
               "calibration": {"test": calibration_report(pipe, X_te, y_te),
                               "validation": calibration_report(pipe, X_val, y_val)}}

    # ---- unknown threshold: tuned on validation, reported on test --------------------
    threshold, tuned, tbl_val, tbl_test, ood_info = DEFAULT_THRESHOLD, False, None, None, None
    if ood_dir:
        ood_texts, ood_labels, ood_stats = load_folder(ood_dir)
        ood_groups = group_near_duplicates(ood_texts, near_dup_threshold)
        ood_labels_arr = np.array(ood_labels)
        if min(Counter(ood_labels).values()) < 2:
            raise SystemExit("Each out-of-domain category needs at least 2 documents (one half validates, one half tests)")
        v_idx, t_idx = grouped_split(ood_labels_arr, ood_groups, 0.5, seed)   # held half = test
        ood_val = [ood_texts[i] for i in v_idx]
        ood_test = [ood_texts[i] for i in t_idx]
        tbl_val = threshold_table(pipe, X_val, y_val, ood_val)
        tbl_test = threshold_table(pipe, X_te, y_te, ood_test)
        ood_info = {"categories": _counts(ood_labels), "n_validation": len(ood_val), "n_test": len(ood_test), "load_stats": ood_stats}
        if tune_threshold:
            threshold, tuned = pick_threshold(tbl_val), True

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "text_svm.joblib"
    joblib.dump(pipe, model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:8]
    libs = library_versions()
    real = _is_real(data_source)
    meta = {
        "name": config.registered_model_name(),
        "version": f"local-{sha}",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_source": data_source,
        "real_data_validated": real,
        "evaluation_note": ("Evaluated on documents declared as real data." if real else
                            "DEMO MODEL: trained on synthetic or unspecified data. Scores do NOT reflect real-world performance."),
        "categories": categories,
        "dataset": {
            "path": str(data_dir), "n_documents": len(texts), "per_category": _counts(labels),
            "load_stats": stats, "near_duplicate_threshold": near_dup_threshold, "n_groups": n_groups,
            "split": {"seed": seed, "test_size": test_size, "val_size": val_size, "cv": cv,
                      "train": {"n": len(X_tr), "per_category": _counts(y_tr)},
                      "validation": {"n": len(X_val), "per_category": _counts(y_val)},
                      "test": {"n": len(X_te), "per_category": _counts(y_te)}},
            "out_of_domain": ood_info,
        },
        "n_documents": len(texts), "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te),
        "features": feature_config(pipe),
        "libraries": libs,
        "sklearn_version": libs["scikit-learn"],
        "unknown_threshold": threshold, "unknown_threshold_tuned": tuned,
        "threshold_table_validation": tbl_val, "threshold_table_test": tbl_test,
        "metrics": metrics,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if register:
        from . import registry
        params = {"test_size": test_size, "val_size": val_size, "seed": seed, "cv": cv, "data_source": data_source,
                  "tfidf_ngram": "1-2", "svm_C": feature_config(pipe)["classifier"]["C"], "n_documents": len(texts),
                  "unknown_threshold": threshold}
        meta["registry"] = registry.log_and_register(pipe, meta, params, promote=register)
        (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _print_summary(meta)
    return meta


def _print_table(title, table):
    print(f"\n{title}")
    print("  thr   coverage  sel.acc  unk.prec  unk.recall  unk.F1")
    for r in table:
        sa = "  n/a " if r["selective_accuracy"] is None else f"{r['selective_accuracy']:.3f}"
        print(f"  {r['threshold']:.2f}  {r['coverage']:8.3f}  {sa}   {r['unknown_precision']:7.3f}   {r['unknown_recall']:8.3f}  {r['unknown_f1']:6.3f}")


def _print_summary(meta):
    m = meta["metrics"]
    print(f"\n=== {meta['evaluation_note']}")
    print(f"Test ({m['n']} docs): accuracy={m['accuracy']}  precision(macro)={m['precision_macro']}  recall(macro)={m['recall_macro']}  "
          f"F1(macro)={m['f1_macro']}  F1(weighted)={m['f1_weighted']}")
    print(f"Validation ({m['validation']['n']} docs): accuracy={m['validation']['accuracy']}  F1(macro)={m['validation']['f1_macro']}")
    print(f"5-fold grouped CV on train: macro-F1 {m['cv_f1_macro_mean']} +/- {m['cv_f1_macro_std']}")
    c = m["calibration"]["test"]
    print(f"Calibration (test, n={c['n']}): mean confidence {c['mean_confidence']} vs accuracy {c['accuracy']}  "
          f"ECE={c['ece']}  Brier={c['brier']}   (0 = perfect; noisy with few documents)")
    for c, r in m["per_class"].items():
        print(f"  {c:18s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f} n={int(r['support'])}")
    cm = m["confusion_matrix"]
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("  " + " ".join(f"{c[:8]:>8s}" for c in cm["labels"]))
    for lab, row in zip(cm["labels"], cm["matrix"]):
        print(f"  {lab[:12]:12s}" + " ".join(f"{v:>8d}" for v in row))
    if meta["threshold_table_validation"]:
        _print_table("Unknown-threshold trade-off on VALIDATION data (used to choose the threshold):", meta["threshold_table_validation"])
        _print_table("Same thresholds on held-out TEST data (what to report):", meta["threshold_table_test"])
    print(f"\nunknown_threshold = {meta['unknown_threshold']} "
          f"({'tuned on the validation split' if meta['unknown_threshold_tuned'] else 'UNTUNED default'})")
    print(f"Model: {meta['version']}  data: {meta['data_source']}  real_data_validated={meta['real_data_validated']}")
    if "registry" in meta:
        print(f"MLflow: {meta['registry']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: text/models")
    ap.add_argument("--ood-dir", type=Path, default=None, help="folder of documents from categories NOT in training")
    ap.add_argument("--tune-threshold", action="store_true",
                    help="choose unknown_threshold on the validation split (needs --ood-dir; use REAL out-of-domain documents)")
    ap.add_argument("--data-source", default="unspecified",
                    help="recorded in the metadata, e.g. 'drdo-real' or 'synthetic-demo'. Anything starting with 'synthetic' or containing 'demo' marks the model as a DEMO model")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--near-dup-threshold", type=float, default=0.9)
    ap.add_argument("--register", nargs="?", const="auto", choices=["auto", "force", "never"],
                    help="log to MLflow and register; value = promotion policy (default auto: promote if macro-F1 >= champion)")
    a = ap.parse_args()
    train(a.data_dir, a.out_dir or config.model_dir(), a.test_size, a.val_size, a.seed, a.cv, a.ood_dir,
          a.data_source, a.register, a.tune_threshold, a.near_dup_threshold)


if __name__ == "__main__":
    main()
