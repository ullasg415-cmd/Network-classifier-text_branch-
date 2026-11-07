"""Check a labelled document folder BEFORE training.

    python -m text.data.check_dataset datasets/real

Reports per-category counts and formats, unreadable files, too-short documents, exact duplicates
(including identical text filed under two labels), near-duplicate groups, and class imbalance.
Exit code 0 = ready to train, 1 = problems that stop training.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ..extract import ALLOWED_EXTENSIONS
from ..train import MIN_DOCS_PER_CATEGORY, group_near_duplicates, load_folder


def check(data_dir: Path, near_dup_threshold: float = 0.9) -> int:
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a folder")
        return 1
    fmt = Counter()
    skipped_ext = Counter()
    for f in data_dir.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                fmt[(f.parent.name, ext)] += 1
            elif f.name not in ("README.md", ".gitkeep", ".gitignore") and ext:
                skipped_ext[ext] += 1
    try:
        texts, labels, stats = load_folder(data_dir)
    except SystemExit as e:
        print(f"ERROR: {e}")
        return 1

    problems, warnings = [], []
    counts = Counter(labels)
    print(f"Folder: {data_dir}")
    print(f"Files read: {stats['files']}  usable: {len(texts)}  unreadable: {stats['failed']}  "
          f"too short: {stats['too_short']}  exact duplicates dropped: {stats['duplicates']}")
    print(f"\n{'category':24s}{'usable':>8s}   formats")
    for cat in sorted(counts):
        formats = ", ".join(f"{ext[1:]}:{n}" for (c, ext), n in sorted(fmt.items()) if c == cat)
        print(f"{cat:24s}{counts[cat]:>8d}   {formats}")

    if len(counts) < 2:
        problems.append("need at least 2 categories")
    for cat, n in counts.items():
        if n < MIN_DOCS_PER_CATEGORY:
            problems.append(f"'{cat}' has {n} usable documents; minimum is {MIN_DOCS_PER_CATEGORY}")
        elif n < 50:
            warnings.append(f"'{cat}' has only {n} documents; 50+ gives more trustworthy scores")
    if counts and max(counts.values()) > 5 * min(counts.values()):
        warnings.append("class imbalance above 5:1; the smallest categories will have noisy scores")
    if stats["conflicting_labels"]:
        problems.append(f"{stats['conflicting_labels']} identical document(s) filed under two different categories")
    if stats["failed"]:
        warnings.append(f"{stats['failed']} file(s) could not be read (corrupt / encrypted / binary)")
    if stats["too_short"]:
        warnings.append(f"{stats['too_short']} file(s) had almost no text (scanned PDF? needs OCR)")
    if skipped_ext:
        warnings.append("ignored unsupported file types: " + ", ".join(f"{e} x{n}" for e, n in skipped_ext.items()))

    if texts and not problems:
        groups = group_near_duplicates(texts, near_dup_threshold)
        n_groups = int(groups.max()) + 1
        merged = len(texts) - n_groups
        print(f"\nNear-duplicate groups (cosine >= {near_dup_threshold}): {n_groups} groups from {len(texts)} documents")
        if merged > 0.3 * len(texts):
            warnings.append(f"{merged} documents are near-duplicates of others; the effective dataset is smaller than it looks")

    for w in warnings:
        print(f"WARNING: {w}")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("\nRESULT:", "READY to train" if not problems else "NOT ready")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--near-dup-threshold", type=float, default=0.9)
    a = ap.parse_args()
    sys.exit(check(a.data_dir, a.near_dup_threshold))


if __name__ == "__main__":
    main()
