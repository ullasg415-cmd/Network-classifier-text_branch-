"""Quick command line check:  python -m text path/to/file.pdf [more files...]"""
import json
import sys
from pathlib import Path

from . import classify_text_detailed


def main(paths):
    if not paths:
        print(__doc__)
        return 1
    for p in map(Path, paths):
        try:
            r = classify_text_detailed(p.read_bytes(), p.name)
            print(f"{p.name}: {r['category']} ({r['confidence']:.2f})  " + json.dumps({k: r[k] for k in ("reason", "extractor", "top_k")}))
        except Exception as e:  # CLI: show the error, keep going
            print(f"{p.name}: ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
