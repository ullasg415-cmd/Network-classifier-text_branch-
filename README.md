# Text branch (Person 2)

PDF / DOCX / TXT → **Apache Tika** (extract text) → text cleaning → **TF-IDF** → **Linear SVM** → category.
Lives in `backend/text/` and exposes the agreed contract:

```python
from text import classify_text

classify_text(file_bytes: bytes, filename: str) -> {"category": "invoice", "confidence": 0.91}
```

`category` is a trained category or `"unknown"`. `confidence` is 0–1. Errors raise `text.TextExtractionError`
(corrupt / encrypted / binary) or `text.UnsupportedFileType`.

---

## Status — read this first

| Part | State |
|---|---|
| Extraction: TXT / PDF (pypdf) / DOCX (python-docx), type sniffing, encrypted + corrupt + empty + unsupported files, large-file timing | **Tested** (local tier) |
| Tika client code | **Tested against fake servers only** (success, HTTP 204/415/422/500, timeout, garbled or truncated reply, dropped connection, invalid `TIKA_URL`, malformed documents; auto mode falls back to local extractors every time). **Never run against a real Tika server** |
| Training: grouped near-duplicate split, metrics (accuracy, precision, recall, macro/weighted F1, confusion matrix), metadata, dataset checker | **Tested** (local tier) |
| "Unknown" rejection, validation-based threshold tuning, precision/recall trade-off table | **Tested** (local tier) on synthetic data |
| MLflow registry logic (`registry.py`): promotion gate, champion alias, rollback + reload, fallback | **Tested against an in-memory FAKE MLflow.** Never run against real MLflow |
| FastAPI router (`api.py`) | **Never served by real FastAPI.** Imported and called with stand-in classes (with warnings as errors); `tests/test_api.py` (12 tests) is written and runs once `fastapi` + `httpx` are installed |
| `docker-compose.tika.yml` + `docker-compose.text-backend.yml` | **Config validated** with `docker compose config` (valid standalone; merges correctly with a stand-in main file). **Containers never started** (no Docker daemon was available) |
| Readable HTTP payloads (Person 1) | **Tested**: HTTP messages with text bodies are classified; TLS records and gzip bytes are refused, never decoded |
| Confidence calibration | **Measured** (ECE / Brier, reliability bins, in `model_meta.json`) on synthetic data only: the demo model is *under*-confident (mean confidence 0.92, accuracy 1.00). Real-data calibration unknown |
| Shipped model in `models/` | **DEMO MODEL**, trained on synthetic template documents. Its 1.0 scores say nothing about real accuracy |
| Validation on real documents | **PENDING.** No real labelled dataset was available, and none was invented |

Exact results and what each skipped test needs: **[TEST_RESULTS.md](TEST_RESULTS.md)**.

First thing to do on your machine: follow **[MAC_VALIDATION.md](MAC_VALIDATION.md)** (exact commands, in order).
Anything that fails in the Tika / MLflow / FastAPI / Docker tiers is a real finding, not something to hide.

---

## Folder layout

```
backend/text/
├── __init__.py            classify_text() / classify_text_detailed()           <- the contract
├── extract.py             bytes -> text (Tika, or local pypdf / python-docx / plain text), Tika health check
├── model.py               TF-IDF + calibrated LinearSVC, model loading/caching, library-version check
├── train.py               python -m text.train ...
├── registry.py            MLflow: log, register, champion alias, load
├── service.py             upload validation, error codes, health report (no FastAPI needed)
├── api.py                 FastAPI router
├── config.py              every setting is an environment variable
├── models/                text_svm.joblib + model_meta.json   (DEMO model)
├── datasets/              where real documents go (git-ignored) + instructions
├── data/                  generate_sample_dataset.py (synthetic demo data), check_dataset.py
├── samples/               six sample files to try
├── docker-compose.tika.yml          Tika only; valid on its own
├── docker-compose.text-backend.yml  override to merge into Person 3's compose (backend -> Tika / MLflow)
├── requirements.txt                 pins scikit-learn only; works on Python 3.11+
├── requirements-lock-py313.txt      exact versions the tests were run with (Python 3.13)
├── pytest.ini
├── tests/                 143 local tests + integration tiers (tika / mlflow / docker / fastapi)
├── README.md              (this file)
├── INTEGRATION_GUIDE.md   Person 3: imports, router, start, curl, errors, MLflow, compose
├── MAC_VALIDATION.md      exact commands to validate everything on a MacBook
└── TEST_RESULTS.md        what was run, results, what is still unverified
```

---

## API integration guide (Person 3)

Full step-by-step version (imports, start-up, curl, errors, MLflow, compose): **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)**. Summary:

```python
from text.api import router as text_router
app.include_router(text_router)
```

Or call the function directly: `from text import classify_text`.

| Route | Purpose | Status codes |
|---|---|---|
| `POST /classify/text` (multipart field `file`) | classify one document | 200, 400 empty, 413 too large, 415 unsupported type, 422 corrupt / encrypted / binary, 503 model unavailable |
| `GET /classify/text/model` | live version, categories, threshold, data source, `real_data_validated`, `evaluation_note`, metrics (registry view) | 200, 503 |
| `POST /classify/text/model/reload` | reload after moving the `champion` alias (promote / rollback button) | 200, 503 |
| `GET /classify/text/health` | readiness: model loaded? Tika reachable? Use as the container health check | 200 (`ok`/`degraded`), 503 (`unavailable`) |

`POST /classify/text` response:

```json
{"filename": "inv.pdf", "category": "invoice", "confidence": 0.9262, "source": "svm",
 "best_guess": "invoice", "top_k": [{"category": "invoice", "confidence": 0.9262}],
 "reason": "ok", "extractor": "local-pdf", "file_type": "pdf", "text_chars": 612, "model_version": "local-fc1d4ff3"}
```

* `reason`: `ok`, `low_confidence` (category forced to `unknown`, `best_guess` shows what the model would have said) or `too_little_text` (e.g. a scanned PDF with no text layer).
* **Show a "DEMO MODEL" badge** whenever `GET /classify/text/model` returns `real_data_validated: false`.
* Health: `ok` = all good; `degraded` = Tika is configured but down, local extractors are serving PDF/DOCX/TXT; `unavailable` = cannot classify.

### Conventions shared with the network branch (Sumukh)

| | Network | Text |
|---|---|---|
| Function | `classify_network(flow)` | `classify_text(file_bytes, filename)` |
| Result key | `label` | `category` |
| `source` | `rule` / `ml` | `svm` |
| "Can't tell" | `Unknown` | `unknown` (lowercase; the UI should map both) |
| Confidence | 0–1 | 0–1 (calibrated) |
| MLflow registered name | his own | `text-classifier` |
| Alias | `champion` | `champion` |

Readable HTTP bodies from his captures can go in directly: `classify_text(body_bytes, "payload")`.
Encrypted or binary bytes raise `TextExtractionError` instead of getting a made-up label.

---

## MLflow integration guide

The text branch uses the **same MLflow server** as everyone else; only the registered-model name differs.

```bash
# Person 3's shared server, e.g.:
mlflow server --host 0.0.0.0 --port 5000
export MLFLOW_TRACKING_URI=http://<server>:5000

python -m text.train --data-dir text/datasets/real --register          # promote if macro-F1 >= current champion
python -m text.train --data-dir text/datasets/real --register force    # always promote
python -m text.train --data-dir text/datasets/real --register never    # register only
```

* Each run logs parameters, metrics (`f1_macro`, `accuracy`, `cv_f1_macro_mean`), `model_meta.json` and the model; each retrain is a new version of `text-classifier`.
* **Promote** = move alias `champion` to a version (the script does it under `auto` / `force`, or use the MLflow UI, or `MlflowClient().set_registered_model_alias("text-classifier", "champion", "2")`).
* **Roll back** = move the alias back, then `POST /classify/text/model/reload`. The loaded model is cached, so the backend keeps serving the old version until reload (or restart).
* **Compare versions**: `MlflowClient().search_model_versions("name='text-classifier'")` then each version's run metrics (`f1_macro` etc.).
* The backend loads `models:/text-classifier@champion` when `MLFLOW_TRACKING_URI` is set. If the registry is unreachable or has no champion and `TEXT_MODEL_SOURCE=auto`, it falls back to `models/text_svm.joblib` and logs that it did. `TEXT_MODEL_SOURCE=registry` makes that a hard error instead.
* `registry.py` handles both MLflow 3.x (`log_model(name=...)`) and 2.x (`artifact_path=`). **Not yet verified against real MLflow** — run `pytest text/tests/test_integration_mlflow.py -v` (uses a temporary SQLite store, no server needed).

---

## Tika: configure, start, stop

```bash
docker run -d --name tika -p 9998:9998 apache/tika:latest        # start   (or: docker compose -f text/docker-compose.tika.yml up -d tika)
curl http://localhost:9998/tika                                   # health: a greeting means it is up
export TIKA_URL=http://localhost:9998
docker rm -f tika                                                 # stop and remove
```

Without Docker: `java -jar tika-server-standard-<version>.jar -h 0.0.0.0` (jar from tika.apache.org; from Tika's docs, not run here).
Pin the image/jar version before the final demo. With `TEXT_EXTRACTOR=auto` (default) the backend falls back to local extractors if Tika is down; only `.doc / .rtf / .odt / .html` need Tika.
**None of the Tika / Docker commands above were run in the environment this was built in.**

---

## Environment variables (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `TIKA_URL` | *(empty)* | e.g. `http://tika:9998`. Empty = local libraries only |
| `TEXT_EXTRACTOR` | `auto` | `auto` (Tika, fall back to local), `tika` (Tika only), `local` |
| `TIKA_TIMEOUT` | 30 | seconds |
| `MLFLOW_TRACKING_URI` | *(unset)* | set → load the `champion` model from the registry |
| `TEXT_MODEL_SOURCE` | `auto` | `auto` (registry, fall back to file), `registry`, `local` |
| `TEXT_MODEL_NAME` / `TEXT_MODEL_ALIAS` | `text-classifier` / `champion` | |
| `TEXT_MLFLOW_EXPERIMENT` | `text-classifier` | |
| `TEXT_UNKNOWN_THRESHOLD` | from model metadata | confidence below this → `unknown`; `0` turns rejection off |
| `TEXT_MIN_CHARS` | 30 | less extracted text than this → `unknown` (`too_little_text`) |
| `TEXT_MAX_BYTES` | 20 MB | upload limit |
| `TEXT_MODEL_DIR` | `text/models` | where the joblib + metadata live |
| `TEXT_STRICT_VERSIONS` | off | `1` = refuse to load a model trained with a different scikit-learn |

---

## Training instructions

```bash
cd backend
pip install -r text/requirements.txt
pytest text/tests -v -rs                                              # everything local should pass

# demo data (synthetic) -> demo model
python -m text.data.generate_sample_dataset --out /tmp/sample --per-class 60 --seed 7
python -m text.data.generate_sample_dataset --out /tmp/ood --ood --per-class 30 --seed 99
python -m text.data.check_dataset /tmp/sample
python -m text.train --data-dir /tmp/sample --ood-dir /tmp/ood --data-source synthetic-demo

python -m text <file.pdf> ...                                         # classify files from the command line
```

What the trainer does: extracts text with the same code used at inference · drops exact duplicates (and reports identical
text filed under two labels) · groups near-duplicates (TF-IDF cosine ≥ 0.9) so versions of one document never straddle
train / validation / test · makes a stratified, group-aware 60 / 20 / 20 split · reports accuracy, macro precision / recall / F1,
weighted F1, per-category scores and a confusion matrix on the **test** split · records dataset counts, split sizes, the
TF-IDF / SVM configuration and library versions in `model_meta.json`.
Without real data the model is flagged `real_data_validated: false` and its `evaluation_note` says DEMO MODEL.

## Dataset preparation (real documents)

Full rules in `datasets/README.md`. Short version:

1. One folder per category in `text/datasets/real/<category>/` (`.pdf .docx .txt`; 50+ per category is a sensible start, 12 is the hard minimum). Do not pre-split.
2. Put documents from **other** kinds of document (not your categories) in `text/datasets/ood/<kind>/` — at least 2 per kind, 20+ in total.
3. `python -m text.data.check_dataset text/datasets/real`, then
   `python -m text.train --data-dir text/datasets/real --ood-dir text/datasets/ood --tune-threshold --data-source drdo-real`.
4. Use confidential or classified documents **only with explicit authorisation**. The `datasets/` folder is git-ignored.

Expect real scores far below 1.0. Report them, including the weak categories.

---

## Unknown-category detection — and why it is not a guarantee

The classifier returns `unknown` when the (calibrated) confidence is below a threshold or there is almost no text.
Training prints a precision/recall trade-off table: *coverage* (real documents kept), *selective accuracy*, and
*unknown precision / recall / F1* (unfamiliar documents rejected). The threshold is chosen on a **validation** split plus
half of the out-of-domain set and reported on the untouched **test** half; it is only tuned when `--tune-threshold` is given,
otherwise it is an untuned 0.5.

Limitations, stated plainly:
* Confidence measures similarity to the **trained categories**, not "have I seen this kind of document". An unfamiliar document that happens to resemble one category can score high and be accepted. The test `test_confidence_rejection_can_miss_confident_ood_documents` demonstrates this.
* A threshold is only as good as the out-of-domain sample it was tuned on; kinds of document not in that sample may behave differently.
* Calibration is estimated from a small training set and can be off.
* Sumukh's network results show the same failure (confident answers on traffic from an unseen generator). Treat `unknown` as a safety net, not as detection with known accuracy.
* The numbers printed on synthetic data (e.g. ~83–97 % of out-of-domain documents rejected at thresholds 0.5–0.8) only show the mechanism works; repeat on real documents before quoting any figure.

---

## Design decisions

* **Tika over HTTP, not the `tika` Python package:** that package downloads a jar at import time and needs Java in the app. `PUT /tika` needs only the standard library, and Tika runs in its own container.
* **Local fallback:** Tika is another service that can be down; PDF / DOCX / TXT still work without it.
* **Calibrated probabilities:** a plain Linear SVM gives a distance, not a probability. `CalibratedClassifierCV` makes a 0–1 confidence that the contract returns and the unknown rule thresholds. It is still a Linear SVM.
* **Same extraction in training and inference:** one function, so the model never sees text extracted differently from training.
* **Pre-trained model question:** this branch trains its own Linear SVM, which may not satisfy "use pre-trained models". If Prinitha requires it, the swap point is `model.build_pipeline()` + the registry (a DistilBERT / LayoutLM checkpoint registered under the same name); `classify_text` does not change. Nothing was silently swapped.

## Known limitations

* **Scanned PDFs** (images, no text layer) give no text → `unknown` (`too_little_text`). OCR is not set up.
* English-only stop words and vocabulary. One label per document.
* HTTPS payloads are encrypted; only readable payloads can be classified.
* **Python:** the shipped model needs scikit-learn 1.9.1, i.e. Python **3.11 or newer**. The test suite was run only on 3.13. On 3.11 the source compiles and the stdlib-only parts (Tika client, config, text extraction of plain files) were smoke-tested; the ML packages could not be installed there, so the full suite has **not** been run on 3.11 or 3.12.
* Model accuracy on real documents is unknown until real labelled data is supplied.
