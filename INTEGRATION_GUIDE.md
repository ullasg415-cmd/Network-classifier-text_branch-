# Integration guide for Person 3 (FastAPI + MLflow + React)

Status of everything below: **written and checked locally; NOT yet run against a real FastAPI app, real MLflow or real Tika.**
`TEST_RESULTS.md` says exactly what was and was not run. If a step here fails on your machine, that is a finding: tell Person 2.

## 1. What you get

A folder `backend/text/` with one function and one FastAPI router:

```python
from text import classify_text
classify_text(file_bytes, filename)   # -> {"category": "invoice", "confidence": 0.91}   (or "unknown")

from text.api import router as text_router
app.include_router(text_router)       # adds the four routes below
```

## 2. Required dependencies

Python **3.11 or newer** (the shipped model needs scikit-learn 1.9.1, which needs 3.11+). Tested only on 3.13.

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r text/requirements.txt
```
Your own app additionally needs `fastapi`, `uvicorn`, `python-multipart` (already in that file). Do not change the scikit-learn version
(see "Model location and version rule" below).

## 3. Import path and router registration

The package is called `text`; it must be importable, so **run your app from `backend/`**:

```
backend/
├── app/main.py      <- yours
├── network/         <- Sumukh
└── text/            <- this package
```

Minimal `backend/app/main.py`:

```python
from fastapi import FastAPI
from text.api import router as text_router

app = FastAPI(title="DRDO classifier")
app.include_router(text_router)          # POST /classify/text, GET /classify/text/{model,health}, POST /classify/text/model/reload
```

If your own code also has a module or folder named `text`, one of the two must be renamed. Keep the package name `text` if you can: the agreed contract is `from text import classify_text`.

## 4. Start the application

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Interactive docs: http://localhost:8000/docs

## 5. Test the endpoints

```bash
curl -s http://localhost:8000/classify/text/health | python3 -m json.tool
curl -s -F "file=@text/samples/invoice/invoice_0000.pdf" http://localhost:8000/classify/text | python3 -m json.tool
curl -s http://localhost:8000/classify/text/model | python3 -m json.tool
curl -s -X POST http://localhost:8000/classify/text/model/reload | python3 -m json.tool
```
Automated: `pytest text/tests/test_api.py -v` (needs `fastapi` and `httpx`; skipped otherwise).

## 6. Responses and error handling

`POST /classify/text` (multipart field **`file`**) → 200:

```json
{"filename": "invoice_0000.pdf", "category": "invoice", "confidence": 0.9262, "source": "svm",
 "best_guess": "invoice", "top_k": [{"category": "invoice", "confidence": 0.9262}],
 "reason": "ok", "extractor": "local-pdf", "file_type": "pdf", "text_chars": 612, "model_version": "local-56d173fe"}
```

| Status | Meaning | What the UI should do |
|---|---|---|
| 200, `reason: ok` | classified | show `category` + `confidence` |
| 200, `reason: low_confidence` | `category` is `unknown`; `best_guess` is what the model would have said | show "Unknown", optionally "closest: …" |
| 200, `reason: too_little_text` | e.g. scanned PDF without a text layer | show "No readable text" |
| 400 | empty file | ask for another file |
| 413 | larger than `TEXT_MAX_BYTES` (20 MB) | say so |
| 415 | unsupported file type | list supported types: pdf, docx, doc*, rtf*, odt*, txt, md, log, csv, html* (*needs Tika) |
| 422 | corrupt, encrypted/password-protected, or binary content | say the file cannot be read |
| 503 | no model available (or registry required but unreachable) | show service-unavailable; check `/classify/text/health` |

Error bodies are FastAPI's `{"detail": "..."}`.
`GET /classify/text/health` → 200 `{"status": "ok" | "degraded", ...}` or 503 `{"status": "unavailable"}`; use it as the container health check.
`degraded` = Tika is configured but down and the local extractors are serving PDF/DOCX/TXT.
`GET /classify/text/model` returns `version, source ("local"|"registry"), categories, unknown_threshold, data_source, real_data_validated, evaluation_note, metrics`.
**Show a "DEMO MODEL" badge whenever `real_data_validated` is `false`.**

## 7. Model location and version rule

* Default model: `backend/text/models/text_svm.joblib` + `model_meta.json`. Override the folder with `TEXT_MODEL_DIR`.
* It is a pickle trained with scikit-learn **1.9.1**. Loading it with a different scikit-learn logs a warning; with `TEXT_STRICT_VERSIONS=1` it refuses (→ 503).
  If you must change the version, retrain: `python -m text.train --data-dir <folder> --data-source <name>`.

## 8. Configuration (environment variables)

| Variable | Default | |
|---|---|---|
| `TIKA_URL` | empty | must include the scheme, e.g. `http://tika:9998` |
| `TEXT_EXTRACTOR` | `auto` | `auto` / `tika` / `local` |
| `MLFLOW_TRACKING_URI` | unset | set → serve the registry's `champion` |
| `TEXT_MODEL_SOURCE` | `auto` | `auto` / `registry` / `local` |
| `TEXT_UNKNOWN_THRESHOLD` | from model metadata | `0` disables rejection |
| `TEXT_MAX_BYTES`, `TEXT_MIN_CHARS`, `TIKA_TIMEOUT`, `TEXT_MODEL_NAME`, `TEXT_MODEL_ALIAS`, `TEXT_MODEL_DIR`, `TEXT_STRICT_VERSIONS` | see README | |

## 9. Connect to the shared MLflow server

```bash
export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000     # your server
python -m text.train --data-dir text/datasets/real --register        # logs a run, registers "text-classifier", promotes if macro-F1 >= champion
```
Then start the app with the same variable set: it loads `models:/text-classifier@champion`, and falls back to the local file if the registry
is down (`TEXT_MODEL_SOURCE=auto`). Promote / roll back by moving the alias `champion` (MLflow UI or
`MlflowClient().set_registered_model_alias("text-classifier", "champion", "<version>")`) and then calling **`POST /classify/text/model/reload`** —
the loaded model is cached and is not refreshed automatically.
For a model comparison view: `MlflowClient().search_model_versions("name='text-classifier'")` and each version's run metrics (`f1_macro`, `accuracy`).
Pick a different registered name for Sumukh's network model; both use the alias `champion`.

## 10. Tika and Docker Compose

```bash
docker compose -f text/docker-compose.tika.yml up -d tika            # Tika on :9998
docker compose -f docker-compose.yml -f text/docker-compose.tika.yml -f text/docker-compose.text-backend.yml up -d    # wire your `backend` service to it
```
`docker-compose.text-backend.yml` is an override: it assumes your main compose file has a service called `backend` (and an `mlflow` service on 5000, or set `MLFLOW_TRACKING_URI`).
The files were validated with `docker compose config`; the containers were **not started** by Person 2.

## 11. Readable HTTP payloads from Sumukh's capture code

`classify_text(body_bytes, "payload")` accepts the readable text of an HTTP message (with or without headers). It **never decodes encrypted traffic**:
TLS records / random-looking / gzip bytes raise `TextExtractionError` (HTTP 422). Compressed bodies must be decompressed by the network side first.
Name the payload `payload` or `payload.txt` — other unknown extensions (e.g. `.bin`) are rejected as unsupported.
