# Test results

Run: 2026-11-07 · Linux container · Python 3.13.18 · pytest 9.1.1 · docker CLI + compose plugin present, **no Docker daemon**; no Tika, MLflow, FastAPI, httpx or network.

```
cd backend
pytest text/tests -v -rs -p no:cacheprovider
=============== 143 passed, 14 skipped in ~21 s ===============

pytest text/tests -q -m "not tika and not mlflow and not docker and not fastapi"      # local tier only
143 passed, 2 skipped, 12 deselected
```

Nothing failed. (The "2 skipped" in the local-tier line are the FastAPI and MLflow test files, which skip themselves at import when those packages are missing.)

## A. Passed locally — 143 tests

| File | Passed | Covers |
|---|---|---|
| `test_extract.py` | 23 | TXT (utf-8 / latin-1 / no extension), PDF, DOCX, type sniffing from bytes, corrupt / fake-extension / binary / unsupported files, legacy `.doc` needing Tika, empty text layer, text cap; Tika client against a fake server (success, `.doc`/`.rtf` routing, auto fallback, tika-only error, HTTP 204 / 415 / 422 / 500) |
| `test_tika_failures.py` | 17 | Tika client vs misbehaving fake servers: timeout, garbled status line, truncated body, connection closed with no reply, binary garbage in a 200, invalid `TIKA_URL`, malformed PDF / DOCX. Auto mode always falls back to local extraction; tika-only mode gives a clear `TextExtractionError` |
| `test_payload_and_formats.py` | 13 | readable HTTP request / response / body-only payloads classified; headers-only → unknown; **TLS records and gzip bytes refused, never decoded** (422); same document as TXT / DOCX / PDF → identical text, category and confidence; classification is repeatable and depends on content (not hard-coded) |
| `test_contract.py` | 16 | exact `{"category","confidence"}` contract and signature, JSON-serialisable, each category on unseen documents, PDF + DOCX end to end, too little text → unknown, unfamiliar documents rejected, threshold 0 disables rejection, missing model error |
| `test_training.py` | 24 | all metrics recorded, metadata (dataset counts, split sizes, TF-IDF / SVM config, library versions), synthetic → DEMO flag, near-duplicate grouping, **groups never cross splits**, duplicates / conflicting labels, too-few-documents refusal, threshold tuned on validation / reported on test, threshold-table arithmetic, the documented limitation (a confident unfamiliar document is NOT rejected), **calibration (ECE / Brier) arithmetic and metadata**, dataset checker |
| `test_train_and_service.py` | 13 | training writes model + metadata, trained model is loaded and used, duplicate removal, promotion-gate rule, registry fallback behaviour, service status codes 400 / 413 / 415 / 422 / 503 |
| `test_robustness.py` | 17 | encrypted PDF → 422, large TXT / PDF / DOCX, registry unavailable → 503, low-confidence and unknown documents, health check (ok / degraded / unavailable), scikit-learn version mismatch (warning; strict mode refuses; **shipped model loads with zero warnings**), backend override declares the documented variables |
| `test_registry_fake.py` | 15 | `registry.py` logic against an **in-memory FAKE MLflow**: first model champion, promotion gate, force / never, MLflow 2.x fallback, what is logged, load champion, rollback + reload, fallback to the local model, registry-only mode fails loudly; **skops trusted types passed to `log_model`** (regression for the Mac finding below), MLflow 2.x / no-skops fallbacks, registry fallback logged as a WARNING |
| `test_compose_config.py` | 5 | `docker compose config` (no daemon needed): Tika file valid on its own, image pinnable via `TIKA_IMAGE`, override invalid alone (as documented), merged with a stand-in main file it sets `TIKA_URL`, `TEXT_EXTRACTOR`, `MLFLOW_TRACKING_URI`, `TEXT_MODEL_SOURCE`, `depends_on: tika` |

The fake-MLflow tests prove `registry.py`'s own logic. **They are not evidence that it works with real MLflow.**
The compose tests prove the files are well-formed. **They are not evidence that the containers start.**

## B. Skipped because a dependency or service was unavailable — 14

| Skipped | Count | Needs |
|---|---|---|
| `test_integration_tika.py` | 9 | a real Tika server at `http://localhost:9998` (or `TIKA_TEST_URL`) |
| `test_integration_docker.py` | 3 | a running Docker daemon |
| `test_integration_mlflow.py` | 1 file (5 tests) | `pip install mlflow` (uses a temporary SQLite store; no server needed) |
| `test_api.py` | 1 file (12 tests) | `pip install fastapi httpx` |

## C. Must be run on your machine / with external services

Everything in B (the real-MLflow and real-FastAPI tests have now run once on a Mac; see above), plus the real-server MLflow run and the live app check. Exact commands, in order: **[MAC_VALIDATION.md](MAC_VALIDATION.md)**.
Until then these are **NOT VERIFIED**: real Tika extraction and its timings and error statuses · real MLflow logging / registration / alias / reload ·
real FastAPI HTTP behaviour · container start-up and the `apache/tika` image · Python 3.11 / 3.12 end-to-end.

## First real-MLflow run (user's MacBook, Python 3.13.18): 3 failures found, fixed

On the Mac `requirements.txt` also installed FastAPI, httpx and MLflow, so the real HTTP and MLflow tests ran for the first time: **148 passed, 3 failed, 17 skipped** (17 = 9 Tika + 3 Docker + 5 compose-config, which skip without a Docker CLI).
The failures were `skops.io.exceptions.UntrustedTypesFoundException: Untrusted types found in the file: ['sklearn.calibration._CalibratedClassifier', 'sklearn.calibration._SigmoidCalibration']` (the three failing test names and the Mac's MLflow version were not reported).
Cause (confirmed in MLflow's documentation and source): recent MLflow saves scikit-learn models in the pickle-free skops format and refuses, on load, any type not on the model's `skops_trusted_types` list; `registry.py` never passed that list. The in-memory fake MLflow could not have revealed this.
Fix: `registry.py` now logs the model with `skops_trusted_types` = exactly those two scikit-learn classes (nothing else is trusted), with fallbacks for MLflow versions that lack the argument. Also: when `MLFLOW_TRACKING_URI` is set and the registry model cannot be loaded, the fallback to the local model is now a logged **WARNING** (it was a silent info line, which would have hidden exactly this failure).
**The fix is covered only by the fake-MLflow tests here; it is NOT yet confirmed against real MLflow.** Re-run `pytest text/tests/test_integration_mlflow.py -v` on the Mac to confirm.

## Evidence that the checks are real

* **Tika failure handling.** The 17 failure tests were run against the `extract.py` in the ZIP you sent (delivered 10:14 UTC): **6 failed** — a garbled or truncated Tika reply and a malformed `TIKA_URL` raised an unhandled exception instead of falling back to local extraction. The current `extract.py` handles them; all 17 pass.
* **Compose file.** The compose file in the ZIP you sent is **invalid on its own** (`docker compose -f text/docker-compose.tika.yml config` → *service "backend" has neither an image nor a build context specified*), so the documented command `up -d tika` would fail. Fixed by splitting it into a Tika-only file and a separate override; `test_compose_config.py` guards against regression.
* **Requirements.** The ZIP's `requirements.txt` pinned `numpy==2.5.3` and `scipy==1.18.1`; their package metadata says `Requires-Python >=3.12`, so `pip install` would fail on Python 3.11. scikit-learn 1.9.1 itself needs `>=3.11`. Now only scikit-learn is pinned; `requirements-lock-py313.txt` holds the exact set the tests ran with.
* **Leakage.** On 90 documents made of 45 pairs of near-identical versions, a naive random split put **14 near-duplicate groups on both sides**; the grouped split used by `train.py` put **0**.

## Python versions

| Version | What was actually done |
|---|---|
| 3.13.18 | full test suite (above) |
| 3.12.3 | not tested (no ML packages installable) |
| 3.11.18 | all source files compile; `text.extract` / Tika client / config import and run (plain-text extraction, Tika-unreachable handling). PDF extraction fails there only because `pypdf` is not installed. **Full suite not run** |

## Measured extraction times (this machine, local extractors, no Tika)

| Document | Size | Extract | Extract + classify |
|---|---|---|---|
| TXT | 5.0 MB | 0.15 s | 1.32 s |
| PDF (4,000 lines) | 64 KB | 0.58 s | 0.52 s |
| PDF (20,000 lines) | 316 KB | 2.47 s | 2.93 s |
| DOCX (20,000 paragraphs) | 92 KB | 0.88 s | 0.87 s |

Text is capped at 200,000 characters before classification. Tika timings: **not measured**.

## Model evaluation — SYNTHETIC data only (`data_source = synthetic-demo`, `real_data_validated = false`)

360 template-generated documents (6 categories × 60; TXT / DOCX / PDF) + 60 out-of-domain documents (recipes, sports reports). Grouped split: train 216 / validation 72 / test 72 (292 near-duplicate groups). Model `local-56d173fe`.

| Test split (72 docs) | Value |
|---|---|
| accuracy / macro precision / macro recall / macro F1 / weighted F1 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 |
| 5-fold grouped CV macro-F1 (train) | 1.0 ± 0.0 |
| calibration: mean confidence vs accuracy | 0.917 vs 1.000 (under-confident) |
| expected calibration error / Brier score | 0.084 / 0.009 |

**This is not real-world performance.** The templates are easy to separate; a perfect score only shows the pipeline runs end to end, and the calibration numbers describe this synthetic set only.
Unknown-threshold table on the held-out test half (30 out-of-domain documents; shipped threshold is the untuned default 0.5):

| threshold | coverage | unknown precision | unknown recall |
|---|---|---|---|
| 0.50 | 1.000 | 1.000 | 0.833 |
| 0.70 | 1.000 | 1.000 | 0.867 |
| 0.80 | 1.000 | 1.000 | 0.967 |
| 0.90 | 0.903 | 0.811 | 1.000 |

## Dependency / model compatibility

* The shipped model was trained with scikit-learn 1.9.1 (recorded in `model_meta.json`), which is what is installed here; it loads with **no** version warnings (a test fails if that changes).
* A version mismatch logs a warning; `TEXT_STRICT_VERSIONS=1` makes it a hard error (HTTP 503).

## Changes found already in the working folder (reviewed and kept)

When this audit started, the working folder contained edits made after the 10:14 delivery that are **not in the ZIP you sent**: stronger Tika error handling (`extract.py`), `test_tika_failures.py`, `test_payload_and_formats.py`, calibration reporting in `train.py` with tests, relaxed numpy / scipy pins plus `requirements-lock-py313.txt`, and a retrained demo model (`local-56d173fe`). Each was reviewed and tested before being kept, as described above.

## Remaining blockers

1. **Real labelled documents** — none available; no accuracy figure can be quoted. (DRDO documents only with authorisation.)
2. **Real Tika, MLflow, FastAPI, Docker daemon** — not available in the build environment; see section C.
3. **Python 3.11 / 3.12 full-suite run.**
