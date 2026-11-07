# Validate the text branch on your MacBook

Run these **in order** and keep the output of each. Anything marked *(never run so far)* has not been executed by anyone yet,
so a failure there is useful information, not a mistake of yours.

## 0. Prerequisites

```bash
python3 --version            # must be 3.11 or newer.  If not:  brew install python@3.13
docker --version             # optional: needed only for the Tika / compose steps (install Docker Desktop)
```

## 1. Unzip and install

```bash
cd ~/path/to/project         # the folder that should contain backend/
unzip ~/Downloads/text_branch.zip       # creates backend/text/
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r text/requirements.txt
```
If `pip install` fails on scikit-learn / numpy, send me the last 20 lines of the error. (`requirements.txt` only pins scikit-learn; the others float to what your Python supports.)
Exact set the tests were run with on Python 3.13 (Linux): `pip install -r text/requirements-lock-py313.txt`.

## 2. Local tests (no external services) — expect all green

```bash
pytest text/tests -v -rs -p no:cacheprovider 2>&1 | tee text_tests_local.txt
tail -3 text_tests_local.txt
```
Expected on the build machine (Linux, Python 3.13, no Tika / MLflow / FastAPI / Docker daemon): `143 passed, 14 skipped`.
Expected on a Mac where `requirements.txt` installed FastAPI, httpx and MLflow (no Tika, no Docker): `155 passed, 17 skipped`
(the 12 real HTTP tests and 5 real MLflow tests now run; the skips are 9 Tika + 3 Docker + 5 compose-config).
The skips are the Tika, MLflow, Docker and FastAPI tests; they stop being skipped as you install those (steps 4–6). Without the Docker CLI
on your Mac, the 5 compose-config tests also skip. A different Python / numpy may change timings but not the pass count. **Failures in step 2 matter most**: send me `text_tests_local.txt`.

## 3. Model compatibility check

```bash
python - <<'EOF'
import warnings, json, pathlib, sklearn
warnings.simplefilter("error")          # any scikit-learn version warning becomes an error
import text
print("installed scikit-learn:", sklearn.__version__)
print("model trained with:   ", json.loads(pathlib.Path("text/models/model_meta.json").read_text())["sklearn_version"])
print(text.model_info()["version"], "loaded OK")
EOF
python -m text text/samples/*/*          # classifies the six sample files
```
If it prints a version warning/error, retrain with your versions: `python -m text.data.generate_sample_dataset --out /tmp/sample --per-class 60 --seed 7 && python -m text.train --data-dir /tmp/sample --data-source synthetic-demo`.

## 4. FastAPI HTTP tests *(never run so far)*

```bash
pip install fastapi httpx "uvicorn>=0.27" python-multipart
pytest text/tests/test_api.py -v -rs
```
Then the live app (see `INTEGRATION_GUIDE.md` §3): create `app/main.py` with the 4-line example, and

```bash
uvicorn app.main:app --port 8000 &
curl -s http://localhost:8000/classify/text/health
curl -s -F "file=@text/samples/invoice/invoice_0000.pdf" http://localhost:8000/classify/text
kill %1
```

## 5. Real Apache Tika *(never run so far)*

```bash
docker compose -f text/docker-compose.tika.yml up -d tika
sleep 20 && curl -s http://localhost:9998/tika          # a greeting means it is up
pytest text/tests/test_integration_tika.py -v -rs -s     # -s prints the extraction timings
TIKA_URL=http://localhost:9998 python -m text text/samples/invoice/invoice_0000.pdf     # look for "extractor":"tika"
docker compose -f text/docker-compose.tika.yml down
```
On an Apple-silicon Mac, if Docker reports a platform mismatch, run the first command with `DOCKER_DEFAULT_PLATFORM=linux/amd64` in front (slower, emulated).
The next test needs Docker running: `pytest text/tests/test_integration_docker.py -v -rs` (starts and stops the container itself).

## 6. Real MLflow *(never run so far)*

```bash
pip install mlflow
pytest text/tests/test_integration_mlflow.py -v -rs      # uses a temporary SQLite store, no server needed
```
Then against a real server:

```bash
mlflow server --host 127.0.0.1 --port 5000 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m text.data.generate_sample_dataset --out /tmp/sample --per-class 60 --seed 7
python -m text.train --data-dir /tmp/sample --data-source synthetic-demo --register      # twice, to get v1 and v2
open http://127.0.0.1:5000                                                              # Models -> text-classifier -> alias "champion"
python -c "import text; i = text.model_info(); print(i['source'], i['version'])"                 # expect: registry 2
kill %1
```

## 7. Send me

* `text_tests_local.txt` (step 2) and the output of steps 3–6, especially every failure.
* Which Python version and chip (Intel / Apple silicon) you have.
