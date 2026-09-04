VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: install run test demo lint clean

install:            ## create the venv and install pinned dependencies
	uv venv --python 3.12 $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

run:                ## serve the API on http://127.0.0.1:8000 (docs at /docs)
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

test:               ## full offline test suite
	$(PY) -m pytest -q

demo:               ## three runs: healthy, retry-then-recover, everything down
	$(PY) demo.py

demo-console:       ## same, with human-readable logs
	$(PY) demo.py --log-format console

clean:
	rm -f agentic_search.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
