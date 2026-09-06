VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: install run test demo demo-console report walkthrough clean

install:            ## create the venv and install pinned dependencies
	@if command -v uv >/dev/null 2>&1; then \
	  echo "==> uv found"; \
	  uv venv --python 3.12 --allow-existing $(VENV); \
	  uv pip install --python $(VENV)/bin/python -r requirements.txt; \
	else \
	  echo "==> uv not found; using python3 -m venv"; \
	  python3 -m venv $(VENV); \
	  $(PY) -m pip install --quiet --upgrade pip; \
	  $(PY) -m pip install -r requirements.txt; \
	fi
	@$(PY) -c "import sys, fastapi, langgraph, langchain_openai; \
	print(f'install ok - python {sys.version.split()[0]}')"

run:                ## serve the API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m uvicorn app.main:app --reload --port 8000

test:               ## full offline test suite
	$(PY) -m pytest -q

demo:               ## three runs; writes JSON + logs + HTML report, then opens it
	$(PY) demo.py

report:             ## rebuild demo-report.html from demo-output.json and open it
	$(PY) report.py --open

demo-console:       ## same, with human-readable logs, no browser
	$(PY) demo.py --log-format console --no-open

walkthrough:        ## drive the live API end to end (needs 'make run' in another shell)
	$(PY) walkthrough.py --recheck

clean:
	rm -f agentic_search.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f demo-output.json demo-logs.ndjson demo-report.html
	rm -rf .pytest_cache
