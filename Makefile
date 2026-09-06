VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: install run test demo demo-console clean

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

demo:               ## three runs: healthy, retry-then-recover, everything down
	$(PY) demo.py

demo-console:       ## same, with human-readable logs
	$(PY) demo.py --log-format console

clean:
	rm -f agentic_search.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
