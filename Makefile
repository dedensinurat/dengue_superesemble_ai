.PHONY: setup lint test ci demo forecast evaluate api dashboard clean

PYTHON := python
PKG    := denguecast

setup:
	$(PYTHON) -m pip install -e ".[dev,geo,pdf,image]"

lint:
	ruff check src/ tests/
	mypy src/

test:
	pytest tests/ -v --cov=$(PKG) --cov-report=term-missing --cov-fail-under=80

ci: lint test

demo:
	$(PYTHON) -m $(PKG) data synth --seed 42
	$(PYTHON) -m $(PKG) data validate
	$(PYTHON) -m $(PKG) features build --origin 2022-12
	$(PYTHON) -m $(PKG) run --origin 2022-12 --synthetic
	$(PYTHON) -m $(PKG) evaluate --synthetic

forecast:
	$(PYTHON) -m $(PKG) run --origin $(ORIGIN)

evaluate:
	$(PYTHON) -m $(PKG) evaluate --origin $(ORIGIN)

api:
	uvicorn $(PKG).api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run apps/dashboard/app.py --server.port 8501

clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov
