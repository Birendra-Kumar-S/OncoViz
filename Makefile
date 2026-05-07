.PHONY: install run test clean lint help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

venv:  ## Create virtual environment and install dependencies
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:  ## Run the Streamlit dashboard
	streamlit run app.py

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Lint with flake8
	flake8 src/ tests/ app.py --max-line-length=120

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info
