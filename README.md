# Telecom Customer Churn Analysis

[![CI](https://github.com/ayusprasad/churn-rate-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/churn-rate-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

A notebook-led exploration of telecom customer churn patterns, customer segments, and baseline predictive models.

## What’s inside

- **Exploration** — data quality checks, customer and contract summaries, and churn visualizations.
- **Segmentation** — exploratory comparisons across tenure, charges, and contract groups.
- **Modeling** — baseline classification and clustering experiments.
- **Validation** — notebook structure and Python syntax checks run in GitHub Actions.

```text
notebooks/
  churn_rate_exploration.ipynb   # EDA and churn relationships
  churn_model_comparison.ipynb   # preparation and model experiments
data/
  telco_customer_churn.csv       # local input; intentionally not committed
```

## Data

The dataset is not included. Add a Telco Customer Churn CSV at `data/telco_customer_churn.csv`. The notebooks use common fields such as `Churn`, `tenure`, `Contract`, and `MonthlyCharges`. Keep customer-level data out of Git.

## Run locally

### Prerequisites

- Python 3.11 or newer
- A Telco Customer Churn CSV with the columns used by the notebooks

### Install

```bash
git clone https://github.com/ayusprasad/churn-rate-analysis.git
cd churn-rate-analysis
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install the analysis dependencies and Jupyter:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset at `data/telco_customer_churn.csv`, then launch Jupyter:

```bash
jupyter lab
```

Open either notebook under `notebooks/` and run its cells from top to bottom. The first data-loading cell explains how to proceed if the local dataset is missing.

## Checks

```bash
pip install pytest ruff
ruff check scripts tests
pytest -q
python scripts/validate_notebooks.py
```

The CI workflow checks notebook JSON, parses each Python code cell, runs unit tests, and compiles the validation utilities. It does not execute model training because the source dataset is intentionally local.
