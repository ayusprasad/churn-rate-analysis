from pathlib import Path

from scripts.validate_notebooks import NOTEBOOK_DIRECTORY, validate_notebook

NOTEBOOKS = [
    "churn_rate_exploration.ipynb",
    "churn_model_comparison.ipynb",
]


def test_expected_notebooks_are_present_and_valid():
    for name in NOTEBOOKS:
        path = NOTEBOOK_DIRECTORY / name
        assert path.is_file()
        assert validate_notebook(path) > 0


def test_notebook_data_path_is_consistent():
    for name in NOTEBOOKS:
        source = (NOTEBOOK_DIRECTORY / name).read_text(encoding="utf-8")
        assert "data/telco_customer_churn.csv" in source
