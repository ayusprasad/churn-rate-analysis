from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"


def validate_notebook(path: Path) -> int:
    """Validate notebook JSON, code syntax, and clean output state."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path.name} must use notebook format 4")

    code_cells = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") not in {"code", "markdown"}:
            raise ValueError(f"{path.name} cell {index} has an unsupported type")
        if cell.get("cell_type") != "code":
            continue

        code_cells += 1
        if cell.get("outputs"):
            raise ValueError(f"{path.name} cell {index} contains saved output")
        if cell.get("execution_count") is not None:
            raise ValueError(f"{path.name} cell {index} contains an execution count")
        source = cell.get("source", [])
        ast.parse("".join(source) if isinstance(source, list) else source)

    return code_cells


def main() -> None:
    notebooks = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found in {NOTEBOOK_DIRECTORY}")

    for path in notebooks:
        count = validate_notebook(path)
        print(f"Validated {path.name}: {count} Python cells")


if __name__ == "__main__":
    main()
