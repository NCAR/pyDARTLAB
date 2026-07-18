# run to remove various metadata from running the notebooks
from pathlib import Path
import nbformat

for path in Path("notebooks").rglob("*.ipynb"):
    nb = nbformat.read(path, as_version=4)

    # Remove notebook metadata
    nb.metadata.pop("language_info", None)

    #if "kernelspec" in nb.metadata:
    #    nb.metadata["kernelspec"].pop("display_name", None)

    # Clear outputs and execution counts
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    nbformat.write(nb, path)

    print(f"Cleaned {path}")
