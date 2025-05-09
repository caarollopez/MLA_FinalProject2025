import nbformat

notebook_path = "Fake_News_Detection_bis.ipynb"  # Cambia este nombre

nb = nbformat.read(open(notebook_path, "r", encoding="utf-8"), as_version=4)

nb.metadata.pop("widgets", None)

for cell in nb.cells:
    cell.metadata.pop("widgets", None)

nbformat.write(nb, open(notebook_path, "w", encoding="utf-8"))

print("Cleaned notebook saved (overwritten)")