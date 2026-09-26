import re
from pathlib import Path

def natural_sort_key(path_obj):
    parts = [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_obj.name)]
    return (path_obj.is_file(), parts)

def write_tree(directory: Path, file_obj, prefix: str = ""):
    # Sortiert Ordner vor Dateien
    entries = sorted(directory.iterdir(), key=natural_sort_key)

    for i, path in enumerate(entries):
        if path.name == "tree_overview.txt":
            continue

        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        
        file_obj.write(f"{prefix}{connector}{path.name}\n")
        
        if path.is_dir():
            extension = "    " if is_last else "│   "
            write_tree(path, file_obj, prefix + extension)

root_folder = Path(r"C:\Users\lsb-ps\Documents\predictions_overview_lean")
output_file = root_folder / "tree_overview.txt"

if root_folder.is_dir():
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{root_folder.name}/\n")
        write_tree(root_folder, f)