import re
from pathlib import Path


def natural_sort_key(path_obj):
    parts = [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path_obj.name)
    ]
    return (path_obj.is_file(), parts)


def write_html_tree(directory: Path, file_obj):
    entries = sorted(directory.iterdir(), key=natural_sort_key)

    file_obj.write('<ul style="list-style-type: none; padding-left: 20px;">\n')
    for path in entries:
        if path.name == "tree_overview.html":
            continue

        if path.is_dir():
            file_obj.write(
                f'<li><details><summary style="cursor: pointer;">📁 <b>{path.name}</b></summary>\n'
            )
            write_html_tree(path, file_obj)
            file_obj.write("</details></li>\n")
        else:
            file_obj.write(f"<li>📄 {path.name}</li>\n")
    file_obj.write("</ul>\n")


if __name__ == "__main__":
    root_folder = Path(r"example\Path\results_folder")
    output_file = root_folder / "tree_overview.html"

    if root_folder.is_dir():
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(
                "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n<title>Directory Tree</title>\n"
            )
            f.write(
                "<style>body { font-family: monospace; line-height: 1.6; padding: 20px; }</style>\n"
            )
            f.write("</head>\n<body>\n")

            f.write(f"<h3>📁 {root_folder.name}</h3>\n")
            write_html_tree(root_folder, f)

            f.write("</body>\n</html>\n")

        print(f"Interactive tree successfully exported to {output_file}")
