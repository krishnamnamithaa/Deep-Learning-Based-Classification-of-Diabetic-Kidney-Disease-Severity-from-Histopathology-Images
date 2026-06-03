import json
import sys

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"--- Analysis for {file_path} ---")
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            lines = source.split('\n')
            for line in lines:
                if 'dataset' in line.lower() or 'json' in line.lower() or 'kidney' in line.lower() or 'path' in line.lower() or 'csv' in line.lower():
                    print(f"Cell {i}: {line}")

analyze_notebook('d:/SDP/new.ipynb')
analyze_notebook('d:/SDP/new - Copy.ipynb')
