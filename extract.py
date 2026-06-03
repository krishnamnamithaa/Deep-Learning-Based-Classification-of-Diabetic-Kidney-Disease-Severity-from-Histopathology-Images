import json

def extract_code():
    with open('d:/SDP/new - Copy.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open('d:/SDP/notebook_source.py', 'w', encoding='utf-8') as outfile:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                outfile.write(f'# --- Cell {i} ---\n')
                source = "".join(cell.get('source', []))
                outfile.write(source)
                outfile.write('\n\n')

if __name__ == '__main__':
    extract_code()
