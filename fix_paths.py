import json

def fix_paths():
    nb_path = 'd:/SDP/new - Copy.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                # Data paths
                line = line.replace('"glomeruli/expert1.pickle"', '"KidneyAI-Dataset/expert1.pickle"')
                line = line.replace('"glomeruli/expert2.pickle"', '"KidneyAI-Dataset/expert2.pickle"')
                line = line.replace('"glomeruli/expert3.pickle"', '"KidneyAI-Dataset/expert3.pickle"')
                line = line.replace('BASE_PATH = "glomeruli"', 'BASE_PATH = "KidneyAI-Dataset/glomeruli"')
                line = line.replace('"glomeruli/glomeruli-patches"', '"KidneyAI-Dataset/glomeruli/glomeruli-patches"')
                
                source[i] = line
            cell['source'] = source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("Paths fixed successfully.")

if __name__ == '__main__':
    fix_paths()
