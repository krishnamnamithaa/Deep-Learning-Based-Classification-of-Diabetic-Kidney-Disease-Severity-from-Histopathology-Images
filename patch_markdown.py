import json

def patch_markdown():
    nb_path = 'd:/SDP/kidney_clean.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
            
        source = cell.get('source', [])
        for i in range(len(source)):
            # Replace references to EfficientNetV2S with ViT-B16 in the markdown
            source[i] = source[i].replace('EfficientNetV2S', 'Vision Transformer (ViT-B16)')
        
        cell['source'] = source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("✅ Markdown sections successfully patched!")

if __name__ == '__main__':
    patch_markdown()
