import json
import re

def patch_vit():
    nb_path = 'd:/SDP/kidney_clean.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = cell.get('source', [])
        source_text = "".join(source)
        
        # 1. Update pip installs to include vit-keras
        if "numpy pandas matplotlib" in source_text:
            if "%pip install vit-keras" not in source_text:
                source.append("%pip install vit-keras --quiet\n")
            cell['source'] = source
            
        # 2. Update the definition of the refiner model AND fine-tuning loop which are in the same cell
        if "def build_refiner_model" in source_text:
            # We already modified the fine-tuning loop partially, so we'll replace the old definition logic via strings
            new_text = source_text
            
            # Replace the EfficientNet import
            new_text = new_text.replace(
                "from tensorflow.keras.applications import EfficientNetV2S\n", 
                "from vit_keras import vit\nfrom tensorflow.keras import layers, models, callbacks\nimport tensorflow as tf\n"
            )
            
            # Replace the function body
            old_func = '''def build_refiner_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """Build EfficientNetV2S based refiner classifier."""
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model'''
            
            new_func = '''def build_refiner_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """Build Vision Transformer (ViT) based refiner classifier."""
    base_model = vit.vit_b16(
        image_size=IMG_SIZE[0],
        activation='sigmoid',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model'''
            
            new_text = new_text.replace(old_func, new_func)
            new_text = new_text.replace("EfficientNetV2S", "Vision Transformer (ViT-B16)")
            new_text = new_text.replace("[:-40]", "[:-10]")
            
            # rewrite source back as a list with keepends
            cell['source'] = [line + '\n' for line in new_text.split('\n')]
            # removing trailing empty strings from split
            if cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]
            # fix double newlines in list:
            cell['source'] = [s.replace('\n\n', '\n') if s != '\n' else s for s in cell['source']]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    patch_vit()
