import json

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
            
        # 2. Update the definition of the refiner model
        if "def build_refiner_model" in source_text:
            vit_source = [
                "# =============================================\n",
                "# 4 Build & Train Refiner Model: Vision Transformer (ViT-B16)\n",
                "# =============================================\n",
                "from vit_keras import vit\n",
                "from tensorflow.keras import layers, models, callbacks\n",
                "import tensorflow as tf\n",
                "\n",
                "def build_refiner_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):\n",
                "    \"\"\"Build Vision Transformer (ViT) based refiner classifier.\"\"\"\n",
                "    base_model = vit.vit_b16(\n",
                "        image_size=IMG_SIZE[0],\n",
                "        activation='sigmoid',\n",
                "        pretrained=True,\n",
                "        include_top=False,\n",
                "        pretrained_top=False\n",
                "    )\n",
                "    base_model.trainable = False\n",
                "\n",
                "    model = models.Sequential([\n",
                "        base_model,\n",
                "        layers.Flatten(),\n",
                "        layers.BatchNormalization(),\n",
                "        layers.Dropout(0.4),\n",
                "        layers.Dense(512, activation='relu'),\n",
                "        layers.BatchNormalization(),\n",
                "        layers.Dropout(0.4),\n",
                "        layers.Dense(256, activation='relu'),\n",
                "        layers.Dropout(0.3),\n",
                "        layers.Dense(num_classes, activation='softmax')\n",
                "    ])\n",
                "    return model, base_model\n",
                "\n",
                "refiner_model, refiner_base = build_refiner_model()\n",
                "refiner_model.summary()\n"
            ]
            cell['source'] = vit_source

        # 3. Update the finetuning loop & plot name
        if "Unfreezing top layers of refiner" in source_text:
            new_source = []
            for line in source:
                if "[:-40]" in line:
                    new_source.append("for layer in refiner_base.layers[:-10]:\n")
                elif "plot_training_history" in line and "EfficientNetV2S" in line:
                    new_source.append("plot_training_history(ref_history_p1, ref_history_p2, 'Vision Transformer (Refiner)')\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    patch_vit()
