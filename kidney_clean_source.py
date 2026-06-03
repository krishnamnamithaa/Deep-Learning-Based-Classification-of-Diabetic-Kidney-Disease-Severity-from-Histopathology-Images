# --- Cell 1 ---
%pip install torch torchvision torchaudio

# --- Cell 2 ---
# Install all required packages
# Note: kiwisolver must be installed BEFORE matplotlib (cp313t compatibility)
%pip install kiwisolver --pre --quiet
%pip install numpy pandas matplotlib seaborn scikit-learn pillow tqdm --quiet


# --- Cell 3 ---
# advanced image processing
%pip install opencv-python 

# --- Cell 4 ---
%pip install tensorflow --quiet


# --- Cell 5 ---
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Cell 6 ---
# =============================================
# ✅ Load Dataset — kidney_dataset.json
# =============================================
import os, json
import pandas as pd

JSON_FILE     = "kidney_dataset.json"
LABEL_MAP     = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
CLASS_NAMES   = ["Normal", "Mild", "Moderate", "Severe"]
NUM_CLASSES   = 4
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
RANDOM_STATE  = 42

with open(JSON_FILE, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['image_path'] = df['image_path'].apply(os.path.normpath)  # fix Windows mixed paths
df['label']      = df['class'].map(LABEL_MAP)

print(f"✅ Loaded {len(df)} records")
print("\n📊 Class distribution:")
for k, v in LABEL_MAP.items():
    print(f"  Class {k} ({v:8s}): {(df['class'] == k).sum()}")

df.head()


# --- Cell 7 ---
from collections import Counter

# --- Cell 8 ---
# details of json file

#  Load JSON file

json_file = "kidney_dataset.json"  # replace with your JSON file path

with open(json_file, "r") as f:
    data = json.load(f)

# Convert to pandas DataFrame
df_json = pd.DataFrame(data)

# Optional: Check image existence
df_json['exists'] = df_json['image_path'].apply(os.path.exists)


# Optional: Check for corrupted images

corrupted_images = []
for path in df_json['image_path']:
    if os.path.exists(path):
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            corrupted_images.append(path)

df_json['corrupted'] = df_json['image_path'].apply(lambda x: x in corrupted_images)

#  Display first 20 rows in table format
# Color coding the 'class' column
def highlight_class(val):
    colors = {0: '#A1D490', 1: '#F9E79F', 2: '#F5B7B1', 3: '#D2B4DE'}
    return f'background-color: {colors.get(val, "")}'

# Show first 20 rows with styling
df_json[['image_path', 'class', 'exists', 'corrupted']].head(10).style.applymap(highlight_class, subset=['class'])

# --- Cell 9 ---
# -----------------------------
# 1️⃣ Load JSON metadata
# -----------------------------
json_file = "kidney_dataset.json"  # replace with your JSON file
with open(json_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['image_path'] = df['image_path'].apply(os.path.normpath)
print(f"✅ Total samples in JSON: {len(df)}")

# -----------------------------
# 2️⃣ Filter missing/corrupted images
# -----------------------------
def is_valid_image(path):
    if not os.path.exists(path):
        return False
    try:
        img = Image.open(path)
        img.verify()  # check for corruption
        return True
    except:
        return False

df = df[df['image_path'].apply(is_valid_image)].reset_index(drop=True)
print(f"✅ Usable images after filtering: {len(df)}")

# -----------------------------
# 3️⃣ Load images into arrays
# -----------------------------
def load_image(path, target_size=(224, 224)):
    """
    Load an image, convert to RGB, resize and return as numpy array
    """
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img)

# Load all images
images = np.array([load_image(p) for p in df['image_path']])
labels = df['class'].values

print("✅ Images shape:", images.shape)
print("✅ Labels shape:", labels.shape)

# -----------------------------
# 4️⃣ Quick preview of first 5 entries
# -----------------------------
preview_df = df[['image_path', 'class']].head()
print(preview_df)

# --- Cell 11 ---
# Load JSON
with open("kidney_dataset.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Normalize path (fix slashes for OS compatibility)
df['image_path'] = df['image_path'].apply(os.path.normpath)

print("✅ JSON Loaded:", len(df))
df.head()

# --- Cell 12 ---
## Extract Slide Information
def extract_slide(path):
    # Example path: glomeruli/glomeruli-patches/slide_01/xxx.png
    parts = path.split(os.sep)
    for p in parts:
        if "slide_" in p:
            return p
    return None

df['slide'] = df['image_path'].apply(extract_slide)

print("✅ Unique slides:", df['slide'].nunique())
df[['image_path', 'slide']].head()

# --- Cell 13 ---
## Verify Image Exists in Slide Folder
missing = df[~df['image_path'].apply(os.path.exists)]

print("❌ Missing images:", len(missing))

# Keep only valid ones
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

print("✅ Valid dataset size:", len(df))

# --- Cell 14 ---
## Cross-Check with Folder Structure (Slide Matching Proof)
from collections import defaultdict


# -----------------------------
# Count images from folder
# -----------------------------
folder_map = defaultdict(int)

for root, dirs, files in os.walk("KidneyAI-Dataset/glomeruli/glomeruli-patches"):
    for file in files:
        if file.endswith(".png"):
            slide = os.path.basename(root)
            folder_map[slide] += 1

# Convert to DataFrame for sorting
folder_df = pd.DataFrame(list(folder_map.items()), columns=["slide", "count"])
folder_df = folder_df.sort_values(by="count", ascending=True).reset_index(drop=True)

print("📂 Images per slide (FOLDER - Ascending Order):")
display(folder_df)


# -----------------------------
# JSON slide distribution
# -----------------------------
json_df = df['slide'].value_counts().reset_index()
json_df.columns = ['slide', 'count']
json_df = json_df.sort_values(by="count", ascending=True).reset_index(drop=True)

print("\n📊 Images per slide (JSON - Ascending Order):")
display(json_df)

# --- Cell 17 ---
import cv2
import os
import matplotlib.pyplot as plt

def visualize_json_mapping(df, num_samples=6):
    plt.figure(figsize=(18, 6))  # wider figure for path names
    
    for i in range(num_samples):
        row = df.sample(1).iloc[0]  # random sample
        
        img_path = row['image_path']
        label = row['class']
        
        # Check existence
        exists = os.path.exists(img_path)
        
        if not exists:
            print(f"❌ Missing file: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f"Class: {label}", fontsize=10)
        plt.xlabel(os.path.basename(img_path), fontsize=8)  # show file name below image
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle("JSON ↔ Image Mapping Verification", fontsize=14)
    plt.tight_layout()
    plt.show()

# Run the visualization
visualize_json_mapping(df, num_samples=6)

# --- Cell 18 ---
# Strict Validation (Production-Level Check)
invalid_paths = df[~df['image_path'].apply(os.path.exists)]

print(f"Invalid paths: {len(invalid_paths)}")

if len(invalid_paths) > 0:
    print(invalid_paths.head())
else:
    print("✅ All image paths are valid and correctly linked to JSON")

# --- Cell 20 ---
IMG_SIZE = (224, 224) #IMG_SIZE = (width, height)
RANDOM_STATE = 42
BATCH_SIZE = 32

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.array(img)

# Load small batch (test)
sample_df = df.sample(10)

images = np.array([load_image(p) for p in sample_df['image_path']])
labels = sample_df['class'].values

print("✅ Images shape:", images.shape)
print("✅ Labels shape:", labels.shape)

# --- Cell 21 ---
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print("Valid samples:", len(df))

# --- Cell 23 ---
# Load JSON metadata
json_file = "kidney_dataset.json"
df = pd.read_json(json_file)
df['image_path'] = df['image_path'].apply(os.path.normpath)

print("✅ Total samples:", len(df))
print(df['class'].value_counts())

# -----------------------------
# Class Distribution Plot
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Show sample images per class
# -----------------------------
def show_samples_per_class(df, n=2):
    classes = df['class'].unique()
    plt.figure(figsize=(12,4*len(classes)))
    for i, cls in enumerate(classes):
        cls_samples = df[df['class']==cls].sample(n)
        for j, img_path in enumerate(cls_samples['image_path']):
            img = Image.open(img_path).convert("RGB")
            plt.subplot(len(classes), n, i*n + j + 1)
            plt.imshow(img)
            plt.title(f"Class {cls}")
            plt.axis('off')
    plt.show()

show_samples_per_class(df, n=3)

# --- Cell 25 ---
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit  # ✅ import added

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = (224, 224)
RANDOM_STATE = 42

# -----------------------------
# Function to load a single image
# -----------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.array(img)

# -----------------------------
# Create 'slide' column if not exists
# -----------------------------
df['slide'] = df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

# -----------------------------
# Slide-wise Train/Test Split
# -----------------------------
split = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_STATE)
train_idx, test_idx = next(split.split(df, groups=df['slide']))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# -----------------------------
# Split test_df further into Validation
# -----------------------------
split_val = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=RANDOM_STATE)
val_idx, test_idx_final = next(split_val.split(test_df, groups=test_df['slide']))

val_df = test_df.iloc[val_idx].reset_index(drop=True)
test_df = test_df.iloc[test_idx_final].reset_index(drop=True)

# -----------------------------
# Summary
# -----------------------------
print("✅ Train samples:", len(train_df))
print("✅ Validation samples:", len(val_df))
print("✅ Test samples:", len(test_df))

# --- Cell 27 ---
# -----------------------------
# Convert class column to string
# -----------------------------
train_df['class'] = train_df['class'].astype(str)
val_df['class'] = val_df['class'].astype(str)
test_df['class'] = test_df['class'].astype(str)

# -----------------------------
# ImageDataGenerator for augmentation
# -----------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# -----------------------------
# Flow from DataFrame
# -----------------------------
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # now works
    shuffle=True
)



val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Preview batch
# -----------------------------
images, labels = next(train_generator)
print("✅ Batch images shape:", images.shape)
print("✅ Batch labels shape:", labels.shape)

# --- Cell 29 ---
# Load JSON
with open("kidney_dataset.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['image_path'] = df['image_path'].apply(os.path.normpath)

# -----------------------------
# Class Distribution
# -----------------------------
class_counts = df['class'].value_counts().sort_index()

print("📊 Class Distribution:")
print(class_counts)

# -----------------------------
# Percentage Distribution
# -----------------------------
class_percent = (class_counts / len(df)) * 100

print("\n📊 Class Percentage (%):")
print(class_percent)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# -----------------------------
# Imbalance Check Logic
# -----------------------------
max_count = class_counts.max()
min_count = class_counts.min()

imbalance_ratio = max_count / min_count

print(f"\n⚖️ Imbalance Ratio (max/min): {imbalance_ratio:.2f}")

if imbalance_ratio < 1.5:
    print("✅ Dataset is relatively balanced")
elif imbalance_ratio < 3:
    print("⚠️ Dataset is moderately imbalanced")
else:
    print("❌ Dataset is highly imbalanced")

# --- Cell 30 ---
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1️⃣ Compute class weights for balanced training
# -----------------------------
classes = np.unique(df['class'])
y = df['class'].values

# compute weights inversely proportional to class frequency
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)

# convert to dictionary for Keras
class_weights_dict = dict(zip(classes, class_weights_array))
print("⚖️ Class weights to balance training:")
for k, v in class_weights_dict.items():
    print(f"Class {k}: {v:.2f}")

# -----------------------------
# 2️⃣ Ensure class column is string for flow_from_dataframe
# -----------------------------
for dataset in [train_df, val_df, test_df]:
    dataset['class'] = dataset['class'].astype(str)

# -----------------------------
# 3️⃣ ImageDataGenerator
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Validation/Test (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# -----------------------------
# 4️⃣ Create generators
# -----------------------------
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# 5️⃣ Train your model with class weights
# -----------------------------
# Example:
# model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=50,
#     class_weight=class_weights_dict,  # ✅ balances all classes including class 2
#     callbacks=[early_stopping, checkpoint]
# )

# --- Cell 31 ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# 1️⃣ Compute original class distribution
# -----------------------------
original_counts = train_df['class'].value_counts().sort_index()
original_percent = 100 * original_counts / original_counts.sum()

# -----------------------------
# 2️⃣ Compute class weights (balanced)
# -----------------------------
classes = np.unique(train_df['class'])
y = train_df['class'].values

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)
class_weights_dict = dict(zip(classes, class_weights_array))

print("⚖️ Class weights for training:")
for cls, w in class_weights_dict.items():
    print(f"Class {cls}: {w:.2f}")

# -----------------------------
# 3️⃣ Simulate "after balancing" percentages using class weights
# -----------------------------
# The idea: multiply original counts by class weight for visualization
balanced_counts_sim = original_counts.copy()
for cls in classes:
    balanced_counts_sim[cls] = int(original_counts[cls] * class_weights_dict[cls])

balanced_percent_sim = 100 * balanced_counts_sim / balanced_counts_sim.sum()

# -----------------------------
# 4️⃣ Plot before vs after balancing
# -----------------------------
x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, original_percent.values, width, label='Before Balancing', color='skyblue', edgecolor='black')
ax.bar(x + width/2, balanced_percent_sim.values, width, label='After Balancing (simulated)', color='salmon', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels([f'Class {c}' for c in classes])
ax.set_ylabel('Percentage (%)')
ax.set_title('Class Distribution Before and After Balancing')
ax.legend()
plt.show()

# -----------------------------
# 5️⃣ Print percentages for reference
# -----------------------------
print("\n📊 Class percentages BEFORE balancing:")
for cls, pct in zip(classes, original_percent):
    print(f"Class {cls}: {pct:.2f}%")

print("\n📊 Class percentages AFTER balancing (simulated using weights):")
for cls, pct in zip(classes, balanced_percent_sim):
    print(f"Class {cls}: {pct:.2f}%")

# --- Cell 32 ---
from sklearn.utils import resample, class_weight


# -----------------------------
# Function to compute counts and weights
# -----------------------------
def get_class_stats(df):
    counts = df['class'].value_counts().sort_index()
    percentages = df['class'].value_counts(normalize=True).sort_index() * 100
    imbalance_ratio = counts.max() / counts.min()
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['class']),
        y=df['class'].values
    )
    weights_dict = dict(zip(np.unique(df['class']), weights))
    return counts, percentages, imbalance_ratio, weights_dict

# -----------------------------
# 1️⃣ Before balancing
# -----------------------------
counts_before, perc_before, ratio_before, weights_before = get_class_stats(df)
print("⚖️ Before Balancing:")
print("Counts:\n", counts_before)
print("Percentages (%):\n", perc_before.round(2))
print(f"Imbalance Ratio (max/min): {ratio_before:.2f}")
print("Class Weights:", weights_before)

# -----------------------------
# 2️⃣ Balance dataset by oversampling
# -----------------------------
dfs = [df[df['class']==c] for c in df['class'].unique()]
max_count = max(len(d) for d in dfs)
balanced_dfs = [resample(d, replace=True, n_samples=max_count, random_state=42) for d in dfs]
df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# 3️⃣ After balancing
# -----------------------------
counts_after, perc_after, ratio_after, weights_after = get_class_stats(df_balanced)
print("\n⚖️ After Balancing:")
print("Counts:\n", counts_after)
print("Percentages (%):\n", perc_after.round(2))
print(f"Imbalance Ratio (max/min): {ratio_after:.2f}")
print("Class Weights:", weights_after)

# -----------------------------
# 4️⃣ Combined bar chart
# -----------------------------
import numpy as np

labels = counts_before.index.astype(str)
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, counts_before.values, width, label='Before Balancing', color='skyblue')
rects2 = ax.bar(x + width/2, counts_after.values, width, label='After Balancing', color='lightgreen')

ax.set_xlabel('Class')
ax.set_ylabel('Number of Samples')
ax.set_title('Dataset Class Distribution: Before vs After Balancing')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# --- Cell 33 ---
import tensorflow as tf

def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224,224))
    img = img / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices(
    (df['image_path'].values, df['class'].values)
)

dataset = dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

print("✅ TensorFlow dataset ready")

# --- Cell 35 ---
# =============================================
# 1️⃣ Build Primary Model: EfficientNet-B3
# =============================================
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB3

NUM_CLASSES = 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def build_primary_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """Build EfficientNet-B3 based classifier."""
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Freeze base layers initially
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model

primary_model, primary_base = build_primary_model()
primary_model.summary()

# --- Cell 37 ---
# =============================================
# 2️⃣ Train Primary Model (EfficientNet-B3)
# =============================================
import os

# Create directory for model checkpoints
os.makedirs('checkpoints', exist_ok=True)

# --- Phase 1: Transfer Learning (frozen base) ---
primary_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase1_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    )
]

print('🚀 Phase 1: Transfer Learning (frozen base)...')
history_phase1 = primary_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=phase1_callbacks,
    verbose=1
)

# --- Phase 2: Fine-tuning (unfreeze top layers) ---
print('\n🔓 Unfreezing top layers for fine-tuning...')
primary_base.trainable = True
# Freeze all layers except the last 30
for layer in primary_base.layers[:-30]:
    layer.trainable = False

primary_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase2_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ModelCheckpoint(
        'checkpoints/primary_best.keras',
        monitor='val_accuracy', save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    )
]

print('🚀 Phase 2: Fine-tuning...')
history_phase2 = primary_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=phase2_callbacks,
    verbose=1
)

# Save the final primary model
primary_model.save('checkpoints/primary_model_final.keras')
print('\n✅ Primary model training complete!')
print('💾 Model saved to: checkpoints/primary_model_final.keras')
print('💾 Best checkpoint: checkpoints/primary_best.keras')

# --- Cell 39 ---
# =============================================
# 3️⃣ Plot Training History
# =============================================
import matplotlib.pyplot as plt

def plot_training_history(history_phase1, history_phase2, model_name='Primary Model'):
    """Plot combined training history from both phases."""
    # Combine histories
    acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    loss = history_phase1.history['loss'] + history_phase2.history['loss']
    val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    phase1_end = len(history_phase1.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy plot
    axes[0].plot(epochs_range, acc, 'b-', label='Train Accuracy')
    axes[0].plot(epochs_range, val_acc, 'r-', label='Val Accuracy')
    axes[0].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tune Start')
    axes[0].set_title(f'{model_name} — Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs_range, loss, 'b-', label='Train Loss')
    axes[1].plot(epochs_range, val_loss, 'r-', label='Val Loss')
    axes[1].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tune Start')
    axes[1].set_title(f'{model_name} — Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_training_history(history_phase1, history_phase2, 'EfficientNet-B3 (Primary)')

# --- Cell 41 ---
# =============================================
# 4️⃣ Build & Train Refiner Model: EfficientNetV2S
# =============================================
from tensorflow.keras.applications import EfficientNetV2S

def build_refiner_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
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

    return model, base_model

refiner_model, refiner_base = build_refiner_model()
refiner_model.summary()

# --- Phase 1: Transfer Learning ---
refiner_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

ref_callbacks_p1 = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    )
]

print('🚀 Refiner Phase 1: Transfer Learning...')
ref_history_p1 = refiner_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=ref_callbacks_p1,
    verbose=1
)

# --- Phase 2: Fine-tuning ---
print('\n🔓 Unfreezing top layers of refiner...')
refiner_base.trainable = True
for layer in refiner_base.layers[:-40]:
    layer.trainable = False

refiner_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

ref_callbacks_p2 = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ModelCheckpoint(
        'checkpoints/refiner_best.keras',
        monitor='val_accuracy', save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    )
]

print('🚀 Refiner Phase 2: Fine-tuning...')
ref_history_p2 = refiner_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=ref_callbacks_p2,
    verbose=1
)

# Save the final refiner model
refiner_model.save('checkpoints/refiner_model_final.keras')
print('\n✅ Refiner model training complete!')
print('💾 Model saved to: checkpoints/refiner_model_final.keras')
print('💾 Best checkpoint: checkpoints/refiner_best.keras')

# Plot refiner history
plot_training_history(ref_history_p1, ref_history_p2, 'EfficientNetV2S (Refiner)')

# --- Cell 43 ---
# =============================================
# 5️⃣ Confidence Cascade Evaluation
# =============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

CONFIDENCE_THRESHOLD = 0.7
CLASS_NAMES = ['Normal', 'Mild', 'Moderate', 'Severe']

# --- Step 1: Get test predictions from primary model ---
print('🔍 Running primary model on test set...')
test_generator.reset()
primary_probs = primary_model.predict(test_generator, verbose=1)
primary_preds = np.argmax(primary_probs, axis=1)
primary_confidence = np.max(primary_probs, axis=1)

# True labels
true_labels = test_generator.classes

# --- Step 2: Identify low-confidence samples ---
low_conf_mask = primary_confidence < CONFIDENCE_THRESHOLD
high_conf_mask = ~low_conf_mask
n_low = np.sum(low_conf_mask)
n_total = len(true_labels)
print(f'\n📊 Confidence Cascade Stats:')
print(f'   Total test samples: {n_total}')
print(f'   High confidence (≥{CONFIDENCE_THRESHOLD}): {n_total - n_low} ({100*(n_total-n_low)/n_total:.1f}%)')
print(f'   Low confidence (<{CONFIDENCE_THRESHOLD}): {n_low} ({100*n_low/n_total:.1f}%)')

# --- Step 3: Run refiner on low-confidence samples ---
final_preds = primary_preds.copy()
final_probs = primary_probs.copy()

if n_low > 0:
    print(f'\n🔍 Running refiner model on {n_low} low-confidence samples...')
    low_conf_indices = np.where(low_conf_mask)[0]

    # Collect low-confidence images from the test generator
    test_generator.reset()
    all_test_images = []
    all_test_labels = []
    for i in range(len(test_generator)):
        batch_x, batch_y = test_generator[i]
        all_test_images.append(batch_x)
        all_test_labels.append(batch_y)
    all_test_images = np.concatenate(all_test_images, axis=0)

    low_conf_images = all_test_images[low_conf_indices]
    refiner_probs = refiner_model.predict(low_conf_images, verbose=1)
    refiner_preds = np.argmax(refiner_probs, axis=1)

    # Replace low-confidence predictions with refiner's
    final_preds[low_conf_mask] = refiner_preds
    final_probs[low_conf_mask] = refiner_probs
    print('✅ Cascade complete!')
else:
    print('✅ All samples are high-confidence, no refiner needed!')

# --- Step 4: Evaluation ---
print('\n' + '='*60)
print('📋 CLASSIFICATION REPORT (Confidence Cascade)')
print('='*60)
print(classification_report(
    true_labels, final_preds,
    target_names=CLASS_NAMES, digits=4
))

# Also show primary-only report for comparison
print('\n' + '='*60)
print('📋 CLASSIFICATION REPORT (Primary Model Only)')
print('='*60)
print(classification_report(
    true_labels, primary_preds,
    target_names=CLASS_NAMES, digits=4
))

# --- Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Primary only
cm_primary = confusion_matrix(true_labels, primary_preds)
sns.heatmap(cm_primary, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('Confusion Matrix — Primary Only')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Cascade
cm_cascade = confusion_matrix(true_labels, final_preds)
sns.heatmap(cm_cascade, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('Confusion Matrix — Confidence Cascade')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()

# --- AUC-ROC Curves ---
true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231']

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(true_labels_bin[:, i], final_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Confidence Cascade')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('\n✅ Evaluation complete!')

