import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os
import shutil
import random

print("🌿 Disease Detection Model Training Started...")
print("=" * 50)

# ── SETTINGS ─────────────────────────────────
FULL_DATASET  = 'plant_disease_data'
LIMITED_PATH  = 'plant_disease_limited'
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 5
MAX_IMAGES    = 150
MODEL_SAVE    = 'models/disease_model.h5'
LABELS_SAVE   = 'models/disease_labels.json'

# ── STEP 1: CREATE LIMITED DATASET ───────────
print(f"\n📂 Creating limited dataset ({MAX_IMAGES} images per class)...")

if os.path.exists(LIMITED_PATH):
    shutil.rmtree(LIMITED_PATH)
    print("🗑️  Removed old limited dataset")

os.makedirs(LIMITED_PATH)

classes = sorted(os.listdir(FULL_DATASET))
print(f"✅ Found {len(classes)} disease classes")

for cls in classes:
    src = os.path.join(FULL_DATASET, cls)
    dst = os.path.join(LIMITED_PATH, cls)
    if not os.path.isdir(src):
        continue
    os.makedirs(dst, exist_ok=True)
    images = [f for f in os.listdir(src)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)
    selected = images[:MAX_IMAGES]
    for img in selected:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

total = sum(len(os.listdir(os.path.join(LIMITED_PATH, c)))
            for c in os.listdir(LIMITED_PATH))
print(f"✅ Limited dataset ready — {total} total images")

# ── STEP 2: DATA GENERATORS ──────────────────
print("\n📂 Loading images into model...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    LIMITED_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    LIMITED_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = len(train_data.class_indices)
print(f"✅ Training images:   {train_data.samples}")
print(f"✅ Validation images: {val_data.samples}")
print(f"✅ Total classes:     {NUM_CLASSES}")
print(f"✅ Steps per epoch:   {len(train_data)}")

# ── STEP 3: BUILD MODEL ───────────────────────
print("\n🧠 Building MobileNetV2 model...")

base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

x      = base.output
x      = GlobalAveragePooling2D()(x)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model ready — {model.count_params():,} parameters")

# ── STEP 4: TRAIN ─────────────────────────────
print(f"\n🚀 Training for {EPOCHS} epochs...")
print(f"⏳ Estimated time: 15-20 minutes. Please wait...\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    verbose=1
)

acc     = history.history['accuracy'][-1] * 100
val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n✅ Training Accuracy:   {acc:.1f}%")
print(f"✅ Validation Accuracy: {val_acc:.1f}%")

# ── STEP 5: SAVE ──────────────────────────────
print("\n💾 Saving model and labels...")

model.save(MODEL_SAVE)

labels = {str(v): k for k, v in train_data.class_indices.items()}
with open(LABELS_SAVE, 'w') as f:
    json.dump(labels, f, indent=2)

print(f"✅ Model saved  → {MODEL_SAVE}")
print(f"✅ Labels saved → {LABELS_SAVE}")
print("\n" + "=" * 50)
print("🎉 Disease Detection model training complete!")
print("Next: Add disease detection route to app.py")