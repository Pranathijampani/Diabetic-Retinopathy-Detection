"""
train.py
Lesion-Based Diabetic Retinopathy Detection Using Deep Learning
Run once: python train.py
Saves: dr_model.keras
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                         "B.%20Disease%20Grading", "B. Disease Grading")
TRAIN_IMG = os.path.join(BASE, "1. Original Images", "a. Training Set")
TEST_IMG  = os.path.join(BASE, "1. Original Images", "b. Testing Set")
TRAIN_CSV = os.path.join(BASE, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
TEST_CSV  = os.path.join(BASE, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dr_model.keras")

IMG_SIZE   = 224
BATCH_SIZE = 8       # smaller batch = more gradient updates per epoch on small dataset
EPOCHS     = 50      # more epochs, early stopping will cut it short if needed
LABELS     = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ── 1. Load & Preprocess Data ─────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading and preprocessing data")
print("=" * 55)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns  = [c.strip() for c in test_df.columns]

train_df = train_df[["Image name", "Retinopathy grade"]].dropna().drop_duplicates("Image name")
test_df  = test_df[["Image name", "Retinopathy grade"]].dropna().drop_duplicates("Image name")

train_df["Image name"] = train_df["Image name"].str.strip() + ".jpg"
test_df["Image name"]  = test_df["Image name"].str.strip() + ".jpg"
train_df["Retinopathy grade"] = train_df["Retinopathy grade"].astype(str)
test_df["Retinopathy grade"]  = test_df["Retinopathy grade"].astype(str)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print("Class distribution:\n", train_df["Retinopathy grade"].value_counts().sort_index())

# ── 2. Data Generators ────────────────────────────────────────────────────────
print("\nSTEP 2: Building data generators")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.75, 1.25],
    shear_range=0.1,
    fill_mode="nearest"
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, TRAIN_IMG, x_col="Image name", y_col="Retinopathy grade",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="sparse", subset="training", shuffle=True, seed=SEED
)
val_gen = train_datagen.flow_from_dataframe(
    train_df, TRAIN_IMG, x_col="Image name", y_col="Retinopathy grade",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="sparse", subset="validation", shuffle=False, seed=SEED
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, TEST_IMG, x_col="Image name", y_col="Retinopathy grade",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="sparse", shuffle=False
)

# ── 3. Build MobileNetV2 Model ────────────────────────────────────────────────
print("\nSTEP 3: Building MobileNetV2 model")

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Phase 1: freeze all base layers, train only the head
for layer in base.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(5, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

# ── 4. Class Weights ──────────────────────────────────────────────────────────
classes = np.array([0, 1, 2, 3, 4])
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["Retinopathy grade"].astype(int).values
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", {k: round(v, 2) for k, v in class_weights.items()})

# ── 5. Phase 1: Train head only ───────────────────────────────────────────────
print("\nSTEP 4a: Phase 1 - Training head only (15 epochs)")

model.compile(optimizer=Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks_p1 = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1, mode="max"),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1, mode="max")
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks_p1,
    class_weight=class_weights
)

# ── 6. Phase 2: Fine-tune top 50 base layers ─────────────────────────────────
print("\nSTEP 4b: Phase 2 - Fine-tuning top 50 base layers")

for layer in base.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks_p2 = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1, mode="max"),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=5, min_lr=1e-7, verbose=1, mode="max"),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1, mode="max")
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks_p2,
    class_weight=class_weights
)

# Merge histories for plotting
combined = {}
for k in ["accuracy", "val_accuracy", "loss", "val_loss"]:
    combined[k] = history1.history.get(k, []) + history2.history.get(k, [])

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(combined["accuracy"], label="Train")
axes[0].plot(combined["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(combined["loss"], label="Train")
axes[1].plot(combined["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Training history saved -> training_history.png")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
print("\nSTEP 5: Evaluating model")

test_gen.reset()
y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
y_true = test_gen.classes

acc    = accuracy_score(y_true, y_pred)
f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Recall   : {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix - Diabetic Retinopathy Grading")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved -> confusion_matrix.png")

# ── 8. Save Model ─────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved -> {MODEL_PATH}")
print("\nTraining complete!")
