"""Train a CNN on CIFAR-10 (60,000 32x32 images, 10 classes).

The dataset is downloaded automatically by Keras on first run (~170 MB).
Usage:  python train.py --epochs 30
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
OUT = "models"


def conv_block(x, filters, drop):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(drop)(x)


def build_model():
    inp = keras.Input((32, 32, 3))
    # Augmentation layers are only active during training
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.Rescaling(1 / 255.0)(x)  # model accepts raw 0-255 pixels
    for filters, drop in [(32, 0.2), (64, 0.3), (128, 0.4)]:
        x = conv_block(x, filters, drop)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out, name="cifar10_cnn")


def save_plots(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k, t in zip(ax, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        a.plot(history.history[k], label="train")
        a.plot(history.history["val_" + k], label="validation")
        a.set_title(t), a.set_xlabel("epoch"), a.legend(), a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_curves.png", dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.ravel(), y_test.ravel()
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    # Export 20 test images so you can try them in the web app
    for i in range(20):
        Image.fromarray(x_test[i]).resize((256, 256), Image.LANCZOS).save(
            f"samples/{i:02d}_{CLASSES[y_test[i]]}.png")

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ]
    history = model.fit(x_train, y_train, validation_split=0.1,
                        epochs=a.epochs, batch_size=a.batch_size, callbacks=cbs)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    model.save(f"{OUT}/cifar10_cnn.keras")
    with open(f"{OUT}/classes.json", "w") as f:
        json.dump(CLASSES, f)
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump({"test_accuracy": round(float(acc), 4),
                   "test_loss": round(float(loss), 4),
                   "epochs_run": len(history.history["loss"]),
                   "parameters": int(model.count_params())}, f, indent=2)
    save_plots(history)
    print(f"Saved model and metrics to ./{OUT}/")


if __name__ == "__main__":
    main()
