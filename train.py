#!/usr/bin/env python3
# Training script for Melanoma vs Benign classification using InceptionV3
import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from utils import (
    set_seed,
    get_datasets,
    compute_class_weights,
    plot_history_curves,
    ensure_dir,
)

def build_model(img_size: int) -> keras.Model:
    """InceptionV3 + custom head (binary)."""
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Data augmentation
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")
    x = aug(inputs)

    # InceptionV3 expects specific preprocessing
    x = layers.Lambda(preprocess_input, name="preprocess")(x)

    base = InceptionV3(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False  # Stage 1: freeze

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MelanomaInceptionV3")
    return model, base

def parse_args():
    p = argparse.ArgumentParser(description="Melanoma vs Benign (InceptionV3)")
    p.add_argument("--data_dir", type=str, default="data", help="Root dir with train/ val/ (and optionally test/)")
    p.add_argument("--img_size", type=int, default=299)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr_head", type=float, default=1e-3, help="LR for head training (frozen base)")
    p.add_argument("--lr_finetune", type=float, default=1e-5, help="LR for fine-tuning")
    p.add_argument("--fine_tune_at", type=int, default=249, help="Unfreeze from this base layer index for fine-tuning")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "plots"))

    print(f"[INFO] img_size={args.img_size}, batch_size={args.batch_size}, epochs={args.epochs}")

    # Data
    train_ds, val_ds, class_names = get_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    print(f"[INFO] Classes: {class_names}")

    # Compute class weights for imbalance (from train_ds)
    class_weights = compute_class_weights(train_ds, class_names)
    print(f"[INFO] Class weights: {class_weights}")

    # Build model
    model, base = build_model(args.img_size)

    # Metrics
    metrics = [
        keras.metrics.AUC(name="auc"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]

    # Stage 1: train head only
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_head),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    ckpt_best = os.path.join(args.output_dir, "best_auc.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_best, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    print("[INFO] Stage 1: training head (base frozen)...")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history
    with open(os.path.join(args.output_dir, "history_stage1.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist1.history.items()}, f)
    plot_history_curves(hist1.history, os.path.join(args.output_dir, "plots"), prefix="stage1")

    # Stage 2: fine-tune top layers
    print(f"[INFO] Stage 2: fine-tuning from layer index {args.fine_tune_at} ...")
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= args.fine_tune_at)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    callbacks_ft = [
        keras.callbacks.ModelCheckpoint(
            ckpt_best, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks_ft,
        verbose=1,
    )

    with open(os.path.join(args.output_dir, "history_stage2.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist2.history.items()}, f)
    plot_history_curves(hist2.history, os.path.join(args.output_dir, "plots"), prefix="stage2")

    # Save final model too
    last_path = os.path.join(args.output_dir, "last.h5")
    model.save(last_path)
    print(f"[INFO] Saved final model to {last_path} and best-by-AUC to {ckpt_best}")

if __name__ == "__main__":
    main()
