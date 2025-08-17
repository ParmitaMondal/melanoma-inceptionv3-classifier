#!/usr/bin/env python3
# Utilities for data loading, class weights, plotting, etc.
import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _make_dataset_from_dir(dir_path, img_size, batch_size, shuffle=True):
    return keras.preprocessing.image_dataset_from_directory(
        dir_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=shuffle,
    ).prefetch(tf.data.AUTOTUNE)

def get_datasets(data_dir: str, img_size: int, batch_size: int):
    """
    Returns: train_ds, val_ds, class_names
    Expects:
      data/
        train/
          benign/
          melanoma/
        val/     (optional; if missing, uses validation_split on train/)
          benign/
          melanoma/
    """
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    if os.path.isdir(val_root):
        train_ds = _make_dataset_from_dir(train_root, img_size, batch_size, shuffle=True)
        val_ds   = _make_dataset_from_dir(val_root, img_size, batch_size, shuffle=False)
        class_names = train_ds.class_names
    else:
        # Use validation_split on train
        train_ds = keras.preprocessing.image_dataset_from_directory(
            train_root,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="binary",
            shuffle=True,
        )
        val_ds = keras.preprocessing.image_dataset_from_directory(
            train_root,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="binary",
            shuffle=False,
        )
        class_names = train_ds.class_names

    # Cache + prefetch
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names

def compute_class_weights(train_ds, class_names):
    """
    Compute class weights from a labeled tf.data Dataset created by
    image_dataset_from_directory with label_mode='binary'.
    """
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(int(y.numpy()))
    labels = np.array(labels, dtype=int)
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {0: float(weights[0]), 1: float(weights[1])}

def plot_history_curves(history_dict, out_dir: str, prefix: str = ""):
    """Save training curves (loss, auc, accuracy, precision, recall) to PNG."""
    ensure_dir(out_dir)
    def _save(fig, name):
        fig.savefig(os.path.join(out_dir, f"{prefix}_{name}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Loss
    fig = plt.figure()
    plt.plot(history_dict.get("loss", []), label="train")
    plt.plot(history_dict.get("val_loss", []), label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("BinaryCrossentropy"); plt.legend()
    _save(fig, "loss")

    # AUC
    if "auc" in history_dict:
        fig = plt.figure()
        plt.plot(history_dict.get("auc", []), label="train")
        plt.plot(history_dict.get("val_auc", []), label="val")
        plt.title("AUC"); plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.legend()
        _save(fig, "auc")

    # Accuracy
    if "accuracy" in history_dict:
        fig = plt.figure()
        plt.plot(history_dict.get("accuracy", []), label="train")
        plt.plot(history_dict.get("val_accuracy", []), label="val")
        plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        _save(fig, "accuracy")

    # Precision
    if "precision" in history_dict:
        fig = plt.figure()
        plt.plot(history_dict.get("precision", []), label="train")
        plt.plot(history_dict.get("val_precision", []), label="val")
        plt.title("Precision"); plt.xlabel("Epoch"); plt.ylabel("Precision"); plt.legend()
        _save(fig, "precision")

    # Recall
    if "recall" in history_dict:
        fig = plt.figure()
        plt.plot(history_dict.get("recall", []), label="train")
        plt.plot(history_dict.get("val_recall", []), label="val")
        plt.title("Recall"); plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.legend()
        _save(fig, "recall")
