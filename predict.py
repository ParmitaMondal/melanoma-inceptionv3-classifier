#!/usr/bin/env python3
# Batch inference script
import os
import argparse
import glob
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input

def parse_args():
    p = argparse.ArgumentParser(description="Predict melanoma probability on a folder of images.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--out", type=str, default="predictions.csv")
    p.add_argument("--img_size", type=int, default=299)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()

def list_image_files(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(exts)]

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # InceptionV3 preprocessing
    return img

def main():
    args = parse_args()
    paths = list_image_files(args.images_dir)
    if not paths:
        raise SystemExit(f"No images found in {args.images_dir}")

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: (load_image(p, args.img_size), p), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = keras.models.load_model(args.model_path, compile=False)

    probs = []
    files = []
    for batch_imgs, batch_paths in ds:
        p = model.predict(batch_imgs, verbose=0).reshape(-1)
        probs.extend(p.tolist())
        files.extend([x.numpy().decode("utf-8") for x in batch_paths])

    import numpy as np
    preds = (np.array(probs) >= args.threshold).astype(int)

    df = pd.DataFrame({
        "filepath": files,
        "prob_melanoma": probs,
        "pred_label": preds
    })
    df.to_csv(args.out, index=False)
    print(f"[INFO] Wrote predictions: {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
