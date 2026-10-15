"""
Image Preprocessing and Data Pipeline Module
Author: Ayush Kumar Dubey
Description: Loads, resizes, normalizes, and packages medical images into TensorFlow datasets.
"""

import os
import tensorflow as tf

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8

def load_and_preprocess_datasets(data_dir="data"):
    """
    Creates tf.data.Dataset pipelines for training and testing splits.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    print("[INFO] Loading training dataset...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True
    )

    print("[INFO] Loading testing dataset...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )

    # Normalization layer: scales pixels from [0, 255] to [0.0, 1.0]
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    print("[SUCCESS] Data preprocessing pipelines created successfully!")
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_ds, test_ds = load_and_preprocess_datasets()
    for images, labels in train_ds.take(1):
        print(f"Batch image tensor shape: {images.shape}")
        print(f"Batch label tensor shape: {labels.shape}")
        break