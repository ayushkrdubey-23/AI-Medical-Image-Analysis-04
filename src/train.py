"""
Model Training Script for AI-Powered Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Trains the CNN model using processed datasets and saves weights.
"""

import os
import tensorflow as tf
from preprocessing import load_and_preprocess_datasets
from model import build_medical_cnn

def train_model():
    print("[INFO] Initializing data pipelines...")
    train_ds, test_ds = load_and_preprocess_datasets(data_dir="data")

    print("[INFO] Building CNN model architecture...")
    model = build_medical_cnn(input_shape=(224, 224, 3))

    print("[INFO] Starting model training...")
    epochs = 5
    
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "medical_cnn_model.keras")
    
    print(f"[INFO] Saving trained model to {model_path}...")
    model.save(model_path)
    print("[SUCCESS] Model trained and saved successfully!")

    return history

if __name__ == "__main__":
    train_model()