"""
Model Evaluation Module for AI-Powered Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Evaluates the saved CNN model performance on test datasets.
"""

import os
import tensorflow as tf
from preprocessing import load_and_preprocess_datasets

def evaluate_model():
    model_path = os.path.join("models", "medical_cnn_model.keras")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at {model_path}. Please run train.py first.")
        return

    print("[INFO] Loading test dataset...")
    _, test_ds = load_and_preprocess_datasets(data_dir="data")

    print(f"[INFO] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("[INFO] Evaluating model performance...")
    results = model.evaluate(test_ds)

    print("\n==============================")
    print("      EVALUATION RESULTS       ")
    print("==============================")
    for name, value in zip(model.metrics_names, results):
        print(f"{name.capitalize()}: {value:.4f}")
    print("==============================\n")

    print("[SUCCESS] Model evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_model()