"""
Prediction and Inference Module for AI-Powered Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Loads a single medical image, preprocesses it, and performs model inference.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

def predict_single_image(image_path):
    model_path = os.path.join("models", "medical_cnn_model.keras")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at {model_path}. Please train the model first.")
        return

    print(f"[INFO] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"[INFO] Processing image: {image_path}...")
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path does not exist: {image_path}")
        return

    # Load and resize image to 224x224
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    
    # Normalize pixel values to [0.0, 1.0]
    img_array = img_array / 255.0
    
    # Expand dimensions to match model input shape (Batch, Height, Width, Channels)
    img_array = np.expand_dims(img_array, axis=0)

    print("[INFO] Performing model inference...")
    prediction = model.predict(img_array)[0][0]
    
    # Threshold at 0.5 for binary classification (0: Normal, 1: Pneumonia)
    class_names = ["Normal", "Pneumonia"]
    predicted_class_idx = 1 if prediction >= 0.5 else 0
    confidence = prediction if predicted_class_idx == 1 else (1.0 - prediction)

    print("\n==============================")
    print("      DIAGNOSTIC PREDICTION    ")
    print("==============================")
    print(f"Image File : {image_path}")
    print(f"Prediction : {class_names[predicted_class_idx]}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("==============================\n")

    print("[SUCCESS] Prediction completed successfully!")

if __name__ == "__main__":
    # Test with a sample from the test directory if available
    sample_img_dir = os.path.join("data", "test", "Pneumonia")
    if os.path.exists(sample_img_dir):
        files = os.listdir(sample_img_dir)
        if files:
            sample_path = os.path.join(sample_img_dir, files[0])
            predict_single_image(sample_path)
        else:
            print("[WARNING] No test images found in Pneumonia directory.")
    else:
        print("[WARNING] Test directory not found. Please run dataset setup first.")