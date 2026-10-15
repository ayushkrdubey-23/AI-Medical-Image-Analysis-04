"""
Dataset Setup & Simulation Script for AI-Powered Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Sets up directory structures and generates synthetic medical image samples 
for training and testing the classification model.
"""

import os
import numpy as np
from PIL import Image

def create_directory_structure():
    print("[INFO] Creating dataset directory structure...")
    base_dir = "data"
    categories = ["Normal", "Pneumonia"]
    splits = ["train", "test"]
    
    for split in splits:
        for category in categories:
            dir_path = os.path.join(base_dir, split, category)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")

def generate_synthetic_medical_images(num_samples_per_class=15):
    print("[INFO] Generating synthetic medical image samples for simulation...")
    base_dir = "data"
    categories = ["Normal", "Pneumonia"]
    splits = ["train", "test"]
    
    np.random.seed(42)
    
    for split in splits:
        # More samples in train, fewer in test
        count = num_samples_per_class if split == "train" else num_samples_per_class // 3
        
        for category in categories:
            class_dir = os.path.join(base_dir, split, category)
            
            for i in range(count):
                # Generate a random 224x224 grayscale image mimicking an X-ray scan
                if category == "Normal":
                    # Smoother distribution for normal scans
                    img_array = np.random.normal(loc=120, scale=30, size=(224, 224))
                else:
                    # Higher intensity variations or patches mimicking anomalies/opacities
                    img_array = np.random.normal(loc=140, scale=50, size=(224, 224))
                    # Add a synthetic opacity patch in the center
                    img_array[90:130, 90:130] = np.random.normal(loc=200, scale=20, size=(40, 40))
                
                # Clip values to valid pixel range [0, 255] and convert to uint8
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                # Save as PNG image
                img = Image.fromarray(img_array)
                file_path = os.path.join(class_dir, f"sample_{i+1}.png")
                img.save(file_path)
                
    print("[SUCCESS] Synthetic medical datasets generated successfully in 'data/'!")

if __name__ == "__main__":
    create_directory_structure()
    generate_synthetic_medical_images()