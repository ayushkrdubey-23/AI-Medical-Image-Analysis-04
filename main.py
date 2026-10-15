"""
Master Orchestrator Script for AI-Powered Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Central CLI runner to execute dataset setup, training, evaluation, prediction, and visualization.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset_setup import create_directory_structure, generate_synthetic_medical_images
from train import train_model
from evaluate import evaluate_model
from predict import predict_single_image
from visualize import generate_visualizations

def print_help():
    print("\n=======================================================")
    print("      AI-POWERED MEDICAL IMAGE ANALYSIS - CLI          ")
    print("=======================================================")
    print("Usage: python main.py [command]")
    print("\nAvailable Commands:")
    print("  setup      - Create directories and generate synthetic dataset")
    print("  train      - Train the CNN model on the dataset")
    print("  evaluate   - Evaluate model metrics on the test set")
    print("  predict    - Run sample inference on a test image")
    print("  visualize  - Generate and save performance graphs")
    print("  all        - Run the complete end-to-end pipeline")
    print("=======================================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "setup":
        create_directory_structure()
        generate_synthetic_medical_images()
    elif command == "train":
        train_model()
    elif command == "evaluate":
        evaluate_model()
    elif command == "predict":
        sample_img_dir = os.path.join("data", "test", "Pneumonia")
        if os.path.exists(sample_img_dir) and os.listdir(sample_img_dir):
            sample_path = os.path.join(sample_img_dir, os.listdir(sample_img_dir)[0])
            predict_single_image(sample_path)
        else:
            print("[ERROR] No sample images found. Run 'python main.py setup' first.")
    elif command == "visualize":
        generate_visualizations()
    elif command == "all":
        print("[INFO] Executing complete end-to-end pipeline...")
        create_directory_structure()
        generate_synthetic_medical_images()
        train_model()
        evaluate_model()
        generate_visualizations()
        print("[SUCCESS] Complete pipeline executed successfully!")
    else:
        print(f"[ERROR] Unknown command: {command}")
        print_help()