"""
Result Visualization Module for AI-Powered Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Generates performance curves and saves visual diagnostic sample plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations():
    os.makedirs("outputs", exist_ok=True)
    
    print("[INFO] Generating sample training history plots...")
    # Simulated metrics history for visualization demo if training history object isn't passed directly
    epochs = range(1, 6)
    acc = [0.65, 0.78, 0.85, 0.91, 0.94]
    val_acc = [0.60, 0.72, 0.80, 0.86, 0.89]
    loss = [0.68, 0.45, 0.32, 0.21, 0.15]
    val_loss = [0.70, 0.52, 0.39, 0.30, 0.25]

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join("outputs", "training_accuracy.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"[SUCCESS] Saved training accuracy plot to {acc_plot_path}")

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join("outputs", "training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"[SUCCESS] Saved training loss plot to {loss_plot_path}")

    print("[SUCCESS] All visualization assets generated successfully!")

if __name__ == "__main__":
    generate_visualizations()
    