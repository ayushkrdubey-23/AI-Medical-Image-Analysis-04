"""
CNN Model Architecture Module for Medical Image Analysis
Author: Ayush Kumar Dubey
Description: Defines a custom Convolutional Neural Network (CNN) for binary medical classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_medical_cnn(input_shape=(224, 224, 3)):
    """
    Constructs and compiles a lightweight CNN architecture.
    """
    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary classification: Normal (0) vs Pneumonia (1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    print("[SUCCESS] CNN model architecture built and compiled successfully!")
    return model

if __name__ == "__main__":
    model = build_medical_cnn()
    model.summary()