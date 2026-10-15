# 🏥 AI-Powered Medical Image Analysis & Clinical Dashboard System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Commit Strategy](https://img.shields.io/badge/Git-Atomic%20Commits-informational.svg)]()

**An end-to-end deep learning diagnostic pipeline and interactive web application built to classify medical imaging scans (Normal vs. Pneumonia X-Rays) with high accuracy, precision, and recall. Designed as robust portfolio proof of work for placements, internships, and healthcare technology roles.**

</div>

---

## 📑 Table of Contents
- [🏥 AI-Powered Medical Image Analysis \& Clinical Dashboard System](#-ai-powered-medical-image-analysis--clinical-dashboard-system)
  - [📑 Table of Contents](#-table-of-contents)
  - [🚀 Project Overview](#-project-overview)
  - [🏥 Problem Statement \& Healthcare Impact](#-problem-statement--healthcare-impact)
    - [How This AI System Solves Healthcare Challenges:](#how-this-ai-system-solves-healthcare-challenges)
  - [🌐 Industry Relevance \& Use Cases](#-industry-relevance--use-cases)
  - [🛠️ Tech Stack \& Dependencies](#️-tech-stack--dependencies)
  - [📂 Project Architecture \& Folder Structure](#-project-architecture--folder-structure)
  - [⚙️ Core Technical Modules](#️-core-technical-modules)
- [📁 Supporting Files for the `docs/` Folder](#-supporting-files-for-the-docs-folder)
    - [1. `docs/architecture.md`](#1-docsarchitecturemd)
- [System Architecture \& Data Flow - AI-Powered Medical Image Analysis](#system-architecture--data-flow---ai-powered-medical-image-analysis)
  - [1. High-Level Data Flow Diagram](#1-high-level-data-flow-diagram)
    - [2. `docs/api_reference.md`](#2-docsapi_referencemd)
- [API \& Module Reference Guide](#api--module-reference-guide)
  - [Core Python Modules (`src/`)](#core-python-modules-src)
    - [`dataset_setup.py`](#dataset_setuppy)
    - [`preprocessing.py`](#preprocessingpy)
    - [`model.py`](#modelpy)
    - [`train.py`](#trainpy)
    - [`evaluate.py`](#evaluatepy)
    - [`predict.py`](#predictpy)
    - [`visualize.py`](#visualizepy)

---

## 🚀 Project Overview

The **AI-Powered Medical Image Analysis System** is a complete, modular, and production-ready machine learning framework combined with an interactive Streamlit web dashboard (`dashboard.py`). Because direct access to hospital PACS (Picture Archiving and Communication Systems) is restricted for students, this project implements a rigorous **virtual simulation and public dataset architecture** that faithfully replicates enterprise medical diagnostic software workflows.

The system ingests raw chest X-ray scans, preprocesses and normalizes pixel distributions via `tf.data` pipelines, extracts spatial patterns through a custom Convolutional Neural Network (CNN), performs binary classification (Normal vs. Pneumonia), and renders interactive visual analytics, confidence scores, and diagnostic reports via a web user interface[cite: 1].

---

## 🏥 Problem Statement & Healthcare Impact

Diagnostic delays and human error in radiology departments can severely impact patient outcomes, especially in emergency and critical care settings where rapid triage of conditions like pneumonia or pulmonary edema is vital. 

### How This AI System Solves Healthcare Challenges:
* **Accelerated Triaging:** Instantly screens incoming chest X-rays to prioritize critical cases for senior radiologists.
* **Reduction of Human Error:** Serves as a reliable secondary opinion to mitigate fatigue-induced misdiagnoses.
* **Standardized Interpretation:** Applies consistent mathematical thresholding to subtle opacities and lung textures.
* **Assistive Decision Support:** Furnishes confidence probabilities rather than definitive diagnoses, ensuring doctors remain firmly in the clinical loop[cite: 1].

---

## 🌐 Industry Relevance & Use Cases

This project aligns directly with tooling deployed by global health-tech leaders and diagnostic equipment manufacturers, including:
* **Google DeepMind (AlphaFold)** & **IBM Watson Health**[cite: 1] (Medical imaging analytics and predictive diagnostics).
* **Siemens Healthineers**, **GE Healthcare**, & **Philips Healthcare**[cite: 1] (Clinical workflow automation).
* **Qure.ai**, **Aidoc**, & **Annalise.ai**[cite: 1] (AI-driven radiology triage platforms).

---

## 🛠️ Tech Stack & Dependencies

* **Programming Language:** Python 3.10+[cite: 1]
* **Deep Learning Framework:** TensorFlow / Keras[cite: 1]
* **Computer Vision & Image Utilities:** OpenCV (`opencv-python`), Pillow (PIL)[cite: 1]
* **Numerical & Data Analysis:** NumPy, Pandas, Scikit-learn[cite: 1]
* **Visualization Libraries:** Matplotlib, Seaborn[cite: 1]
* **Web Dashboard Framework:** Streamlit[cite: 1]
* **Version Control:** Git & GitHub Actions (CI/CD ready)[cite: 1]

---

## 📂 Project Architecture & Folder Structure

```text
AI-Medical-Image-Analysis/
│
├── data/                         # Training and testing datasets (Categorized by class)
│   ├── train/
│   │   ├── Normal/               # Synthetic/Public Normal X-Ray samples
│   │   └── Pneumonia/            # Synthetic/Public Pneumonia X-Ray samples
│   └── test/
│       ├── Normal/
│       └── Pneumonia/
│
├── notebooks/                    # Jupyter notebooks for experimentation and prototyping
│
├── src/                          # Core modular Python source code
│   ├── __init__.py               # Package initializer
│   ├── dataset_setup.py          # Automated synthetic medical dataset generator
│   ├── preprocessing.py          # tf.data pipeline, resizing, and min-max normalization
│   ├── model.py                  # Custom CNN architecture definition & compilation
│   ├── train.py                  # Model training loop, validation, & weight exporter
│   ├── evaluate.py               # Metrics evaluation (Loss, Accuracy, Precision, Recall)
│   ├── predict.py                # Single image inference & probability thresholding pipeline
│   └── visualize.py              # Training history curves & diagnostic visual generators
│
├── models/                       # Saved trained model weights
│   └── medical_cnn_model.keras   # Exported production model weights
│
├── outputs/                      # Generated performance graphs and visual logs
│   ├── training_accuracy.png     # Epoch-wise accuracy progression curve
│   └── training_loss.png         # Epoch-wise loss minimization curve
│
├── docs/                         # Project documentation, guides, and architecture assets
│   ├── architecture.md           # Detailed system data flow documentation
│   └── api_reference.md          # Internal module reference guide
│
├── images/                       # UI screenshots and dashboard demo assets
│   ├── 01-Dashboard-Overview.png
│   ├── 02-Analytics.png
│   ├── 03-Run-Diagnosis.png
│   └── 04-Predict-Xray-img-Test.png
│
├── dashboard.py                  # Streamlit interactive clinical web dashboard
├── main.py                       # Master CLI Orchestrator Script for end-to-end execution
├── requirements.txt              # Pinned project dependencies
├── .gitignore                    # Excludes venv, cache, and raw model weights
└── README.md                     # Comprehensive project documentation


## ⚙️ Core Technical Modules

1. src/dataset_setup.py
Automates directory provisioning and programmatically generates realistic synthetic medical image samples simulating X-ray contrast profiles, ensuring the repository runs immediately on any machine without external download bottlenecks.

2. src/preprocessing.py
Constructs high-performance tf.data.Dataset pipelines[cite: 1]. Resizes images to standard 224x224x3 dimensions and applies rescaling layers to normalize pixel values to the [0.0, 1.0] interval for stable gradient descent.

3. src/model.py
Builds a custom, lightweight Convolutional Neural Network (CNN) featuring stacked Conv2D feature extractors, MaxPooling2D downsampling layers, Dropout regularization (0.5) to prevent overfitting, and a Dense sigmoid output head for binary classification.

4. src/train.py
Executes model training across epochs with validation split monitoring, compiling with the Adam optimizer (lr=0.001) and binary cross-entropy loss, and exports trained weights to models/medical_cnn_model.keras.

5. src/evaluate.py & src/predict.py
Calculates granular evaluation metrics (Accuracy, Precision, and Recall) and runs inference on individual test scans to output prediction classes with calibrated confidence scores.

6. dashboard.py
An interactive web application built using Streamlit that provides clinicians with real-time scan uploads, visual preview panes, live AI diagnosis cards, and performance history tracking.

💻 Installation & Environment Setup
Follow these steps to set up the project locally on your system:

Step 1: Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/AI-Powered-Medical-Image-Analysis.git](https://github.com/YOUR_USERNAME/AI-Powered-Medical-Image-Analysis.git)
cd AI-Powered-Medical-Image-Analysis
Step 2: Create and Activate Virtual Environment
Windows (PowerShell):

PowerShell
python -m venv venv
.\venv\Scripts\Activate
macOS / Linux (Bash):

Bash
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
Bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit
🚀 Execution Guide (Master CLI & Streamlit Dashboard)
You can run the project either through the master command-line interface or the interactive web dashboard.

Option A: Master CLI Orchestrator (main.py)
Execute specific phases or the entire pipeline using command-line arguments:

Bash
python main.py setup      # Generates synthetic dataset structure
python main.py train      # Trains the CNN model and saves weights
python main.py evaluate   # Evaluates model performance metrics
python main.py predict    # Runs inference on a sample test scan
python main.py visualize  # Generates accuracy and loss plots
python main.py all        # Executes the complete end-to-end pipeline
Option B: Interactive Web Dashboard (dashboard.py)
Launch the Streamlit clinical interface locally:

Bash
streamlit run dashboard.py
Open your browser and navigate to http://localhost:8501 to test live image uploads and diagnostics.

📊 Model Training & Evaluation Metrics
During evaluation against testing partitions, the model tracks:

Loss: Binary cross-entropy measuring error minimization.

Accuracy: Overall correct classification ratio.

Precision: Ratio of true positive pneumonia detections over all predicted positives (minimizing false alarms).

Recall: Ratio of true positive detections over actual positives (ensuring no pneumonia cases are missed).

🔬 Virtual Simulation & Workflow Design
Simulation of Clinical Scans: Synthetic pixel distributions mimic radiographic density variations (normal lung fields vs. localized opacities).

Doctor-AI Collaboration Model: The system acts as a preliminary screening assistant, delivering confidence percentages to guide physician review.

Audit Trail: Every prediction logs parameters for post-hoc clinical review.

🗂️ Git Commit History Strategy
To maintain absolute transparency and demonstrate professional software engineering standards, every file in this repository was added and committed independently with atomic commit messages:

chore: add .gitignore

build: add project dependencies and requirements.txt

feat: add main CLI orchestrator script

feat: implement Streamlit clinical diagnostic web dashboard

feat(src): add automated synthetic medical dataset generator

feat(src): add tf.data preprocessing pipeline

feat(src): implement custom CNN architecture

feat(src): add model training script

feat(src): add model evaluation metrics

feat(src): add single image prediction pipeline

feat(src): add result visualization module

docs: update comprehensive README.md

docs(images): add dashboard screenshots and UI assets

🔮 Future Enhancements
Multi-Class Expansion: Extend classification beyond binary (Normal vs. Pneumonia) to include COVID-19, Tuberculosis, and Lung Nodules.

Grad-CAM Integration: Implement Gradient-weighted Class Activation Mapping to visually highlight regions in X-rays driving the AI's diagnosis.

DICOM Support: Enable direct native parsing of medical .dcm files.

👨‍💻 Author
Ayush Kumar Dubey

B.Tech Computer Science and Engineering[cite: 1]

AI Engineer & Machine Learning Developer

⭐ If you find this project useful for your portfolio or placements, please consider giving it a star on GitHub!


---

# 📁 Supporting Files for the `docs/` Folder

Create the `docs/` folder in your root directory and add the following two markdown files to complete your repository documentation.

### 1. `docs/architecture.md`
```markdown
# System Architecture & Data Flow - AI-Powered Medical Image Analysis

## 1. High-Level Data Flow Diagram
```text
[ Raw Medical X-Ray Image / DICOM ]
               │
               ▼
[ Preprocessing Module (preprocessing.py) ]
  ├── Resize to (224, 224, 3)
  └── Min-Max Normalization [0.0, 1.0]
               │
               ▼
[ TensorFlow tf.data Pipeline (Batched & Shuffled) ]
               │
               ▼
[ CNN Feature Extractor & Classifier (model.py) ]
  ├── Conv2D + MaxPooling Blocks
  ├── Flatten & Dense Layers (Dropout 0.5)
  └── Sigmoid Output Head (Probability Score)
               │
               ▼
[ Inference & Visualization Engines (predict.py / visualize.py) ]
               │
               ▼
[ Streamlit Clinical Dashboard (dashboard.py) ]
2. Module Interactions
dataset_setup.py populates structured directories (data/train/, data/test/).

preprocessing.py consumes raw directories and returns optimized tf.data.Dataset objects.

model.py compiles the sequential neural network architecture.

train.py orchestrates training epochs and saves weights to models/medical_cnn_model.keras.

dashboard.py links the user interface directly to trained weights for real-time inference.


### 2. `docs/api_reference.md`
```markdown
# API & Module Reference Guide

## Core Python Modules (`src/`)

### `dataset_setup.py`
* `create_directory_structure()`: Sets up root and split data directories.
* `generate_synthetic_medical_images(num_samples_per_class)`: Generates baseline simulation arrays.

### `preprocessing.py`
* `load_and_preprocess_datasets(data_dir)`: Returns normalized training and testing `tf.data.Dataset` pipelines.

### `model.py`
* `build_medical_cnn(input_shape)`: Returns a compiled TensorFlow Sequential CNN model.

### `train.py`
* `train_model()`: Fits model on training data, validates against test data, and saves `.keras` weights.

### `evaluate.py`
* `evaluate_model()`: Computes test loss, accuracy, precision, and recall.

### `predict.py`
* `predict_single_image(image_path)`: Loads image, preprocesses, runs inference, and outputs class confidence.

### `visualize.py`
* `generate_visualizations()`: Plots training curves and saves figures to `outputs/`.