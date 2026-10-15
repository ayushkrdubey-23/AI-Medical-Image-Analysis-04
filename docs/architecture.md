# 🏛️ System Architecture & Workflow Design - AI-Powered Medical Image Analysis

## 1. End-to-End Data Flow Diagram
```text
[ Raw Medical X-Ray Scans (.png / .jpg) ]
                   │
                   ▼
[ Preprocessing Module (`src/preprocessing.py`) ]
  ├── Resize Scans to (224, 224, 3)
  └── Min-Max Pixel Normalization [0.0, 1.0]
                   │
                   ▼
[ TensorFlow `tf.data` Pipeline (Batched & Shuffled) ]
                   │
                   ▼
[ Custom CNN Architecture (`src/model.py`) ]
  ├── Conv2D + MaxPooling Feature Extraction Blocks
  ├── Flatten & Dense Classification Head (Dropout 0.5)
  └── Sigmoid Binary Output Head (Probability Score)
                   │
                   ▼
[ Inference & Visualization Engine (`src/predict.py` & `src/visualize.py`) ]
                   │
                   ▼
[ Interactive Streamlit Clinical Web Dashboard (`dashboard.py`) ]

### 2. `docs/faq_troubleshooting.md` (Project FAQ, Design Decisions & Troubleshooting Log)
```markdown
# ❓ Frequently Asked Questions, Design Rationale & Troubleshooting Log

This document records the architectural choices made during development and solutions to common challenges encountered while building the system.

---

## 🧐 Key Design Decisions & FAQ

### Q1: Why use synthetic/public simulation instead of real hospital PACS data?
* **Rationale:** As a student developer without direct hospital network access, building a virtual simulation pipeline and utilizing standardized dataset folder hierarchies allows for rigorous end-to-end model training, pipeline verification, and portfolio demonstration without privacy or regulatory violations.

### Q2: Why is a custom CNN used instead of a heavy pre-trained model like ResNet50 or EfficientNet?
* **Rationale:** A custom lightweight CNN keeps memory footprints small, ensures fast training cycles on standard local hardware (CPU/GPU), and clearly demonstrates fundamental computer vision and deep learning mechanics for recruiters and evaluation panels.

### Q3: Why include both a CLI (`main.py`) and a Web Dashboard (`dashboard.py`)?
* **Rationale:** The CLI orchestrator allows for automated script execution and CI/CD testing, while the Streamlit web dashboard transforms the backend into a user-friendly software product suitable for clinical presentations.

---

## 🛠️ Troubleshooting & Common Issues Log

### Issue 1: TensorFlow Version Incompatibility with Modern Python
* **Symptom:** `ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0`
* **Solution:** Upgraded pip using `python -m pip install --upgrade pip` and allowed pip to resolve the latest compatible stable release of TensorFlow for Python 3.10+.

### Issue 2: Streamlit Port or Cache Conflicts
* **Symptom:** `Uvicorn server started on port 8501` or cache warnings during model loading.
* **Solution:** Used `@st.cache_resource` on the model loader function to prevent reloading weights into memory on every user interaction.

