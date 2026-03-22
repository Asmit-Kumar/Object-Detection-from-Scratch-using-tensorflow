# Object Detection from Scratch using TensorFlow

A custom object detection system built entirely from scratch — from synthetic dataset generation to bounding box prediction — using TensorFlow and classical computer vision techniques.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Results](#results)
7. [Branches](#branches)
8. [Contributing](#contributing)
9. [Author](#author)

---

## 🔍 Overview

This project generates synthetic training data from MNIST digits and trains a deep learning model to predict bounding boxes for the digits within 128×128 canvas images. It combines a CNN-based bounding box regression model with a classical connected-component fallback for robust detection.

---

## ✨ Features

- **Synthetic Dataset Generator** — Converts MNIST digits (28×28) into 128×128 canvas images with random placement, generating corresponding bounding box labels
- **CNN Bounding Box Model** — Deep learning regression model for bounding box prediction
- **Classical Fallback Detector** — Connected-component analysis (scipy) as a robust alternative when the CNN prediction is unreliable
- **Evaluation Metrics** — IoU, MAE, MSE for measuring detection accuracy
- **Visualization** — Tools to overlay predicted vs actual bounding boxes

---

## 📂 File Structure

```
root/
├── GenerateDataset.py        # Synthetic dataset generator (MNIST → 128×128 canvas)
├── ObjectDetect.ipynb        # Model training & evaluation notebook
├── bbox_detector.py          # BboxDetector: classical connected-component detector
├── evaluate_bbox.py          # Bbox evaluation metrics (IoU, MAE, MSE, RMSE)
├── utils/
│   ├── __init__.py
│   ├── reader.py             # FileReader: image/label file I/O
│   └── visualizer.py         # Visualizer: bbox overlay visualization
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note**: Data directories (`Images/`, `TestImages/`, `label/`, `Testlabel/`), model files (`.h5`, `.keras`), and logs are gitignored. Generate them locally using the steps below.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Asmit-Kumar/Object-Detection-from-Scratch-using-tensorflow.git
   cd Object-Detection-from-Scratch-using-tensorflow
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Dataset Generation

Place the MNIST CSV file (`train.csv`) in the `csvs/` directory, then run:

```bash
python GenerateDataset.py
```

This generates:
- `Images/` — 128×128 canvas images with randomly placed digits
- `label/` — Bounding box label files (`x_min y_min x_max y_max`)

### Training

Open `ObjectDetect.ipynb` in Jupyter Notebook or VS Code. The notebook covers:
- Data loading and preprocessing
- CNN model architecture for bbox regression
- Training loop with MSE loss
- Evaluation and visualization

### Evaluation

Evaluate the classical bounding box detection accuracy:

```bash
python evaluate_bbox.py
```

Outputs: MAE, MSE, RMSE, Mean IoU, IoU@0.50, IoU@0.75.

---

## 📊 Results

### Base Model Predictions

| Image | Visualization |
|-------|---------------|
| `result/1.png` | ![1.png](result/1.png) |
| `result/2.png` | ![2.png](result/2.png) |
| `result/3.png` | ![3.png](result/3.png) |
| `result/4.png` | ![4.png](result/4.png) |
| `result/5.png` | ![5.png](result/5.png) |

---

## 🌿 Branches

| Branch | Description |
|--------|-------------|
| `main` | Base object detection — dataset generation, CNN bbox model, classical fallback, evaluation |
| `v3` | Performance-tuned model — improved architecture and training with `ObjectDetect_Performance.ipynb` |
| `pipeline` | Full detection pipeline — integrates bbox detection (CNN + fallback) with digit classification |

---

## Contributing

Contributions are welcome!  
Fork the repository, create a branch, and submit a pull request.

---

## 👤 Author

- **Asmit Kumar** — [GitHub](https://github.com/Asmit-Kumar)
