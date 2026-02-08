# Object Detection from Scratch using TensorFlow

This project demonstrates an end-to-end solution for generating a custom object detection dataset using the MNIST digits dataset and training a deep learning model to predict bounding boxes for the digits in 128x128 canvas images.

![Example Prediction](result/1.png)

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Usage](#usage)
   - [Dataset Generation](#dataset-generation)
   - [Training](#training)
   - [Evaluation & Inference](#evaluation--inference)
6. [Results](#results)
7. [License](#license)
8. [Author](#author)

---

## 🔍 Overview

This project involves generating images derived from MNIST digits and training a deep learning model to predict bounding boxes for the digits within those images. It provides a practical demonstration of dataset preparation, deep learning model training, and visualization of results.

## ✨ Features

- **Custom Dataset Generator**: Converts MNIST digit images (28x28) into 128x128 canvas images with random placement.
- **Object Detection Model**: Deep learning model (CNN-based) to predict bounding boxes.
- **Digit Recognition**: Classification model to identify the digit within the bounding box.
- **Visualization**: Tools to visualize predictions vs ground truth.

## 📂 File Structure

```
root/
├── GenerateDataset.py   # Script to generate the synthetic dataset
├── ObjectDetect.ipynb   # Main notebook for training and evaluation
├── bbox_detector.py     # Bounding box detector class
├── digit_detector.py    # Pipeline for detection + classification
├── evaluate_bbox.py     # Script to evaluate model performance
├── Models/              # Saved model files (.h5, .keras)
├── utils/               # Utility functions
│   ├── reader.py        # File reading utilities
│   └── visualizer.py    # Visualization utilities
├── result/              # Generated results and plots
└── README.md            # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scipy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Asmit-Kumar/Object-Detection-from-Scratch-using-tensorflow
   cd Object-Detection-from-Scratch-using-tensorflow
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Dataset Generation
1. Place the MNIST dataset (`train.csv`) in the repository root directory.
2. Run the generation script:
   ```bash
   python GenerateDataset.py
   ```
   This will create `Images/` and `label/` directories.

### Training
Open `ObjectDetect.ipynb` in Jupyter Notebook or VS Code to train the model. The notebook covers:
- Data loading and preprocessing
- Model architecture definition
- Training loop
- Evaluation

### Evaluation & Inference
You can use the provided scripts to evaluate the model or run inference on new images.

**Run evaluation on test set:**
```bash
python evaluate_bbox.py
```

**Run the full pipeline (Detection + Recognition):**
```bash
python digit_detector.py
```

## 📊 Results

### Training Metrics
- **Loss**: Mean Squared Error (MSE) is used as the loss function.
- **Accuracy**: Intersection over Union (IoU) is often used to metric for object detection.

### Visualizations
The `result/` directory contains visualization of the model's predictions:

| Image                  | Visualization                      |
|------------------------|-------------------------------------|
| `result/1.png`         | ![1.png](result/1.png)             |
| `result/2.png`         | ![2.png](result/2.png)             |
| `result/3.png`         | ![3.png](result/3.png)             |
| `result/4.png`         | ![4.png](result/4.png)             |
| `result/5.png`         | ![5.png](result/5.png)             |

---

## Contributing

Contributions are welcome!  
If you have suggestions for improvement or want to fix any issues, feel free to fork the repository, create a branch, and submit a pull request.

---

## 👤 Author

- **Asmit Kumar** - [GitHub](https://github.com/Asmit-Kumar)
