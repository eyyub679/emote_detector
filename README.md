# Real-Time Facial Emotion Detector

A computer vision application combining **OpenCV** face tracking with a custom **PyTorch Convolutional Neural Network (CNN)** to perform real-time facial expression and emotion recognition from a live webcam feed.

---

## Overview

The pipeline captures frames from a video stream, detects facial bounding boxes with Haar Cascade classifiers, preprocesses facial crops (48×48 grayscale tensor), and predicts emotional states across 7 categories:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Sad**
- **Surprise**
- **Neutral**

---

## Model Architecture

The custom deep CNN (`Emote_detector`) is structured as follows:
- **Block 1:** Conv2D (1 $\rightarrow$ 64, 3x3) + BatchNorm + ReLU $\rightarrow$ Conv2D (64 $\rightarrow$ 64) + BatchNorm + ReLU $\rightarrow$ MaxPool2d (2x2)
- **Block 2:** Conv2D (64 $\rightarrow$ 128, 3x3) + BatchNorm + ReLU $\rightarrow$ Conv2D (128 $\rightarrow$ 128) + BatchNorm + ReLU $\rightarrow$ MaxPool2d (2x2) + Dropout2d (0.1)
- **Block 3:** Conv2D (128 $\rightarrow$ 256, 3x3) + BatchNorm + ReLU $\rightarrow$ Conv2D (256 $\rightarrow$ 256) + BatchNorm + ReLU $\rightarrow$ MaxPool2d (2x2) + Dropout2d (0.1)
- **Classifier Head:** Flatten $\rightarrow$ Linear $\rightarrow$ Dropout $\rightarrow$ Linear (7 output logits)

Weights are saved in `model_params/emotion_detector.pth`.

---

## Getting Started

### 1. Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/eyyub679/emote_detector.git
cd emote_detector
uv sync
```

Or with `pip`:
```bash
pip install -r requirements.txt # or pip install opencv-python torch torchvision
```

### 2. Run the Real-Time Webcam Demo

```bash
python main.py
```

* Ensure your webcam is connected.
* Bounding boxes with predicted emotions will appear in real time over each detected face.
* Press `q` to quit the application window.

### 3. Model Training & Notebooks
- `notebookeaf70af481.ipynb`: Full training notebook including dataset loading, data augmentation (`torchvision.transforms.v2`), training loops, and validation metrics.
- `notebook_code.py`: Standalone Python script version of the training and evaluation workflow.
