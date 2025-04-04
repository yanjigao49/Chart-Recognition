# 📊 Chart Recognition System

This project is a machine learning based pipeline for recognizing and extracting data from charts, including bar charts, line charts, scatter plots, and dot plots. It combines computer vision, object detection, and OCR to deliver structured, chart-specific insights.

---

## 🔍 Overview

The system is composed of five core components:

1. **Chart Type Classifier** – Classifies the input chart using a Vision Transformer.
2. **Axis Bounding Box Detector** – Detects axis regions and extracts values using Tesseract OCR.
3. **Chart Element Detection** – Identifies bars, dots, and lines using RCNN models.
4. **Tick Label Detector** – Detects tick label locations for both axes.
5. **Integration and Inference** – Merges outputs for final chart-specific analysis.

---

## 🧠 Part 1: Chart Type Classifier

- Vision Transformer model trained on balanced dataset.
- 80/10/10 train-validation-test split.
- Adam optimizer with early stopping.

---

## 📏 Part 2: Axis Detection + OCR

- RCNN model detects axis label regions.
- Tesseract OCR extracts text.
- Post-processing corrects numerical values and classifies them (categorical/numerical).

---

## 📊 Part 3: Chart Element Detection

- Bar chart detection: Separate RCNNs for vertical and horizontal bars.
- Line, dot, scatter: Bounding boxes created using ground truth dot locations.
- Center points extracted from predicted boxes.

---

## 📍 Part 4: Tick Label Detection

- Single RCNN model detects x and y tick labels.
- Points converted to bounding boxes for training and reversed during inference.

---

## 🔄 Part 5: Integration & Inference

- Axis and tick boxes paired using overlap heuristics.
- Correction algorithms infer missing elements.
- Chart-specific logic applied:
  - **Bar charts**: Value per pixel calculated from sample OCR.
  - **Line/Scatter**: Sample points used to interpolate values.
  - **Dot charts**: Dots grouped and counted based on proximity.

---

## ⚙️ Technologies

- PyTorch
- RCNN Models
- Tesseract OCR
- Vision Transformers
- NumPy / OpenCV
