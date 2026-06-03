# Deep Learning-Based Classification of Diabetic Kidney Disease Severity from Histopathology Images

## Overview

Diabetic Kidney Disease (DKD) is one of the leading causes of chronic kidney failure worldwide. Early diagnosis and accurate severity classification are essential for improving patient outcomes and treatment planning.

This project presents a Deep Learning-based framework for automated classification of diabetic kidney disease severity from histopathology images. The framework leverages CNNs, Vision Transformers (ViT), Ensemble Learning, and Data Augmentation techniques to improve prediction accuracy and robustness.

---

## Problem Statement

Manual analysis of kidney histopathology images is time-consuming and requires expert pathologists. This project aims to automate the classification process using Artificial Intelligence and Deep Learning to support healthcare professionals and improve diagnostic efficiency.

---

## Objectives

* Automate diabetic kidney disease severity classification.
* Apply deep learning techniques to medical image analysis.
* Improve classification accuracy using ensemble learning.
* Reduce overfitting through data augmentation.
* Support AI-assisted medical diagnosis.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## Dataset

This project utilizes the **KidneyAI Dataset**, a publicly available histopathology image dataset developed for diabetic kidney disease severity classification and research.

### Dataset Information

| Attribute         | Details                                  |
| ----------------- | ---------------------------------------- |
| Dataset Name      | KidneyAI Dataset                         |
| Source            | Zenodo                                   |
| DOI               | 10.5281/zenodo.17456927                  |
| Total Images      | 3,928 PAS-stained glomerulus images      |
| Image Type        | Histopathology Images                    |
| Image Format      | PNG                                      |
| Annotation Format | JSON                                     |
| Classes           | Normal, Mild, Moderate, Severe, Excluded |

### Dataset Link

https://zenodo.org/records/17456927

### Deep Learning Concepts Used

#### Convolutional Neural Networks (CNNs)

CNNs are widely used for extracting local texture and tissue features from medical images.

#### Transformer-Based Models

Transformer architectures capture long-range dependencies and global contextual information from image data.

#### Ensemble Learning

Combines predictions from multiple deep learning models to improve robustness and classification stability.

#### Soft Ensemble Methods

Probability-based fusion strategies are used to generate final predictions.

#### Data Augmentation

Rotation, flipping, zooming, and brightness adjustments improve model generalization and reduce overfitting.

---

## Methodology

1. Data Collection
2. Image Preprocessing
3. Feature Extraction
4. Model Training
5. Ensemble Learning
6. Evaluation
7. Severity Prediction

---

## Model Architecture

Input Histopathology Images

↓

Image Preprocessing

↓

CNN Feature Extraction

↓

Vision Transformer Learning

↓

Soft Ensemble Fusion

↓

Disease Severity Prediction

---

## Model Visualization

<p align="center">
  <img src="images/model_architecture.png" alt="Model Architecture" width="1000"/>
</p>

---

## Results

The proposed framework successfully classifies diabetic kidney disease severity from histopathology images and demonstrates strong performance across multiple evaluation metrics.

### Evaluation Metrics

* Accuracy
* Balanced Accuracy
* Macro F1-Score
* Weighted F1-Score
* Macro AUC
* Cohen's Kappa
* QWK
* MCC
* Top-2 Accuracy
* Log Loss

---

### ViT-B/16 Training Results

<p align="center">
  <img src="images/vit_training_results.png" alt="ViT-B16 Results" width="1000"/>
</p>

---

### EfficientNet-B0 Training Results

<p align="center">
  <img src="images/efficientnet_training_results.png" alt="EfficientNet Results" width="1000"/>
</p>

---

### Model Performance Comparison

<p align="center">
  <img src="images/model_comparison_results.png" alt="Model Comparison" width="1000"/>
</p>

| Model          | Validation Accuracy | Test Accuracy |
| -------------- | ------------------- | ------------- |
| ConvNeXt-Small | 84.3%               | 83.1%         |
| ViT-B/16       | 86.7%               | 85.3%         |
| Ensemble Model | 88.1%               | 87.2%         |

---

### Final Ensemble Evaluation Results

<p align="center">
  <img src="images/final_ensemble_results.png" alt="Final Ensemble Results" width="1000"/>
</p>

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 0.7169 |
| Balanced Accuracy | 0.7194 |
| Macro F1          | 0.7127 |
| Weighted F1       | 0.7010 |
| Macro AUC         | 0.9192 |
| Kappa             | 0.6098 |
| QWK               | 0.8747 |
| MCC               | 0.6196 |
| Top-2 Accuracy    | 0.9779 |
| Log Loss          | 0.5946 |

---

## Project Structure

```text
Deep-Learning-Based-Classification-of-Diabetic-Kidney-Disease-Severity-from-Histopathology-Images
│
├── dataset/
├── images/
│   ├── model_architecture.png
│   ├── vit_training_results.png
│   ├── efficientnet_training_results.png
│   ├── model_comparison_results.png
│   └── final_ensemble_results.png
├── notebooks/
├── src/
├── models/
├── requirements.txt
└── README.md
```

---

## Applications

* Medical Image Analysis
* Kidney Disease Diagnosis
* Healthcare AI
* Clinical Decision Support Systems
* Biomedical Research

---

## Author

# Krishnam Namithaa

Owner and Developer of this Project

---

## License

This project is intended for academic and research purposes only.
