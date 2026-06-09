# X-Ray DL Project

## Overview

This project investigates the use of deep learning for chest X-ray analysis using a multi-task learning approach. Rather than only predicting disease labels from chest X-ray images, the model simultaneously predicts disease labels and patient metadata (gender). The motivation behind this approach is to determine whether jointly learning multiple related tasks improves feature learning and overall model performance compared to traditional disease-only classification models.

The project was implemented in PyTorch using a ResNet18-based architecture with separate prediction heads for disease classification and gender classification.

## Dataset

This project uses the NIH Chest X-ray dataset:

https://www.kaggle.com/datasets/nih-chest-xrays/data

The full NIH dataset contains 112,120 chest X-ray images and associated metadata. For this project, a subset of 4,999 images was used for experimentation and model development.

Metadata used from the dataset includes:

* Disease labels
* Patient gender
* Patient age
* View position

The disease classification task was treated as a multi-label classification problem using the following disease categories:

* Atelectasis
* Cardiomegaly
* Effusion
* Infiltration
* Mass
* Nodule
* Pneumonia
* Pneumothorax
* Consolidation
* Edema
* Emphysema
* Fibrosis
* Pleural Thickening
* Hernia

## Model Architecture

The model uses a ResNet18 backbone as a shared feature extractor.

The shared image representation is then passed into two prediction heads:

1. Disease Classification Head

   * Multi-label disease prediction
   * BCEWithLogitsLoss

2. Gender Classification Head

   * Binary gender classification
   * CrossEntropyLoss

The total training loss is computed as the sum of the disease loss and gender classification loss.

## Training

The dataset was split into:

* 80% Training
* 20% Validation

Training was performed using:

* Optimizer: Adam
* Learning Rate: 0.0001
* Batch Size: 32
* Epochs: 4
* Hardware: NVIDIA GPU (CUDA)

The training script is located at:

src/xray_project/train_model.py

## Results

Training loss consistently decreased throughout training:

| Epoch | Training Loss |
|---------|---------|
| 1 | 0.7841 |
| 2 | 0.5386 |
| 3 | 0.4299 |
| 4 | 0.3625 |

Final validation metrics:

| Metric | Value |
|---------|---------|
| Validation Loss | 0.4499 |
| Gender Accuracy | 88.8% |
| Precision | 0.89 |
| Recall | 0.89 |
| F1 Score | 0.89 |

### Gender Classification Report

| Class | Precision | Recall | F1 Score |
|---------|---------|---------|---------|
| Female | 0.87 | 0.89 | 0.88 |
| Male | 0.90 | 0.88 | 0.89 |

### Confusion Matrix

| Actual \\ Predicted | Female | Male |
|---------|---------|---------|
| Female | 419 | 51 |
| Male | 61 | 469 |

### Baseline Comparison

The dataset was relatively balanced, containing approximately:

- Male: 52.2%
- Female: 47.8%

A naive classifier predicting only the majority class would achieve approximately 52.2% accuracy. The multitask model achieved 88.8% validation accuracy, substantially outperforming the baseline.
## Evaluation

Evaluation was performed using:

* Validation Loss
* Gender Classification Accuracy
* Disease Label Predictions
* Dataset Distribution Analysis

The evaluation notebook is located at:

notebooks/evaluation.ipynb

The notebook includes:

* Dataset exploration
* Disease frequency visualization
* Gender distribution visualization
* Sample image inspection
* Model evaluation

## Limitations

Several limitations exist in the current implementation:

* Only a subset of the NIH dataset was used.
* Disease classification performance was not evaluated with full precision, recall, F1-score, and AUC metrics.
* Only gender metadata was included in the multi-task learning framework.
* Additional metadata such as patient age and view position could be incorporated in future work.

## Future Work

Potential future improvements include:

* Training on the complete NIH dataset.
* Using larger backbone architectures such as ResNet50 or EfficientNet.
* Including additional patient metadata.
* Comparing performance against a disease-only baseline model.
* Computing more comprehensive disease classification metrics.

## Repository Structure

* notebooks/data_demo.ipynb
* notebooks/evaluation.ipynb
* src/xray_project/models.py
* src/xray_project/train_model.py
* src/xray_project/dataset/xray_dataset.py
* outputs/xray_multitask_model.pt
* outputs/training_loss.png

## Conclusion

This project demonstrates that a multi-task learning framework can be successfully applied to chest X-ray analysis. The model learned both disease-related information and patient metadata simultaneously, achieving a validation loss of 0.4499 and a gender classification accuracy of 88.8% on the validation set. Future work can extend this framework using larger datasets and more advanced model architectures.
