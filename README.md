# X-Ray DL Project

## Overview

This project investigates the use of deep learning for chest X-ray analysis using a multi-task learning approach. The model will use chest X-ray images as input and predict both disease labels and patient metadata such as patient gender.

The goal of this project is to explore whether jointly learning disease classification and metadata prediction improves the model’s learned image representations compared to standard disease-only classification approaches.

## Dataset

This project uses the NIH Chest X-ray dataset:

https://www.kaggle.com/datasets/nih-chest-xrays/data

For this milestone, a few sample X-ray images and labels are included in the repository under:

```text
data/sample_images/
```

and

```text
data/sample_labels.csv
```

---


## Repository Structure

```text
X-ray-DL-project/
├── notebooks/
│   └── data_demo.ipynb
├── data/
│   ├── sample_images/
│   └── sample_labels.csv
├── src/
│   ├── my_dl_project/
│   └── xray_project/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/suhas-makineni/X-ray-DL-project.git
cd X-ray-DL-project
```

Create a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

---

## Data Loader Demo

The notebook below demonstrates:
- loading chest X-ray images
- loading disease labels
- loading metadata labels
- visualizing sample images

Notebook location:

```text
notebooks/data_demo.ipynb
```

---

## Project Goal

Many existing chest X-ray projects focus only on disease classification. This project explores a multi-task learning approach where the model predicts both disease labels and patient metadata simultaneously.

The final model will likely use a pretrained CNN architecture such as ResNet or MobileNet implemented in PyTorch.

---

## Evaluation

The project will evaluate:
- disease classification accuracy
- F1 score
- metadata prediction accuracy
- comparison between single-task and multi-task learning performance
