# Lung Cancer Prediction using Deep Neural Networks

This project implements a Deep Neural Network (DNN) model to predict lung cancer based on gene expression data. The dataset used is sourced from Kaggle, which contains microarray gene expression data for lung cancer patients. 

## Dataset

The dataset can be found at the following link: [Lung Cancer Patients mRNA Microarray](https://www.kaggle.com/datasets/josemauricioneuro/lung-cancer-patients-mrna-microarray).

### Data Overview

- The dataset includes various gene expression levels represented as features.
- The target variable indicates whether the patient has lung cancer (1) or not (0).

## Model Overview

The DNN model is built using Keras and TensorFlow. It consists of several layers to learn the complex patterns in the gene expression data.

### Performance Metrics

- **Training Accuracy**: 0.95
- **Validation Accuracy**: 0.63
- **Test Accuracy**: 0.68
- **Training Loss**: 1.00
- **Validation Loss**: 1.71
- **AUC-ROC (Test)**: 0.73

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 0.76      | 0.65   | 0.70     | 20      |
| 1.0   | 0.59      | 0.71   | 0.65     | 14      |
| **Accuracy** |       |        | **0.68** | 34      |
| **Macro Avg** | 0.68  | 0.68   | 0.67     | 34      |
| **Weighted Avg** | 0.69 | 0.68 | 0.68   | 34      |

### Graphical Representation

The model's accuracy and loss during training and validation have been graphically represented in the notebook. These plots provide insights into the model's performance over the training epochs.

- **Accuracy Plot**: Shows the training and validation accuracy over epochs.
- **Loss Plot**: Shows the training and validation loss over epochs.

### Conclusion

This project demonstrates the application of Deep Learning techniques to predict lung cancer based on genetic data. While the model shows promise, further improvements can be made by optimizing hyperparameters, experimenting with more complex architectures, and increasing the dataset size.

For more details, refer to the Jupyter notebook included in this project.
