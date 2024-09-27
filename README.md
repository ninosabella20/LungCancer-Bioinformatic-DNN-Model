# Lung Cancer Prediction using Deep Neural Networks

This project implements a Deep Neural Network (DNN) model to predict lung cancer based on gene expression data. The dataset used is sourced from Kaggle, which contains microarray gene expression data for lung cancer patients.

## Dataset

The dataset can be found at the following link: [Lung Cancer Patients mRNA Microarray](https://www.kaggle.com/datasets/josemauricioneuro/lung-cancer-patients-mrna-microarray).

### Data Overview

- The dataset includes various gene expression levels represented as features.
- The target variable indicates whether the patient has lung cancer (1) or not (0).

## Model Overview

The DNN model is built using Keras and TensorFlow. It consists of several layers to learn the complex patterns in the gene expression data. Overfitting issues have been addressed to improve the model's generalization capabilities.

### Performance Metrics

- **Training Accuracy**: 0.8535
- **Validation Accuracy**: 0.7407
- **Test Accuracy**: 0.79
- **Training Loss**: 1.2733
- **Validation Loss**: 1.5401
- **AUC-ROC (Test)**: 0.8357

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 0.84      | 0.80   | 0.82     | 20      |
| 1.0   | 0.73      | 0.79   | 0.76     | 14      |
| **Accuracy** |       |        | **0.79** | 34      |
| **Macro Avg** | 0.79  | 0.79   | 0.79     | 34      |
| **Weighted Avg** | 0.80 | 0.79 | 0.80   | 34      |

### Graphical Representation

The model's accuracy and loss during training and validation have been graphically represented in the notebook. These plots provide insights into the model's performance over the training epochs.

- **Accuracy Plot**: Shows the training and validation accuracy over epochs.
- **Loss Plot**: Shows the training and validation loss over epochs.

### Training

The model is trained using a binary cross-entropy loss function and Adam optimizer. The training process involves splitting the dataset into training, validation, and test sets. Overfitting issues were addressed to improve model performance.

### Evaluation

The model is evaluated using accuracy, precision, recall, F1-score, and AUC-ROC metrics. The improvements made to address overfitting have resulted in better validation performance compared to earlier iterations.

## Conclusion

This project demonstrates the application of Deep Learning techniques to predict lung cancer based on genetic data. The model has shown significant improvement in accuracy and loss metrics, with an AUC-ROC of 0.8357, indicating enhanced discriminative power. Further improvements can be made by optimizing hyperparameters, experimenting with more complex architectures, and increasing the dataset size.

For more details, refer to the Jupyter notebook included in this project.

### Author

Nino Sabella
