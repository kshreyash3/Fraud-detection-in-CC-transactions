# Design a Deep Learning Model for Fraud Detection in Credit Card Transactions

This project explores the development of a deep learning model to detect fraudulent credit card transactions using the highly imbalanced Kaggle dataset. The primary goal was to maximize **recall** to catch as many fraudulent transactions as possible.

## Project Overview

The core challenge of this dataset is the extreme class imbalance: fraudulent transactions account for only 0.172% of the data. This makes "accuracy" a misleading metric.

To solve this, we implemented a supervised Multi-Layer Perceptron (MLP) model using Keras. The class imbalance was addressed using **SMOTE (Synthetic Minority Over-sampling Technique)**, which balances the training set by creating synthetic examples of the minority fraud class.

## Final Model Results

The selected model was evaluated on the original, imbalanced test set.

* **Primary Metric: Recall (Fraud)**: **85.71%**
    * This is the most important result. It means the model successfully identified **84 out of the 98** fraudulent transactions.
* **Precision (Fraud)**: **39.62%**
    * This is the expected trade-off. To catch 84 frauds, the model incorrectly flagged 128 legitimate transactions.
* **Overall Accuracy**: **99.75%**
    * This metric is misleading and was not the primary goal.

### Confusion Matrix

| | Predicted: Not Fraud | Predicted: Fraud |
| :--- | :--- | :--- |
| **Actual: Not Fraud** | 56736 (True Negative) | 128 (False Positive) |
| **Actual: Fraud** | 14 (False Negative) | 84 (True Positive) |

## Repository Files

* `DL Project Group 55.ipynb`: The Google Colab notebook containing all code for data preprocessing, model building, training, and evaluation.
* `DL Report Team 55.pdf`: The final, detailed project report in PDF format.
* `credit_fraud_model.keras`: The saved, trained Keras model file for the selected MLP.
* `README.md`: This file.

## How to Run

1.  **Environment:** This project is designed to run in Google Colab.
2.  **Data:** Download the `creditcard.csv` dataset from [this Kaggle competition](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3.  **Upload:** Upload the `creditcard.csv` file to your Google Drive.
4.  **Open Notebook:** Open `DL Project Group 55.ipynb` in Google Colab.
5.  **Run Cells:**
    * Mount your Google Drive when prompted.
    * Update the `file_path` variable to point to the `creditcard.csv` file in your Drive.
    * Run all cells sequentially to preprocess the data, train the model, and see the final evaluation.
