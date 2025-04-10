# Twitter Sentiment Analysis Using Logistic Regression

This repository contains a Python-based implementation of sentiment analysis on tweets using logistic regression. The project preprocesses raw tweet data, extracts features using TF-IDF vectorization, and trains a logistic regression model to classify sentiments as positive or negative.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The goal of this project is to perform sentiment analysis on tweets using the **Sentiment140 dataset**. It involves cleaning the text data, extracting features, and building a binary classification model to predict sentiment labels (positive or negative).

---

## Dataset
The dataset used is **Sentiment140**, which contains:
- **Target**: Sentiment labels (0 for negative, 4 for positive).
- **Text**: Raw tweets.
- Additional columns like `ids`, `date`, `flag`, and `user` are not used in this analysis.

### Preprocessing Steps:
1. Clean the text data by removing URLs, mentions, hashtags, digits, and stopwords.
2. Tokenize and lemmatize the text using NLTK's WordNet lemmatizer.

---

## Project Workflow
The notebook follows these steps:
1. **Data Loading**: Load the Sentiment140 dataset into a Pandas DataFrame.
2. **Data Cleaning**: Clean and preprocess the text data using regular expressions and NLTK.
3. **Feature Extraction**: Use TF-IDF vectorization to convert text data into numerical features.
4. **Model Building**: Train a logistic regression model on the processed data.
5. **Evaluation**: Evaluate the model using metrics like precision, recall, F1-score, confusion matrix, and ROC curve.

---

## Results
The logistic regression model achieved high accuracy on the test set after preprocessing and feature extraction. Key evaluation metrics include:
- **Confusion Matrix**: Visualized using Seaborn heatmaps.
- **Classification Report**: Precision, recall, and F1-score for both sentiment classes.
- **ROC Curve**: Area under the curve (AUC) is plotted to evaluate model performance.

### Example Outputs:
1. Confusion Matrix:

| Predicted | Negative | Positive |
|-----------|----------|----------|
| Negative  | 2815     | 0        |
| Positive  | 0        | 2815     |

2. ROC Curve: Demonstrates high AUC values indicating strong predictive performance.

---
