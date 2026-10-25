# DS605 Lab Assignment 8: Decision Trees and SVMs

## Overview

This project implements and compares Decision Tree and Support Vector Machine (SVM) algorithms using the UCI Banknote Authentication dataset.

## Dataset

**Dataset:** Banknote Authentication  
**Source:** UCI Machine Learning Repository  
**Samples:** 1,372  
**Features:** 4  
**Classes:** 2

The dataset contains the following features:
- Variance of Wavelet Transformed Image
- Skewness of Wavelet Transformed Image
- Curtosis of Wavelet Transformed Image
- Entropy of Image

## Assignment Tasks

### Part A: Decision Tree from Scratch
- Implement a Decision Tree classifier using NumPy and Pandas.
- Implement Gini impurity and entropy.
- Find optimal split thresholds.
- Implement recursive tree construction and prediction.
- Visualize decision boundaries using two features.
- Compare different splitting criteria and tree depths with Scikit-learn.

### Part B: Linear SVM from Scratch
- Implement a linear soft-margin SVM using gradient-based optimization.
- Apply feature scaling and convert class labels to -1 and +1.
- Implement hinge loss and its subgradient.
- Visualize decision boundaries, margins, and margin violations.
- Compare the implementation with Scikit-learn.
- Study the effects of C, learning rate, and iterations.

### Part C: Nonlinear SVM Kernels
- Generate nonlinear datasets using `make_moons` and `make_circles`.
- Train SVM classifiers with linear, polynomial, and RBF kernels.
- Visualize and compare their decision boundaries.

### Part D: Model Comparison
Compare the following models:
- Decision Tree implemented from scratch
- Scikit-learn Decision Tree
- Linear SVM implemented from scratch
- Scikit-learn Linear SVM
- Scikit-learn RBF SVM

Evaluate the models using accuracy, precision, recall, F1-score, training time, and prediction time.

## Requirements

- Python 3
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation and Execution

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Open the notebook:

```bash
jupyter notebook 202618033Lab08.ipynb
```