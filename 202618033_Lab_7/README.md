# DS605 --- Lab Assignment 7

## Image Representation, PCA, Eigenfaces and Face Recognition

**Course:** DS605 --- Fundamentals of Machine Learning\
**Dataset:** AT&T Database of Faces (ORL)\
**Notebook:** `Assignment_7.ipynb`

## Overview

This assignment explores how face images can be represented numerically,
compressed with Principal Component Analysis (PCA), and used for face
recognition. It compares a NumPy-based nearest-neighbour baseline with
traditional machine-learning classifiers trained on PCA features.

The work is organized into the four parts specified in the assignment.

## Objectives

-   Convert grayscale face images into pixel matrices and feature
    vectors.
-   Create a reproducible, stratified training/testing split.
-   Implement PCA using NumPy linear-algebra operations.
-   Compare PCA directions obtained from covariance eigendecomposition
    and Singular Value Decomposition (SVD).
-   Determine the minimum number of components required to preserve at
    least 90% of the training variance.
-   Visualize eigenfaces and reconstruct face images from reduced
    representations.
-   Compare raw-pixel and PCA-based nearest-neighbour recognition.
-   Evaluate traditional classifiers using PCA features and different
    component counts.
-   Compare recognition accuracy and representation/storage
    requirements.

## Dataset

The notebook expects the AT&T ORL face dataset to contain 40 subject
folders, each with 10 face images (400 images in total). The expected
layout is:

``` text
project-folder/
├── Assignment_7.ipynb
├── README.md
└── archive7/
    ├── s1/
    │   ├── 1.pgm
    │   ├── 2.pgm
    │   └── ...
    ├── s2/
    └── ...
    └── s40/
```

Each subject folder is treated as a class label. If your dataset has a
different folder name or location, update `DATASET_PATH` in the
notebook. For a portable GitHub repository, prefer a relative path such
as:

``` python
DATASET_PATH = "archive7"
```

Make sure the dataset files are present at that path before running the
notebook. If the dataset cannot be redistributed with your repository,
include instructions explaining where to obtain it and where to place
it.

## Requirements

Python 3.9 or later is recommended. The notebook uses:

-   NumPy
-   pandas
-   Matplotlib
-   Pillow
-   scikit-learn
-   Jupyter Notebook or JupyterLab

Install the dependencies with:

``` bash
python -m pip install numpy pandas matplotlib pillow scikit-learn jupyter
```

## How to Run

1.  Clone or download this repository.
2.  Place the ORL dataset in the expected `archive7/` directory, or
    update `DATASET_PATH`.
3.  Install the required packages.
4.  Open `Assignment_7.ipynb` in Jupyter Notebook, JupyterLab, or VS
    Code.
5.  Select **Restart Kernel and Run All Cells**.
6.  Review the generated plots, printed metrics, and comparison tables.

The notebook uses a fixed random seed (`RANDOM_STATE = 42`) and a
stratified 80/20 train-test split so that the experiment can be
reproduced and each subject is represented in both sets.

## Methodology

### Part A --- Image Representation

-   Read face images as grayscale matrices.
-   Inspect image dimensions and pixel-intensity values.
-   Flatten each image into a one-dimensional feature vector.
-   Combine the vectors into a feature matrix and assign subject labels.
-   Split the data into training and test sets using stratification.

### Part B --- PCA, Eigenfaces and Compression

-   Calculate the mean face from the training data and center the
    training and test images using that mean.
-   Implement PCA through sample-space covariance eigendecomposition
    using NumPy.
-   Implement PCA using NumPy SVD and compare the resulting principal
    directions/eigenvalues.
-   Plot cumulative explained variance and find the minimum component
    count that preserves at least 90% of training variance.
-   Visualize principal components as eigenfaces.
-   Project faces into PCA space and reconstruct sample faces.
-   Compare reconstruction error and the storage required for raw pixels
    versus the reduced representation.

### Part C --- Recognition in PCA Space

-   Implement Euclidean-distance, 1-nearest-neighbour matching using
    NumPy.
-   Evaluate recognition using raw pixel vectors and PCA feature
    vectors.
-   Report test-set recognition accuracy.
-   Visualize selected correct and incorrect matches.
-   Explore how recognition changes with the number of retained PCA
    components.

### Part D --- Traditional Machine-Learning Classifiers

Traditional classifiers are trained using PCA features. The notebook
compares:

-   Support Vector Machine (RBF kernel)
-   Support Vector Machine (linear kernel)
-   K-Nearest Neighbours (KNN)
-   Logistic Regression

The experiment evaluates different valid PCA component counts and
compares classifier accuracy with the NumPy nearest-neighbour baseline.

## Important Implementation Requirement

PCA/eigenface calculations are implemented with **NumPy linear-algebra
operations**, not `sklearn.decomposition.PCA`, as required by the
assignment. Scikit-learn is used for the train-test split, evaluation
utilities, and final traditional machine-learning classifiers.

## Outputs and Results

Running the notebook produces visualizations and tables for:

-   Example grayscale faces and pixel statistics
-   Mean face
-   Covariance/eigendecomposition and SVD comparison
-   Cumulative explained variance and the 90% threshold
-   Eigenfaces
-   Original and reconstructed face images
-   Reconstruction error and storage comparison
-   Raw-pixel versus PCA nearest-neighbour accuracy
-   Correct and incorrect recognition matches
-   Classifier accuracy across PCA component counts
-   Traditional-classifier comparison at the 90%-variance component
    count

**Record the actual numerical results after running the notebook.**
Results depend on the dataset files and execution environment, so this
README intentionally does not claim fixed accuracy values or a fixed
number of components.

## Observations and Conclusion

After running all notebook cells, complete this section using the values
and plots generated by your run:

1.  **Variance retained:** State the minimum number of principal
    components required to preserve at least 90% of training variance.
2.  **PCA implementation comparison:** Describe whether covariance
    eigendecomposition and SVD produce comparable leading
    eigenvalues/principal directions.
3.  **Reconstruction quality:** Explain which facial structures remain
    visible and which details are lost when fewer components are used.
4.  **Compression:** Compare the storage required by the reduced
    representation with the original pixel representation.
5.  **Recognition performance:** Compare raw-pixel 1-NN, PCA 1-NN, and
    the traditional classifiers using the test-set results.
6.  **Component-count experiment:** Describe how recognition accuracy
    changes as more PCA components are retained and identify the best
    observed configuration.

## Repository Checklist

Before submitting, verify that the public GitHub repository contains:

-   [ ] `Assignment_7.ipynb` with cells that run from top to bottom
-   [ ] This `README.md`
-   [ ] Required explained-variance and eigenface figures
-   [ ] Face reconstruction examples
-   [ ] Recognition results and correct/incorrect match examples
-   [ ] A machine-learning comparison table
-   [ ] Dataset setup instructions
-   [ ] Completed observations based on actual notebook output

The assignment submission requirement is to submit **only the public
GitHub repository link**.

## Author

**Student:** Daksh Trivedi\
**Course:** DS605 --- Fundamentals of Machine Learning
