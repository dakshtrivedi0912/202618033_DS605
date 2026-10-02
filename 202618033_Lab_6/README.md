# DS605 Lab 6 – Image and Text Feature Extraction and Classification

## Feature Extraction and Machine Learning with Image and Text Data

This repository contains the implementation of **DS605 Lab Assignment 6 – Feature Extraction and Machine Learning with Image and Text Data**.

The objective of this assignment is to convert raw image and text data into numerical feature representations and use traditional machine-learning models for classification.

The assignment is divided into three parts:

- **Part A:** Image Feature Extraction and Asphalt Crack Classification
- **Part B:** Text Vectorization and Email Spam Classification
- **Part C:** Image Representation Improvement

---

# Datasets

## 1. Asphalt Crack Dataset

The image dataset contains **400 asphalt images** divided into two classes:

- 200 Crack images
- 200 Non-Crack images

The original images have dimensions:

```text
448 × 448 × 3
```

For feature extraction, all images were resized to:

```text
224 × 224
```

The images were read and processed using **OpenCV**.

### Image Labels

```text
0 = Non-Crack
1 = Crack
```

---

## 2. Email Spam Dataset

The email dataset contains a total of **5,172 emails**.

The class distribution is:

| Class | Number of Emails | Percentage |
|---|---:|---:|
| Non-Spam (0) | 3,672 | 71.00% |
| Spam (1) | 1,500 | 29.00% |

The supplied email dataset contains word-frequency/count columns rather than a single raw email-text column.

Text representations were reconstructed from the supplied word counts and then processed using **CountVectorizer**.

### Email Labels

```text
0 = Non-Spam
1 = Spam
```

---

# Part A – Image Feature Extraction and Classification

## Image Preprocessing

The following preprocessing steps were performed on the asphalt images:

1. Images were read using OpenCV with `cv2.imread()`.
2. Image dimensions and channels were inspected.
3. Images were resized from `448 × 448` to `224 × 224`.
4. Images were converted from BGR to grayscale.
5. Numerical intensity-based features were extracted using NumPy.
6. Canny Edge Detection was applied using OpenCV.
7. Edge-based numerical features were extracted.
8. One feature row was created for every image.

---

## Grayscale Conversion

The resized images were converted to grayscale using:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Grayscale images contain a single intensity value for each pixel ranging from:

```text
0   = Black
255 = White
```

Using grayscale simplifies feature extraction because intensity statistics can be calculated without processing three separate colour channels.

---

# Image Feature Extraction

Nine numerical features were extracted from every asphalt image.

## 1. Mean Brightness

Mean brightness represents the average pixel intensity of an image.

```python
mean_brightness = np.mean(gray)
```

A lower value indicates an overall darker image, while a higher value indicates a brighter image.

---

## 2. Contrast

Contrast was measured using the standard deviation of pixel intensities.

```python
contrast = np.std(gray)
```

A higher standard deviation indicates greater variation between dark and bright areas in the image.

---

## 3. Minimum Intensity

The minimum pixel intensity represents the darkest pixel in the image.

```python
min_intensity = np.min(gray)
```

---

## 4. Maximum Intensity

The maximum pixel intensity represents the brightest pixel in the image.

```python
max_intensity = np.max(gray)
```

---

## 5. Median Intensity

The median represents the middle pixel-intensity value after sorting all pixel values.

```python
median_intensity = np.median(gray)
```

---

## 6. Dark Pixel Ratio

Pixels having an intensity below 50 were considered dark pixels.

```python
dark_pixel_ratio = np.sum(gray < 50) / gray.size
```

The dark-pixel ratio represents the proportion of very dark pixels in the image.

---

## 7. Bright Pixel Ratio

Pixels having an intensity greater than 200 were considered bright pixels.

```python
bright_pixel_ratio = np.sum(gray > 200) / gray.size
```

The bright-pixel ratio represents the proportion of very bright pixels in the image.

---

# Canny Edge Detection

Canny Edge Detection was used to identify strong intensity boundaries in the asphalt images.

The original Canny configuration was:

```python
edges = cv2.Canny(gray, 100, 200)
```

In the resulting Canny image:

```text
0   = Non-edge pixel
255 = Edge pixel
```

Two additional numerical features were extracted from the Canny representation.

---

## 8. Edge Count

Edge count represents the total number of pixels identified as edges.

```python
edge_count = np.count_nonzero(edges)
```

---

## 9. Edge Density

Edge density represents the proportion of edge pixels in the complete image.

```python
edge_density = edge_count / edges.size
```

---

# Extracted Image Feature Dataset

After processing all images, the final image-feature dataset contained:

```text
400 rows
11 columns
```

The columns consist of:

- Filename
- Mean Brightness
- Contrast
- Minimum Intensity
- Maximum Intensity
- Median Intensity
- Dark Pixel Ratio
- Bright Pixel Ratio
- Edge Count
- Edge Density
- Label

The extracted feature table was saved as:

```text
image_features.csv
```

---

# Train-Test Split

The image dataset was divided into training and testing sets using:

```text
Training Data = 80%
Testing Data  = 20%
```

The split was performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This resulted in:

```text
Training Samples = 320
Testing Samples  = 80
```

Stratification was used to maintain the same Crack/Non-Crack class distribution in both sets.

---

# Feature Scaling

`StandardScaler` was used for Logistic Regression.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data to prevent data leakage.

Random Forest was trained using the original numerical features because tree-based models do not require feature scaling.

---

# Part A Models

Two traditional machine-learning classifiers were used:

1. Logistic Regression
2. Random Forest

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Time
- Prediction Time

---

# Logistic Regression Results – Image Classification

The Logistic Regression model produced the following results:

```text
Accuracy:       0.9375
Precision:      0.9268
Recall:         0.9500
F1-Score:       0.9383
Training Time:  ~0.0083 seconds
Prediction Time: ~0.00039 seconds
```

### Confusion Matrix

```text
[[37  3]
 [ 2 38]]
```

This means:

- 37 Non-Crack images were correctly classified.
- 3 Non-Crack images were incorrectly classified as Crack.
- 2 Crack images were incorrectly classified as Non-Crack.
- 38 Crack images were correctly classified.

The model correctly classified:

```text
75 out of 80 test images
```

giving an accuracy of:

```text
93.75%
```

---

# Random Forest Results – Image Classification

The Random Forest model produced:

```text
Accuracy:       0.9500
Precision:      0.9500
Recall:         0.9500
F1-Score:       0.9500
Training Time:  ~0.1752 seconds
Prediction Time: ~0.0071 seconds
```

### Confusion Matrix

```text
[[38  2]
 [ 2 38]]
```

This means:

- 38 Non-Crack images were correctly classified.
- 2 Non-Crack images were incorrectly classified as Crack.
- 2 Crack images were incorrectly classified as Non-Crack.
- 38 Crack images were correctly classified.

The model correctly classified:

```text
76 out of 80 test images
```

giving an accuracy of:

```text
95%
```

---

# Part A Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 93.75% | 95.00% |
| Precision | 92.68% | 95.00% |
| Recall | 95.00% | 95.00% |
| F1-Score | 93.83% | 95.00% |
| Training Time | ~0.0083 s | ~0.1752 s |
| Prediction Time | ~0.00039 s | ~0.0071 s |

## Part A Observation

The handcrafted intensity and edge-based features were effective for asphalt crack classification.

Logistic Regression achieved an accuracy of **93.75%**, while Random Forest achieved **95% accuracy**.

Random Forest achieved slightly higher accuracy, precision, and F1-score. However, Logistic Regression required considerably less training and prediction time.

Therefore, the experiment demonstrates a trade-off between predictive performance and computational cost.

---

# Part B – Text Vectorization and Email Spam Classification

## Dataset Inspection

The email dataset contains:

```text
Total Emails = 5,172
```

The class distribution is:

```text
Non-Spam = 3,672
Spam     = 1,500
```

Percentage distribution:

```text
Non-Spam = 71.00%
Spam     = 29.00%
```

Therefore, unlike the image dataset, the email dataset is not perfectly balanced.

---

# Text Representation

The supplied email dataset already contains approximately 3,000 word-frequency columns.

Since the dataset does not contain the original raw email sentences, a text representation was reconstructed using the supplied word counts.

For example, if the supplied representation contained:

```text
free = 2
offer = 1
meeting = 0
```

the reconstructed representation would contain:

```text
free free offer
```

This preserves word-frequency information, although it does not recover the original order of words.

---

# CountVectorizer

The reconstructed email text was converted into numerical vectors using:

```python
CountVectorizer()
```

The train-test split was performed before fitting CountVectorizer to avoid data leakage.

The training data was processed using:

```python
X_train_count = count_vectorizer.fit_transform(X_train_text)
```

The testing data was processed using:

```python
X_test_count = count_vectorizer.transform(X_test_text)
```

The resulting matrices were:

```text
Training Matrix Shape: (4137, 2974)
Testing Matrix Shape:  (1035, 2974)
Number of Features:    2974
```

Although the supplied dataset contained approximately 3,000 word columns, CountVectorizer generated **2,974 features** from the training data.

This occurred because the vocabulary was learned only from the training portion of the dataset.

---

# Part B Models

Two traditional classification algorithms were used:

1. Logistic Regression
2. Multinomial Naive Bayes

Multinomial Naive Bayes was selected because it is suitable for discrete count-based text features.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Time
- Prediction Time

---

# Logistic Regression Results – Spam Classification

The Logistic Regression model achieved:

```text
Accuracy:        0.9816
Precision:       0.9547
Recall:          0.9833
F1-Score:        0.9688
Training Time:   ~0.629 seconds
Prediction Time: ~0.0040 seconds
```

### Confusion Matrix

```text
[[721  14]
 [  5 295]]
```

This means:

- 721 Non-Spam emails were correctly classified.
- 14 Non-Spam emails were incorrectly classified as Spam.
- 5 Spam emails were incorrectly classified as Non-Spam.
- 295 Spam emails were correctly classified.

The model correctly classified:

```text
1016 out of 1035 test emails
```

Logistic Regression achieved an accuracy of approximately:

```text
98.16%
```

---

# Multinomial Naive Bayes Results

The Multinomial Naive Bayes model achieved:

```text
Accuracy:        0.9420
Precision:       0.8681
Recall:          0.9433
F1-Score:        0.9042
Training Time:   ~0.053 seconds
Prediction Time: ~0.0049 seconds
```

### Confusion Matrix

```text
[[692  43]
 [ 17 283]]
```

This means:

- 692 Non-Spam emails were correctly classified.
- 43 Non-Spam emails were incorrectly classified as Spam.
- 17 Spam emails were incorrectly classified as Non-Spam.
- 283 Spam emails were correctly classified.

The model correctly classified:

```text
975 out of 1035 test emails
```

---

# Part B Model Comparison

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 98.16% | 94.20% |
| Precision | 95.47% | 86.81% |
| Recall | 98.33% | 94.33% |
| F1-Score | 96.88% | 90.42% |
| Training Time | ~0.629 s | ~0.053 s |
| Prediction Time | ~0.0040 s | ~0.0049 s |
| Number of Features | 2,974 | 2,974 |

## Part B Observation

Logistic Regression achieved stronger predictive performance than Multinomial Naive Bayes on this train-test split.

Logistic Regression achieved:

```text
98.16% Accuracy
95.47% Precision
98.33% Recall
96.88% F1-Score
```

It also produced fewer false-positive and false-negative predictions.

Multinomial Naive Bayes achieved lower predictive performance but required substantially less training time.

Therefore, Logistic Regression provided stronger classification performance, while Multinomial Naive Bayes demonstrated greater training efficiency.

---

# Part C – Improving the Image Representation

Part C investigates whether changing the image preprocessing and Canny edge representation can improve the asphalt crack classification approach.

Random Forest was used for the comparison because it achieved the strongest classification performance in Part A.

---

# Original Image Representation

The original approach applied Canny Edge Detection directly to the grayscale image:

```python
edges = cv2.Canny(gray, 100, 200)
```

The original Canny thresholds were:

```text
Lower Threshold = 100
Upper Threshold = 200
```

The original representation contained nine numerical image features.

---

# Modified Image Representation

In the modified representation, Gaussian Blur was applied before Canny Edge Detection.

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)
```

The modified Canny thresholds were:

```text
Lower Threshold = 50
Upper Threshold = 150
```

Gaussian Blur was introduced to reduce small texture-related intensity variations before performing edge detection.

The purpose was to investigate whether reducing small asphalt-texture edges would produce a more useful edge representation for crack classification.

---

# Part C Results

| Approach | Canny Method | Number of Features | Accuracy | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|---:|
| Original Random Forest | Grayscale, thresholds 100–200 | 9 | 95.00% | 95.00% | 95.00% | 95.00% |
| Modified Random Forest | Gaussian Blur, thresholds 50–150 | 9 | 95.00% | 95.00% | 95.00% | 95.00% |

Measured execution times were:

| Approach | Training Time | Prediction Time |
|---|---:|---:|
| Original Representation | ~0.1752 s | ~0.0071 s |
| Modified Representation | ~0.1915 s | ~0.0122 s |

---

# Part C Observation

The modified image representation produced the same classification performance as the original representation.

Both approaches achieved:

```text
Accuracy  = 95%
Precision = 95%
Recall    = 95%
F1-Score  = 95%
```

Therefore, applying Gaussian Blur and changing the Canny thresholds from `100–200` to `50–150` did not improve predictive performance on this train-test split.

The modified approach also showed slightly higher measured training and prediction times. However, the execution times are very small and may vary between runs depending on system conditions.

The experiment demonstrates that additional preprocessing does not necessarily improve machine-learning performance.

In this case, the simpler original representation achieved the same predictive performance as the modified representation.

---

# Overall Results

The main results obtained from the assignment are summarized below.

## Image Classification

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 93.75% | 93.83% |
| Random Forest | 95.00% | 95.00% |

Random Forest achieved the strongest image-classification performance with **95% accuracy**.

---

## Email Spam Classification

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 98.16% | 96.88% |
| Multinomial Naive Bayes | 94.20% | 90.42% |

Logistic Regression achieved the strongest spam-classification performance with **98.16% accuracy**.

---

# Key Findings

1. Raw images can be converted into useful numerical representations using handcrafted intensity and edge-based features.

2. Mean brightness, contrast, dark-pixel ratio, bright-pixel ratio, edge count, and edge density provide useful information for asphalt crack classification.

3. Random Forest achieved **95% accuracy** for asphalt crack classification.

4. Logistic Regression was significantly faster than Random Forest for the image dataset while producing slightly lower classification performance.

5. The email dataset contained approximately **71% Non-Spam and 29% Spam** emails.

6. CountVectorizer generated **2,974 numerical text features** from the training data.

7. Logistic Regression achieved **98.16% accuracy** for email spam classification.

8. Multinomial Naive Bayes trained much faster than Logistic Regression but produced lower predictive performance.

9. Applying Gaussian Blur and changing the Canny thresholds did not improve the Random Forest classification performance.

10. More complex preprocessing does not necessarily produce better machine-learning results.

---

# Machine Learning Models

## Image Classification

- Logistic Regression
- Random Forest

## Text Classification

- Logistic Regression
- Multinomial Naive Bayes

---

# Evaluation Metrics

The following metrics were used to evaluate the models:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Time
- Prediction Time

---

# Repository Structure

```text
DS605_Lab6/
│
├── DS605_Lab6.ipynb
│
├── README.md
│
├── emails.csv
│
├── image_features.csv
│
├── image_model_results.csv
│
├── text_model_results.csv
│
└── 448/
    │
    ├── Cracks/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    └── NonCracks/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```


# Conclusion

This assignment demonstrates how raw image and text information can be converted into numerical feature representations and used with traditional machine-learning algorithms.

For asphalt crack classification, nine handcrafted intensity and edge-based features were extracted using NumPy and OpenCV. Random Forest achieved the strongest image-classification performance with **95% accuracy**, while Logistic Regression achieved **93.75% accuracy** with considerably lower training and prediction time.

For email spam classification, CountVectorizer produced **2,974 text features** from the training data. Logistic Regression achieved **98.16% accuracy** and an F1-score of **96.88%**, while Multinomial Naive Bayes achieved **94.20% accuracy** and trained substantially faster.

In Part C, Gaussian Blur and modified Canny thresholds were evaluated as an alternative image representation. The modified representation maintained the same **95% accuracy, precision, recall, and F1-score** as the original representation but did not improve predictive performance.

Overall, the experiments demonstrate the importance of feature extraction, representation choice, model selection, evaluation metrics, and computational trade-offs when applying traditional machine-learning techniques to image and text classification problems.