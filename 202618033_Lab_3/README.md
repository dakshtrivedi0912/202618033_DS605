# Hotel Booking Cancellation Prediction using Machine Learning

## Assignment Title
Hotel Booking Demand - Classification and Preprocessing Comparison

## Student Details
**Name:** Daksh Trivedi  
**ID:** 202618033

## Dataset
**Dataset:** Kaggle Hotel Booking Demand  
**File:** `hotel_bookings.csv`  
**Dataset Link:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## Objective
The objective of this assignment is to predict whether a hotel booking will be canceled using Logistic Regression and Decision Tree classifiers and compare the effect of StandardScaler and MinMaxScaler preprocessing.

## Preprocessing Choices
- Removed `reservation_status` and `reservation_status_date` to prevent data leakage.
- Removed `company` because it had very high missingness.
- Used `KNNImputer(n_neighbors=5)` for numerical features.
- Used `SimpleImputer(strategy="most_frequent")` for categorical features.
- Used `OneHotEncoder(handle_unknown="ignore")` for categorical features.
- Created two preprocessing pipelines:
  - **Pipeline A:** KNN Imputer + StandardScaler
  - **Pipeline B:** KNN Imputer + MinMaxScaler
- Used `train_test_split` with:
  - Test size = 20%
  - Stratification = `is_canceled`
  - Random state = 42

## Models Used
1. Logistic Regression with Pipeline A
2. Logistic Regression with Pipeline B
3. Decision Tree with Pipeline A
4. Decision Tree with Pipeline B

## Final Observations
1. The Decision Tree models achieved the best overall performance with approximately **86.02% test accuracy** and an **F1-score of 0.81**.
2. StandardScaler performed slightly better than MinMaxScaler for Logistic Regression, with test accuracy improving from **81.35% to 81.62%**.
3. Scaling had almost no effect on the Decision Tree, with both pipelines achieving approximately **86.02% test accuracy**.
4. The Decision Tree showed noticeable overfitting, with approximately **99.63% training accuracy** compared with **86.02% testing accuracy**.
5. Logistic Regression showed better generalization, while the Decision Tree achieved higher recall and detected more canceled bookings.

## Deliverables
- `cleaned_hotel_bookings.csv`
- `model_comparison_table.csv`
- `logistic_confusion_matrix.png`
- `decision_tree_confusion_matrix.png`