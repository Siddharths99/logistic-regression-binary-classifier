# Logistic Regression Binary Classifier
A complete binary classification project using logistic regression on the Breast Cancer dataset. Includes data preprocessing, model training, evaluation with confusion matrix, ROC-AUC, threshold tuning, and visualization ideal for understanding sigmoid-based classifiers.

---

## 🛠️ Tools Used
- Scikit-learn  
- Pandas  
- Matplotlib  
- Seaborn  
- NumPy  


---
Installation

pip install pandas numpy matplotlib seaborn scikit-learn

----
Steps Performed

1.Data Cleaning: Removed irrelevant columns (id, unnamed).

2.Label Encoding: Converted target variable ('M' → 1, 'B' → 0).

3.Train-Test Split: Split into 80% training and 20% testing.

4.Feature Scaling: Applied StandardScaler for normalization.

5.Model Training: Built a logistic regression model.

6.Evaluation Metrics: Generated Confusion Matrix and Classification Report

7.Calculated ROC-AUC Score : Threshold Tuning: Tested predictions with a threshold of 0.3.

8.Visualizations: a.ROC Curve

b.Confusion Matrices for default and tuned thresholds

----
📊 **Model Evaluation Results**

CONFUSION MATRIX (Threshold = 0.5)

 [[70  1]
 [ 2 41]]

-------

 **Classification Report**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.97    |  0.99  |   0.98   |   71    |
|   1   |   0.98    |  0.95  |   0.96   |   43    |

**Accuracy**: 0.97  
**Macro Avg**: Precision = 0.97, Recall = 0.97, F1-score = 0.97  
**Weighted Avg**: Precision = 0.97, Recall = 0.97, F1-score = 0.97  

---

**ROC-AUC Score**: `0.9974`

---

Confusion Matrix (Threshold = 0.3):
 [[67  4]
 [ 1 42]]

