# 🧠 Logistic Regression Binary Classifier
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

📊 **Model Evaluation Metrics**

🧮 **Confusion Matrix (Default Threshold = 0.5)**  
[[41  2]  
 [ 1 70]]

📋 **Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.98    |  0.95  |   0.96   |   43    |
|   1   |   0.97    |  0.99  |   0.98   |   71    |

**Accuracy:** 0.97  
**Macro Avg:** Precision = 0.97, Recall = 0.97, F1-score = 0.97  
**Weighted Avg:** Precision = 0.97, Recall = 0.97, F1-score = 0.97

📈 **ROC-AUC Score:** 0.9974

🧪 **Confusion Matrix (Threshold = 0.7)**  
[[42  1]  
 [ 4 67]]
