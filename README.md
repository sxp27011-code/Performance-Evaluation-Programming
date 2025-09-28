**700762701 SREE RAM CHARAN TEJA PUDARI**
# Performance-Evaluation-Programming
# 📊 Performance Evaluation of kNN Classifier on Iris Dataset

This project demonstrates how to evaluate the performance of a **k-Nearest Neighbors (kNN) Classifier** on the Iris dataset.  
We compute performance metrics such as **confusion matrix, accuracy, precision, recall, F1-score**, and also plot **ROC curves with AUC** for multiclass classification.

---

## 📌 Problem Statement

1. Train a **kNN classifier** with `k = 5` on the Iris dataset.  
2. Compute and display the **confusion matrix** using `sklearn.metrics.confusion_matrix`.  
3. Compute **accuracy, precision, recall, and F1-score** using `classification_report`.  
4. Plot the **ROC curve** and compute the **AUC** for each class.  

---

## 💻 Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binarize labels for ROC (since Iris has 3 classes)
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Split dataset
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.3, random_state=42, stratify=y
)

# 2. Train kNN classifier (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 3. Classification Report (Accuracy, Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 4. ROC Curve & AUC (One-vs-Rest for multiclass)
y_score = knn.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curves
plt.figure(figsize=(8, 6))
colors = ["blue", "green", "red"]
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve of class {iris.target_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest) for kNN (k=5)")
plt.legend(loc="lower right")
plt.show()
