# **README — Mental Health Risk Prediction in the Workplace**
*By Bellagha Anaïs, Aired Maximilien, Bamba Bilal, Agrad Badr*
---
## **1. Project Overview**

The goal of this project is to answer the question:

> **Can we predict whether an employee is at risk of developing mental health issues based on their working conditions and personal habits?**

Using the public OSMI Mental Health in Tech Survey dataset found on Kaggle, we explore how workplace culture, benefits, managerial support, and personal history influence mental health vulnerability.
This project includes data exploration, preprocessing, baseline machine learning models, advanced ensemble models, and deep learning (MLP).

The repository contains all the notebooks, processed data, and final models needed to reproduce the full pipeline.

## **2. Repository Structure**

```
project/
│
├── EDA_Preprocessing.ipynb       # Data exploration + cleaning + encoding + preprocessing pipeline
├── Baseline_Models.ipynb         # Logistic Regression, Decision Tree, SVM, KNN (custom implementation)
├── Advanced_Models.ipynb         # RandomForest, XGBoost, Voting, Stacking
├── MLP.ipynb                     # Deep learning approach (Simple + Tuned MLP)
│
├── X_train.pkl                   # Preprocessed train features
├── X_test.pkl                    # Preprocessed test features
├── y_train.pkl                   # Train labels
├── y_test.pkl                    # Test labels
├── preprocessor.pkl              # Saved preprocessing pipeline (OneHotEncoder + StandardScaler)
│
└── README.md                     # Project documentation
```

## **3. Dataset**

* **Source:** OSMI Mental Health in Tech Survey
* **Task:** Binary classification ("treatment": 0 or 1)
* **Size:** 1,250 samples after cleaning
* **Type:** Mostly categorical - requires OneHotEncoding
* **Target variable:**

  * `1` → Employee received mental health treatment
  * `0` → Did not receive treatment

The dataset is relatively balanced, which allows fair model evaluation.

---

## **4. Methodology**

### **4.1 Data Preprocessing**

Performed in `EDA_Preprocessing.ipynb`:

* Handling missing values
* Encoding categorical variables with `OneHotEncoder`
* Scaling numerical variables
* Train/test split (80/20)
* Saving preprocessed objects (`preprocessor.pkl`, `.pkl` datasets)

### **4.2 Baseline Models**

Implemented in `Baseline_Models.ipynb`:

* Logistic Regression
* Decision Tree
* SVM (RBF kernel)
* Custom KNN (manual implementation)

### **4.3 Advanced Models**

Implemented in `Advanced_Models.ipynb`:

* Random Forest
* XGBoost
* Soft Voting (RF + XGB)
* Stacking (Logistic meta-model)

### **4.4 Deep Learning (MLP)**

Implemented in `MLP.ipynb`:

* Simple MLP (2 layers)
* Tuned MLP (larger architecture + regularization)

## **5. Model Performance Summary**

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.768    | 0.798     | 0.722  | 0.758    | 0.836   |
| Decision Tree       | 0.732    | 0.771     | 0.667  | 0.715    | 0.733   |
| SVM (RBF)           | 0.752    | 0.776     | 0.714  | 0.744    | 0.842   |
| Custom KNN          | 0.684    | 0.764     | 0.540  | 0.633    | 0.730   |
| **Random Forest**   | 0.768    | 0.783     | 0.746  | 0.764    | 0.843   |
| **XGBoost**         | 0.776    | 0.792     | 0.754  | 0.772    | 0.865   |
| **Soft Voting**     | 0.780    | 0.793     | 0.762  | 0.777    | 0.860   |
| Simple MLP          | 0.724    | 0.736     | 0.706  | 0.721    | 0.816   |
| **Tuned MLP**       | 0.764    | 0.802     | 0.706  | 0.751    | 0.842   |

**Best performing model overall:**
> **XGBoost**

**Best balance (Precision + Recall):**
> **Soft Voting Ensemble**


## **6. Key Insights**

* Workplace factors like **benefits**, **mental health interference**, and **company support** strongly influence risk.
* Ensemble models (RandomForest, XGBoost) perform best on this tabular dataset.
* The tuned MLP reaches competitive performance but does not surpass boosted trees.
* The models reliably detect a significant portion of at-risk employees (recall ~0.75).


## **7. How to Run the Project**

### **Step 1 — Clone the repository**

```bash
git clone https://github.com/anaisblh/Projet_Machine_Learning.git
cd Projet_Machine_Learning
```

### **Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3 — Open the notebooks**

Run them in order:

1. `EDA_Preprocessing.ipynb`
2. `Baseline_Models.ipynb`
3. `Advanced_Models.ipynb`
4. `MLP.ipynb`

All `.pkl` files will load automatically.

## **8. License**

This project is for educational purposes only.
