Titanic Survival Prediction ML Pipeline

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques.

The goal is to build a robust ML pipeline including:

- Feature engineering
- Data preprocessing
- Model training
- Hyperparameter tuning
- Model evaluation

---

## Dataset
Dataset used from Kaggle Titanic competition.

Features include:
- Passenger class
- Age
- Sex
- Fare
- Family size
- Passenger titles extracted from names

---

## Feature Engineering

Created new features:

FamilySize = SibSp + Parch + 1  
IsAlone = Indicator for solo passengers  
Title extraction from passenger names  

---

## Machine Learning Workflow

1. Exploratory Data Analysis
2. Feature Engineering
3. Data Preprocessing using ColumnTransformer
4. Pipeline creation to prevent data leakage
5. Model training (Logistic Regression & Random Forest)
6. Hyperparameter tuning using RandomizedSearchCV
7. Threshold optimization
8. Model evaluation

---

## Evaluation Metrics

- ROC AUC
- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

---

## Tech Stack

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  

---

## Model Output

Best model is saved using joblib.
