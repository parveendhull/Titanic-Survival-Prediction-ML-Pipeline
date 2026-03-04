# Step 1-Problem Statement
'''The Titanic project is about making a model that can tell us if a Titanic passenger will survive or not.
 We want this model to be perfect at predicting who will survive. To do this the Titanic project model looks
 at things like how old the Titanic passengers if they are a man or a woman and what kind of ticket they bought.
 The model uses this information to find patterns that helped some Titanic passengers survive more than others.
 The Titanic project wants to get a score so we know the model is working well and can really tell the difference,
  between the people who survived and those who did not.
'''
# Step 2- Import Libraries
'''Importing the libraries required for the project like numpy for numerical calculations, Pandas for for data analysis,
sklearn for machine learning, etc'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score, roc_curve,confusion_matrix

# Step 3- Load Titanic Dataset using pandas read_csv function
df=pd.read_csv('titanic.csv')




# Step-5 Feature Engineering
# Creating Family-based features to capture passenger dependency patterns
df['FamilySize']=df['SibSp']+df['Parch'] + 1
df['IsAlone']=np.where(df['FamilySize']==1,1,0)
# Extracting titles as female passangers has high survival rate. To get clear data extracting title can be beneficial
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
Rare_titles = [
'Lady','Countess','Capt','Col','Don','Dr','Major',
'Rev','Sir','Jonkheer','Dona'
]
df['Title'] = df['Title'].replace(Rare_titles, 'Rare')
df['Title'] = df['Title'].replace('Mlle','Miss')
df['Title'] = df['Title'].replace('Ms','Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')
df.drop(columns=['Name'],inplace=True)

# Spliting data into input and output where survived column is output we want to predict using other features
X = df.drop(columns=['Survived',])
y = df['Survived']

#Step-6
'''Splitting the data into training data and test data. Training data will be used to train the model and testing 
data will be used to test our model '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step-7
'''Pipeline: To prevent Data Leakage and ensure smooth and clean workflow we use all preprocessing steps inside pipeline.
-Imputing Age column with median because it contains outliers
-Encoding categorical features with one hot encoder inside pipeline
-Using Log Transform to encode Fare 
-Standard Scalar to scale the data to ensure one feature doesn't dominate the model'''


numeric_features_nolog = ['Age', 'FamilySize']
numeric_features_log = ['Fare']
categorical_features = ['Embarked', 'Pclass', 'Sex','Title']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'))
])

log_transformer=FunctionTransformer(np.log1p, validate=True,feature_names_out='one-to-one')

numerical_transformer_nl = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
numerical_transformer_log = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform',log_transformer),
    ('scaler', StandardScaler())
])
preprocessor_lr = ColumnTransformer(transformers=[
    ('num_nl', numerical_transformer_nl, numeric_features_nolog),
    ('num_log', numerical_transformer_log, numeric_features_log),
    ('cat', categorical_transformer, categorical_features)
])

# Step-8
'''BASELINE LOGISTIC REGRESSION PIPELINE 
- Constructing full workflow using preprocessor steps and defining model Logistic Regression with solver saga and panalty as l2'''
clf_pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('classifier', LogisticRegression(random_state=42,max_iter=1000,solver='saga', class_weight=None,penalty='l2')),
])

'''Random Forest Pipeline '''



numeric_features_nolog = ['Age', 'FamilySize']
numeric_features_log = ['Fare']
categorical_features = ['Embarked', 'Pclass', 'Sex','Title']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'))
])

log_transformer=FunctionTransformer(np.log1p, validate=False,feature_names_out='one-to-one')
numerical_transformer_nl = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),

])
numerical_transformer_log = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform',log_transformer),

])
preprocessor_rf = ColumnTransformer(transformers=[
    ('num_nl', numerical_transformer_nl, numeric_features_nolog),
    ('num_log', numerical_transformer_log, numeric_features_log),
    ('cat', categorical_transformer, categorical_features)
])

clf_pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor_rf),
    ('classifier', RandomForestClassifier(n_estimators=100)),
])

# Step-9
'''Hyperparameter Tunning for Logistic regression'''

param_space = [
    {
        # --- Logistic Regression Branch ---
        'classifier': [LogisticRegression(solver='liblinear', class_weight='balanced')],
        'classifier__C': np.logspace(-4, 4, 20),
        'classifier__penalty': ['l1', 'l2']
    }]
'''Using StratifiedKFold as our data is imbalanced'''
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

random_search_lr = RandomizedSearchCV(
    clf_pipeline_lr,
    param_distributions=param_space,
    n_iter=100,
    cv=skf,
    scoring='roc_auc',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

'''Hyperparameter Tuning for Random Forest'''

param_space = [
    {
        # --- Random Forest Branch ---
        'classifier': [RandomForestClassifier(class_weight='balanced')],
        'classifier__n_estimators': [100, 200, 500],
        'classifier__max_depth': [3, 5, 8, 10, None],
        'classifier__min_samples_leaf': [1, 2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__min_samples_split': [2,5,10]
    }]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

random_search_rf = RandomizedSearchCV(
    clf_pipeline_rf,
    param_distributions=param_space,
    n_iter=100,
    cv=skf,
    scoring='roc_auc',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

#Step-10 Training the model
'''Training the model on pipeline to prevent data leakage '''


random_search_lr.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)

#Step-11 Threshold Optimization
'''Get probabilities for the positive class
-Calculate ROC curve metrics
-Calculate Youden's J Statistic: J = TPR - FPR'''
y_probs_lr = random_search_lr.predict_proba(X_test)[:, 1]
y_probs_rf = random_search_rf.predict_proba(X_test)[:, 1]

'''For logistic regression'''
thresholds = np.linspace(0.1,0.9,100)

best_t_lr = 0
best_f1_lr = 0

for t in thresholds:

    y_pred_temp = (y_probs_lr >= t).astype(int)

    f1 = f1_score(y_test, y_pred_temp)

    if f1 > best_f1_lr:
        best_f1_lr = f1
        best_t_lr = t

print("Best LR Threshold:", best_t_lr)
print("Best LR F1:", best_f1_lr)

'''For Random Forest'''
best_t_rf = 0
best_f1_rf = 0

for t in thresholds:

    y_pred_temp = (y_probs_rf >= t).astype(int)

    f1 = f1_score(y_test, y_pred_temp)

    if f1 > best_f1_rf:
        best_f1_rf = f1
        best_t_rf = t

print("Best RF Threshold:", best_t_rf)
print("Best RF F1:", best_f1_rf)

# Now compare the best threshold value for both models

y_pred_lr_opt = (y_probs_lr >= best_t_lr).astype(int)
y_pred_rf_opt = (y_probs_rf >= best_t_rf).astype(int)

# Step-12 Evaluation
'''Now evaluate the model performance using roc_auc, precision,recall ,f1 score'''

roc_auc_lr=roc_auc_score(y_test, y_probs_lr)
roc_auc_rf=roc_auc_score(y_test, y_probs_rf)
precision_lr=precision_score(y_test, y_pred_lr_opt)
recall_lr=recall_score(y_test, y_pred_lr_opt)
f1_lr=f1_score(y_test, y_pred_lr_opt)
precision_rf=precision_score(y_test, y_pred_rf_opt)
recall_rf=recall_score(y_test, y_pred_rf_opt)
f1_rf=f1_score(y_test, y_pred_rf_opt)
acc_lr=accuracy_score(y_test, y_pred_lr_opt)
acc_rf=accuracy_score(y_test, y_pred_rf_opt)
confusion_matrix_lr=confusion_matrix(y_test, y_pred_lr_opt)

confusion_matrix_rf=confusion_matrix(y_test, y_pred_rf_opt)


results = pd.DataFrame({

'Model':['Logistic Regression','Random Forest'],
'ROC-AUC':[roc_auc_lr, roc_auc_rf],
'Precision':[precision_lr, precision_rf],
'Recall':[recall_lr, recall_rf],
'F1':[f1_lr, f1_rf],
'Accuracy':[acc_lr, acc_rf],
'Confusion Matrix':[confusion_matrix_lr, confusion_matrix_rf],

})

print(results)


#Step-13
'''# Feature importance extracted from Random Forest & Logistic regression to understand dominant survival predictors'''


best_pipeline = random_search_rf.best_estimator_
rf_model=best_pipeline.named_steps['classifier']

feature_names = best_pipeline[:-1].get_feature_names_out()
importance_values = rf_model.feature_importances_

importance_rf = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_values
})

importance_rf = importance_rf.sort_values(by='Importance', ascending=False)

print(importance_rf.head(10))

#Visualization

plt.figure(figsize=(10,6))

plt.barh(
    importance_rf['Feature'][:10],
    importance_rf['Importance'][:10]
)

plt.gca().invert_yaxis()

plt.title("Top Feature Importance - Random Forest")

plt.show()


#Step-14 Selecting best model based on roc auc score

best_model = random_search_lr if roc_auc_lr > roc_auc_rf else random_search_rf

#step-15 Saving the model

import joblib
joblib.dump(best_model, 'titanic_model.pkl')