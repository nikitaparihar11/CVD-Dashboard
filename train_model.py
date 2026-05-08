# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np


# ===============================
# 2. LOAD DATA
# ===============================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]


df = pd.read_csv(url, names=cols)


# Handle missing values
df.replace("?", np.nan, inplace=True)
df = df.dropna()
df = df.astype(float)


# ===============================
# 3. ADD ONLY ONE ENGINEERED FEATURE
# ===============================
df['disease_score'] = df['cp'] + df['exang'] + df['oldpeak'] + df['ca']


# Convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


# ===============================
# 4. ENCODING
# ===============================
df = pd.get_dummies(df, columns=['cp','thal','slope'], drop_first=True)


# Features & Target
X = df.drop('target', axis=1)
y = df['target']


# ===============================
# 5. TRAIN TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ===============================
# 6. MODELS (AS REQUIRED)
# ===============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LightGBM": LGBMClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}


# ===============================
# 7. EVALUATION
# ===============================
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score


results = []


for name, model in models.items():
   
    # Cross-validation
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
   
    # Train
    model.fit(X_train, y_train)
   
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
   
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    kappa = cohen_kappa_score(y_test, y_pred)
   
    results.append([name, acc, prec, auc, kappa, cv_score])


# ===============================
# 8. FINAL RESULTS
# ===============================
results_df = pd.DataFrame(results, columns=[
    "Model","Accuracy","Precision","AUC","Kappa","CV Accuracy"
])


print("\n🔥 FINAL RESULTS (Original + disease_score):\n")
print(results_df.sort_values(by="Accuracy", ascending=False))




# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np


# ===============================
# 2. LOAD DATA
# ===============================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]


df = pd.read_csv(url, names=cols)


# Handle missing values
df.replace("?", np.nan, inplace=True)
df = df.dropna()
df = df.astype(float)


# ===============================
# 3. TARGET PROCESSING
# ===============================
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


# ===============================
# 4. ENCODING
# ===============================
df = pd.get_dummies(df, columns=['cp','thal','slope'], drop_first=True)


# Features & Target
X = df.drop('target', axis=1)
y = df['target']


# ===============================
# 5. TRAIN TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ===============================
# 6. MODELS (SAME AS BEFORE)
# ===============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LightGBM": LGBMClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}


# ===============================
# 7. EVALUATION
# ===============================
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score


results = []


for name, model in models.items():
   
    # Cross-validation
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
   
    # Train
    model.fit(X_train, y_train)
   
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
   
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    kappa = cohen_kappa_score(y_test, y_pred)
   
    results.append([name, acc, prec, auc, kappa, cv_score])


# ===============================
# 8. FINAL RESULTS
# ===============================
results_df = pd.DataFrame(results, columns=[
    "Model","Accuracy","Precision","AUC","Kappa","CV Accuracy"
])


print("\n🔥 FINAL RESULTS (NO FEATURE ENGINEERING):\n")
print(results_df.sort_values(by="Accuracy", ascending=False))

import joblib

best_model = None
best_acc = 0

for name, model in models.items():
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

# Save best model
joblib.dump(best_model, "heart_model.pkl")
joblib.dump(X.columns, "features.pkl")

print(" Best model saved successfully!")