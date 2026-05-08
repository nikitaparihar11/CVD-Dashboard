"""
Train and save the best heart disease model.
Best model: Stack 3 - HetDiv (LGBM+NB+LR+SVM -> RF)
Accuracy: 0.9500 | AUC: 0.9664
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

df = pd.read_csv(url, names=cols)
df.replace("?", np.nan, inplace=True)
df = df.dropna()
df = df.astype(float)

df['target'] = (df['target'] > 0).astype(int)

# ── Train / test split FIRST (prevents data leakage) ───────────────────────────
X_raw = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
X_train = X_train.copy()
X_test  = X_test.copy()

# ── Feature engineering on training data only ──────────────────────────────────
X_train['disease_score'] = X_train['cp'] + X_train['exang'] + X_train['oldpeak'] + X_train['ca']
X_test['disease_score']  = X_test['cp']  + X_test['exang']  + X_test['oldpeak']  + X_test['ca']

# ── Encoding: fit on train only, align test ────────────────────────────────────
X_train = pd.get_dummies(X_train, columns=['cp','thal','slope'], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=['cp','thal','slope'], drop_first=True)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

# Save feature columns so the app can reconstruct them
feature_cols = list(X_train.columns)
joblib.dump(feature_cols, 'feature_cols.pkl')
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Save SHAP background sample (50 representative rows)
shap_background = X_train.sample(50, random_state=42).reset_index(drop=True)
joblib.dump(shap_background, 'shap_background.pkl')
print("SHAP background saved -> shap_background.pkl")

# ── Build best model ───────────────────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_b  = LGBMClassifier(verbose=-1, random_state=42)
nb_b   = Pipeline([("sc", StandardScaler()), ("clf", GaussianNB())])
lr_b   = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
svm_b  = Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True, kernel="rbf", random_state=42))])
meta_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

best_model = StackingClassifier(
    estimators=[("lgbm", lgb_b), ("nb", nb_b), ("lr", lr_b), ("svm", svm_b)],
    final_estimator=meta_rf,
    cv=cv5,
    passthrough=False
)

print("\nTraining best model (Stack 3 - HetDiv: LGBM+NB+LR+SVM -> RF)...")
best_model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nBest Model Performance")
print(f"   Accuracy : {acc:.4f}")
print(f"   AUC      : {auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(best_model, 'best_heart_model.pkl')
print("Model saved -> best_heart_model.pkl")
print("Feature list saved -> feature_cols.pkl")
