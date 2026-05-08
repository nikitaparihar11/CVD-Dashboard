"""
fix_missing_files.py
=====================
Run this ONCE to generate the two missing files that app.py needs:
    - selector.pkl              (SelectKBest wrapper around your existing features)
    - processed.cleveland.data  (raw Cleveland data for DiCE)

It reuses your existing best_heart_model.pkl and feature_cols.pkl —
NO retraining needed.

Run:
    python fix_missing_files.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  CVD Hub — Fix Missing Files")
print("=" * 55)

# ── Step 1: Download Cleveland data (needed by DiCE) ──────────────────────────
DATA_FILE = Path("processed.cleveland.data")
DATA_URL  = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
             "heart-disease/processed.cleveland.data")
COLS      = ["age","sex","cp","trestbps","chol","fbs","restecg",
             "thalach","exang","oldpeak","slope","ca","thal","target"]

if DATA_FILE.exists():
    print(f"\n✅ processed.cleveland.data already exists — skipping download.")
    df_raw = pd.read_csv(DATA_FILE, names=COLS)
else:
    print(f"\n📥 Downloading Cleveland dataset from UCI…")
    try:
        df_raw = pd.read_csv(DATA_URL, names=COLS)
        df_raw.to_csv(DATA_FILE, index=False, header=False)
        print(f"   ✅ Saved to processed.cleveland.data")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print("   Please manually download from:")
        print("   https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
        print("   and place it in this folder as: processed.cleveland.data")
        df_raw = None

# ── Step 2: Build selector.pkl using your existing feature_cols.pkl ───────────
print("\n🔧 Building selector.pkl…")

try:
    feature_cols = joblib.load("feature_cols.pkl")
    print(f"   Loaded feature_cols.pkl — {len(feature_cols)} features")
except FileNotFoundError:
    print("   ❌ feature_cols.pkl not found! Make sure it's in this folder.")
    raise

# Reconstruct training data to fit the selector
if df_raw is not None:
    df = df_raw.copy()
else:
    # Fallback: try to download again
    df = pd.read_csv(DATA_URL, names=COLS)

df.replace("?", np.nan, inplace=True)
df = df.dropna().astype(float)
df["target"]       = (df["target"] > 0).astype(int)
df["disease_score"] = df["cp"] + df["exang"] + df["oldpeak"] + df["ca"]

df_enc = pd.get_dummies(df, columns=["cp", "thal", "slope"], drop_first=True)

# Ensure all expected columns exist
for col in feature_cols:
    if col not in df_enc.columns:
        df_enc[col] = 0

X = df_enc[feature_cols].astype(float)
y = df_enc["target"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

selector = SelectKBest(f_classif, k="all")   # passes all features through — identity
selector.fit(X_train, y_train)

joblib.dump(selector, "selector.pkl")
print(f"   ✅ selector.pkl saved  ({len(feature_cols)} features, k='all')")

# ── Step 3: Verify all required files exist ───────────────────────────────────
print("\n🔍 Verifying all required files…")
required = {
    "best_heart_model.pkl":    "Main Stack6 prediction model",
    "feature_cols.pkl":        "List of 18 engineered feature names",
    "shap_background.pkl":     "SHAP background sample (100 rows)",
    "selector.pkl":            "Feature selector (just created)",
    "processed.cleveland.data":"Raw Cleveland data for DiCE",
}

all_ok = True
for fname, desc in required.items():
    exists = Path(fname).exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {fname:35s} — {desc}")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("✅ All files present! Run:  streamlit run app.py")
else:
    print("❌ Some files still missing. Check errors above.")
print("=" * 55)