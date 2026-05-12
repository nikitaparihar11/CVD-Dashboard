# ===============================
# CVD INTEGRATED DASHBOARD
# Prediction + SHAP + DiCE + Chatbot + Recommendation
# ===============================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import dice_ml
from dice_ml import Dice
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMClassifier
import plotly.graph_objects as go

from nlp_engine import extract_symptoms
from llm_engine import generate_response
from voice_engine import transcribe_audio
from lang_engine import detect_language

from auth import require_auth

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="CVD Intelligence Hub",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth()

# ───────────────────────────────────────────────
# GLOBAL CSS — SOFT BLUE-GREY LIGHT MEDICAL THEME
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── ROOT TOKENS ── */
:root {
    --bg-base:         #eef1f6;
    --bg-surface:      #f5f7fb;
    --bg-card:         #ffffff;
    --bg-sidebar:      #f0f3f9;
    --border:          #d4dce9;
    --border-strong:   #b8c5d6;
    --text-primary:    #1a2535;
    --text-secondary:  #4a607a;
    --text-muted:      #8a9db5;
    --accent-blue:     #2563eb;
    --accent-teal:     #0891b2;
    --accent-green:    #059669;
    --accent-amber:    #d97706;
    --accent-red:      #dc2626;
    --accent-purple:   #7c3aed;
    --shadow-sm:       0 1px 3px rgba(30,60,100,0.08), 0 1px 2px rgba(30,60,100,0.05);
    --shadow-md:       0 4px 12px rgba(30,60,100,0.10), 0 2px 4px rgba(30,60,100,0.06);
    --radius:          12px;
    --font-ui:         'Plus Jakarta Sans', sans-serif;
    --font-mono:       'JetBrains Mono', monospace;
}

/* ── GLOBAL RESET ── */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1400px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-ui) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: var(--accent-blue) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 20px !important;
    margin-bottom: 8px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 10px 0 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.77rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}

/* ── TABS ── */
div[data-testid="stTabs"] > div:first-child {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
    box-shadow: var(--shadow-sm);
}
button[data-baseweb="tab"] {
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border-radius: 7px !important;
    padding: 8px 22px !important;
    transition: all 0.18s ease !important;
    border: none !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--accent-blue) !important;
    background: #eff6ff !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
div[data-testid="stTabs"] > div:nth-child(2) {
    border-top: none !important;
    padding-top: 18px !important;
}

/* ── NATIVE STREAMLIT WIDGETS ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border-strong) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.18) !important;
}
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: all 0.15s !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: #eff6ff !important;
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    border-color: var(--border-strong) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important;
}
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-sm) !important;
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-size: 0.84rem !important;
    background: var(--bg-surface) !important;
    padding: 12px 16px !important;
}

/* ── MARKDOWN TABLES ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
    background: var(--bg-card);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}
th {
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    padding: 9px 12px !important;
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
}
td {
    padding: 8px 12px !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--bg-surface);
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-surface) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

/* ════════════════════════════════════
   COMPONENT CLASSES
   ════════════════════════════════════ */

/* Banner */
.hub-banner {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 45%, #0369a1 100%);
    border-radius: 14px;
    padding: 26px 32px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(37,99,235,0.22);
}
.hub-banner::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hub-banner::after {
    content: '';
    position: absolute; bottom: -40px; left: 30%;
    width: 300px; height: 100px;
    background: radial-gradient(ellipse, rgba(255,255,255,0.06) 0%, transparent 70%);
}
.hub-banner h1 {
    font-family: var(--font-ui);
    font-size: 1.75rem; font-weight: 800;
    color: #ffffff; margin: 0 0 4px;
    letter-spacing: -0.3px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.15);
}
.hub-banner p {
    color: rgba(255,255,255,0.72);
    margin: 0; font-size: 0.87rem; line-height: 1.6;
}
.hub-banner .banner-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.15); backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.72rem; font-weight: 600; color: #fff;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 10px;
}

/* KPI Cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-danger::before  { background: var(--accent-red); }
.kpi-warning::before { background: var(--accent-amber); }
.kpi-success::before { background: var(--accent-green); }
.kpi-info::before    { background: var(--accent-blue); }
.kpi-card .kpi-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;
}
.kpi-card .kpi-value {
    font-family: var(--font-ui);
    font-size: 2rem; font-weight: 800; line-height: 1;
}
.kpi-card .kpi-sub { font-size: 0.74rem; color: var(--text-muted); margin-top: 5px; }
.kpi-danger  .kpi-value { color: var(--accent-red); }
.kpi-warning .kpi-value { color: var(--accent-amber); }
.kpi-success .kpi-value { color: var(--accent-green); }
.kpi-info    .kpi-value { color: var(--accent-blue); }

/* Section Cards */
.sec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
}
.sec-card h3 {
    font-family: var(--font-ui);
    font-size: 0.88rem; font-weight: 700;
    color: var(--text-primary); margin: 0 0 14px;
    letter-spacing: 0.01em;
    display: flex; align-items: center; gap: 7px;
}
.sec-card h3::before {
    content: '';
    display: inline-block; width: 3px; height: 14px;
    background: var(--accent-blue);
    border-radius: 2px;
    flex-shrink: 0;
}

/* Alert Boxes */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 8px 8px 0;
    padding: 11px 15px; font-size: 0.83rem;
    color: #1e40af; line-height: 1.6; margin: 8px 0;
}
.warn-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 3px solid var(--accent-red);
    border-radius: 0 8px 8px 0;
    padding: 11px 15px; font-size: 0.83rem;
    color: #991b1b; line-height: 1.6; margin: 8px 0;
}
.success-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid var(--accent-green);
    border-radius: 0 8px 8px 0;
    padding: 11px 15px; font-size: 0.83rem;
    color: #14532d; line-height: 1.6; margin: 8px 0;
}

/* DiCE Section */
.dice-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px; margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
}
.dice-section h3 {
    font-family: var(--font-ui);
    font-size: 0.88rem; font-weight: 700;
    color: var(--text-primary); margin: 0 0 14px;
}
.dice-banner {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-left: 3px solid var(--accent-purple);
    border-radius: 0 8px 8px 0;
    padding: 11px 15px; font-size: 0.83rem;
    color: #4c1d95; line-height: 1.6; margin: 8px 0;
}
.cf-original { color: var(--accent-red); font-weight: 600; }
.cf-target   { color: var(--accent-green); font-weight: 600; }
.cf-arrow    { color: var(--text-muted); font-weight: 600; }
.shap-positive { color: var(--accent-red); font-weight: 600; }
.shap-negative { color: var(--accent-blue); font-weight: 600; }

/* NL Explanation Boxes */
.nl-explain {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid var(--accent-green);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px; font-size: 0.85rem;
    line-height: 1.75; color: var(--text-primary); margin: 8px 0;
}
.nl-explain-risk {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 3px solid var(--accent-red);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px; font-size: 0.85rem;
    line-height: 1.75; color: var(--text-primary); margin: 8px 0;
}

/* Risk Badges */
.risk-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: var(--font-ui); font-weight: 700;
    font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px;
}
.risk-badge.high   { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.risk-badge.medium { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.risk-badge.low    { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

/* Recommendation Cards */
.rec-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
    transition: box-shadow 0.15s, border-color 0.15s;
}
.rec-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--accent-blue);
}
.rec-card .rec-header {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--font-ui); font-weight: 700;
    font-size: 0.86rem; color: var(--text-primary); margin-bottom: 8px;
}
.rec-card .rec-icon {
    width: 28px; height: 28px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}
.rec-card ul { margin: 0; padding-left: 18px; }
.rec-card ul li { font-size: 0.81rem; color: var(--text-secondary); line-height: 1.75; }

/* Chat Bubbles */
.chat-user {
    background: var(--accent-blue);
    border-radius: 14px 14px 4px 14px;
    padding: 10px 14px; font-size: 0.87rem;
    color: #ffffff; max-width: 78%;
    margin-left: auto; margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2);
}
.chat-bot {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px;
    padding: 10px 14px; font-size: 0.87rem;
    color: var(--text-primary); max-width: 78%;
    margin-bottom: 8px;
    box-shadow: var(--shadow-sm);
}

/* Heartbeat animation */
@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    14%       { transform: scale(1.3); }
    28%       { transform: scale(1); }
    42%       { transform: scale(1.3); }
    70%       { transform: scale(1); }
}
.beat { display: inline-block; animation: heartbeat 1.5s ease-in-out infinite; }
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────
LABEL_MAP = {
    'age': 'Age', 'sex': 'Sex', 'trestbps': 'Blood Pressure',
    'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG', 'thalach': 'Max Heart Rate',
    'exang': 'Exercise Angina', 'oldpeak': 'ST Depression',
    'ca': 'Blocked Vessels', 'disease_score': 'Disease Score',
    'cp_2.0': 'CP: Atypical Angina', 'cp_3.0': 'CP: Non-Anginal',
    'cp_4.0': 'CP: Asymptomatic', 'thal_6.0': 'Thal: Fixed Defect',
    'thal_7.0': 'Thal: Reversible', 'slope_2.0': 'Slope: Flat',
    'slope_3.0': 'Slope: Downsloping',
}

DICE_LABELS = {
    'age': 'Age', 'trestbps': 'Blood Pressure',
    'chol': 'Cholesterol', 'thalach': 'Max Heart Rate',
    'oldpeak': 'ST Depression', 'disease_score': 'Disease Score',
    'cp': 'Chest Pain', 'exang': 'Exercise Angina',
    'ca': 'Blocked Vessels', 'slope': 'ST Slope',
    'thal': 'Thalassemia', 'sex': 'Sex',
    'fbs': 'Fasting Sugar', 'restecg': 'Resting ECG',
}

DICE_UNITS = {
    'age': ' yrs', 'trestbps': ' mmHg', 'chol': ' mg/dl',
    'thalach': ' bpm', 'oldpeak': '', 'disease_score': '',
}

ACTION_MAP = {
    "trestbps": {
        "icon": "💊", "bg": "rgba(220,38,38,0.08)", "title": "Blood Pressure",
        "actions": ["Reduce salt intake to < 1500mg/day", "Daily 30-min cardio exercise", "Stress reduction & meditation", "Monitor BP twice daily"],
    },
    "chol": {
        "icon": "🥗", "bg": "rgba(217,119,6,0.08)", "title": "Cholesterol",
        "actions": ["Adopt a low-fat, high-fibre diet", "Increase omega-3 intake (fish, flaxseed)", "Reduce saturated & trans fats", "Consider plant sterols/stanols"],
    },
    "thalach": {
        "icon": "🏃", "bg": "rgba(37,99,235,0.08)", "title": "Max Heart Rate",
        "actions": ["Structured aerobic training 4x/week", "Zone 2 cardio (60-70% max HR)", "Gradual intensity progression", "Track heart rate during exercise"],
    },
    "oldpeak": {
        "icon": "🧘", "bg": "rgba(124,58,237,0.08)", "title": "ST Depression",
        "actions": ["Yoga & breathing exercises", "Avoid intense exertion until reviewed", "Stress management therapy", "Consult cardiologist for ECG review"],
    },
    "disease_score": {
        "icon": "🏥", "bg": "rgba(5,150,105,0.08)", "title": "Overall Disease Score",
        "actions": ["Complete lifestyle overhaul", "Regular cardiac monitoring", "Medication adherence if prescribed", "Dietitian + exercise physiologist referral"],
    },
}

# ───────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history        = []
if "lang"           not in st.session_state: st.session_state.lang           = "en"
if "last_audio_id"  not in st.session_state: st.session_state.last_audio_id  = None
if "pending_voice"  not in st.session_state: st.session_state.pending_voice  = None

# ───────────────────────────────────────────────
# MODEL LOADING
# ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("best_heart_model.pkl")
    features = joblib.load("feature_cols.pkl")
    try:
        background = joblib.load("shap_background.pkl")
    except FileNotFoundError:
        background = None
    return model, features, background

@st.cache_resource
def load_rec_data():
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df   = pd.read_csv(url, names=cols)
    df.replace("?", np.nan, inplace=True)
    df   = df.dropna().astype(float)
    df['disease_score'] = df['cp'] + df['exang'] + df['oldpeak'] + df['ca']
    df['target']        = (df['target'] > 0).astype(int)

    df_enc = pd.get_dummies(df, columns=['cp','thal','slope'], drop_first=True)
    X = df_enc.drop('target', axis=1)
    y = df_enc['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    svd    = TruncatedSVD(n_components=5, random_state=42)
    latent = svd.fit_transform(X_train_s)

    content_sim = cosine_similarity(X_train_s)
    collab_sim  = cosine_similarity(latent)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_train_s": X_train_s, "scaler": scaler,
        "svd": svd, "latent": latent,
        "content_sim": content_sim, "collab_sim": collab_sim,
        "feature_cols": list(X_train.columns),
    }

@st.cache_resource
def get_shap_explainer(_model, _background, _feature_cols):
    if _background is None:
        return None
    bg = _background.astype(float)
    try:
        lgb_s = LGBMClassifier(verbose=-1, random_state=42)
        bg_preds = (_model.predict_proba(bg)[:, 1] >= 0.5).astype(int)
        if len(np.unique(bg_preds)) < 2:
            raise ValueError("single class")
        lgb_s.fit(bg, bg_preds)
        explainer = shap.TreeExplainer(lgb_s)
        return {"kind": "tree", "explainer": explainer}
    except Exception:
        predict_fn = lambda x: _model.predict_proba(
            pd.DataFrame(x, columns=_feature_cols).astype(float))[:, 1]
        explainer = shap.KernelExplainer(predict_fn, bg.values)
        return {"kind": "kernel", "explainer": explainer}

@st.cache_resource
def get_dice_pipeline():
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]

    df_dice = pd.read_csv("processed.cleveland.data", names=cols)
    df_dice.replace("?", np.nan, inplace=True)
    df_dice = df_dice.dropna().astype(float)
    df_dice['target'] = (df_dice['target'] > 0).astype(int)
    df_dice['disease_score'] = df_dice['cp'] + df_dice['exang'] + df_dice['oldpeak'] + df_dice['ca']

    feature_cols_dice = ["age","sex","cp","trestbps","chol","fbs","restecg",
                         "thalach","exang","oldpeak","slope","ca","thal","disease_score"]
    categorical_cols  = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
    continuous_cols   = ["age","trestbps","chol","thalach","oldpeak","disease_score"]

    X_dice = df_dice[feature_cols_dice].copy()
    y_dice = df_dice['target'].copy()
    for c in categorical_cols: X_dice[c] = X_dice[c].astype(int)
    for c in continuous_cols:  X_dice[c] = X_dice[c].astype(float)

    X_train, _, y_train, _ = train_test_split(X_dice, y_dice, test_size=0.2, random_state=42)

    lgb_d  = LGBMClassifier(verbose=-1, random_state=42)
    nb_d   = Pipeline([("sc", StandardScaler()), ("clf", GaussianNB())])
    lr_d   = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    svm_d  = Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True, kernel="rbf", random_state=42))])
    meta_d = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

    stack6_dice = StackingClassifier(
        estimators=[("lgbm", lgb_d), ("nb", nb_d), ("lr", lr_d), ("svm", svm_d)],
        final_estimator=meta_d, cv=5, passthrough=False
    )
    stack6_dice.fit(X_train, y_train)

    train_df = X_train.copy()
    train_df['target'] = y_train.values

    dice_data  = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_cols,
        outcome_name='target'
    )
    dice_model = dice_ml.Model(model=stack6_dice, backend='sklearn')
    dice_exp   = Dice(dice_data, dice_model, method='random')

    return {
        "explainer":        dice_exp,
        "model":            stack6_dice,
        "feature_cols":     feature_cols_dice,
        "categorical_cols": categorical_cols,
        "continuous_cols":  continuous_cols,
    }

try:
    model, feature_cols, shap_bg = load_model()
    rec_bundle   = load_rec_data()
    shap_bundle  = get_shap_explainer(model, shap_bg, feature_cols)
    dice_bundle  = get_dice_pipeline()
    MODEL_OK     = True
    DICE_OK      = True
except FileNotFoundError as e:
    MODEL_OK = False
    DICE_OK  = False

# ───────────────────────────────────────────────
# FEATURE HELPERS
# ───────────────────────────────────────────────
def build_features(inputs: dict, feature_cols: list) -> pd.DataFrame:
    cp, exang, oldpeak, ca = inputs['cp'], inputs['exang'], inputs['oldpeak'], inputs['ca']
    ds = cp + exang + oldpeak + ca
    row = {
        'age': inputs['age'], 'sex': inputs['sex'],
        'trestbps': inputs['trestbps'], 'chol': inputs['chol'],
        'fbs': inputs['fbs'], 'restecg': inputs['restecg'],
        'thalach': inputs['thalach'], 'exang': exang,
        'oldpeak': oldpeak, 'ca': ca, 'disease_score': ds,
        'cp_2.0': 1 if cp == 2.0 else 0,
        'cp_3.0': 1 if cp == 3.0 else 0,
        'cp_4.0': 1 if cp == 4.0 else 0,
        'thal_6.0': 1 if inputs['thal'] == 6.0 else 0,
        'thal_7.0': 1 if inputs['thal'] == 7.0 else 0,
        'slope_2.0': 1 if inputs['slope'] == 2.0 else 0,
        'slope_3.0': 1 if inputs['slope'] == 3.0 else 0,
    }
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
    return df[feature_cols]

def build_dice_query(inputs: dict, feature_cols_dice: list,
                     categorical_cols: list, continuous_cols: list) -> pd.DataFrame:
    disease_score = inputs['cp'] + inputs['exang'] + inputs['oldpeak'] + inputs['ca']
    row = {
        'age': inputs['age'], 'sex': inputs['sex'], 'cp': inputs['cp'],
        'trestbps': inputs['trestbps'], 'chol': inputs['chol'], 'fbs': inputs['fbs'],
        'restecg': inputs['restecg'], 'thalach': inputs['thalach'], 'exang': inputs['exang'],
        'oldpeak': inputs['oldpeak'], 'slope': inputs['slope'], 'ca': inputs['ca'],
        'thal': inputs['thal'], 'disease_score': disease_score,
    }
    df = pd.DataFrame([row])[feature_cols_dice]
    for c in categorical_cols: df[c] = df[c].astype(int)
    for c in continuous_cols:  df[c] = df[c].astype(float)
    return df

def symptoms_to_features(symptoms: dict) -> dict:
    return {
        "age": 50, "sex": 1, "trestbps": 120, "chol": 200,
        "fbs": 0,  "restecg": 1, "thalach": 150,
        "cp": 1, "slope": 1, "thal": 3,
        "exang":   1   if symptoms.get("exercise")   else 0,
        "oldpeak": 1.5 if symptoms.get("chest_pain") else 0,
        "ca":      0,
    }

# ───────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ───────────────────────────────────────────────
def get_recommendations(inputs: dict, rec: dict, prob: float) -> dict:
    user_df = pd.DataFrame([{
        'age': inputs['age'], 'sex': inputs['sex'], 'cp': inputs['cp'],
        'trestbps': inputs['trestbps'], 'chol': inputs['chol'], 'fbs': inputs['fbs'],
        'restecg': inputs['restecg'], 'thalach': inputs['thalach'], 'exang': inputs['exang'],
        'oldpeak': inputs['oldpeak'], 'slope': inputs['slope'], 'ca': inputs['ca'],
        'thal': inputs['thal'], 'disease_score': inputs['cp'] + inputs['exang'] + inputs['oldpeak'] + inputs['ca'],
    }])
    user_df = pd.get_dummies(user_df, columns=['cp','thal','slope'], drop_first=True)
    user_df = user_df.reindex(columns=rec["feature_cols"], fill_value=0)
    user_scaled = rec["scaler"].transform(user_df)

    content_scores = cosine_similarity(user_scaled, rec["X_train_s"])[0]
    user_latent    = rec["svd"].transform(user_scaled)
    collab_scores  = cosine_similarity(user_latent, rec["latent"])[0]
    hybrid_scores  = 0.5 * content_scores + 0.5 * collab_scores
    hybrid_idx     = hybrid_scores.argsort()[-21:-1][::-1]

    healthy_idx = [i for i in hybrid_idx if rec["y_train"].iloc[i] == 0][:10]
    if not healthy_idx:
        healthy_idx = hybrid_idx[:5]

    healthy_pattern = rec["X_train"].iloc[healthy_idx].mean()
    diff = (healthy_pattern - user_df.iloc[0]).sort_values(key=abs, ascending=False)

    ranked_actions = []
    for f in diff.index:
        base_f = f.split("_")[0] if "_" in f else f
        if base_f in ACTION_MAP:
            status = "HIGH" if diff[f] < 0 else "LOW"
            ranked_actions.append({
                "feature": base_f, "status": status,
                "delta": float(diff[f]),
                "user_val": float(user_df.iloc[0].get(f, 0)),
                "target_val": float(healthy_pattern.get(f, 0)),
                **ACTION_MAP[base_f],
            })
        if len(ranked_actions) >= 4:
            break

    if prob < 0.2:
        feedback = ("Healthy", "Your cardiovascular profile looks good. Maintain your current lifestyle.", "success")
    elif prob < 0.4:
        feedback = ("Low Risk", "Minor lifestyle improvements recommended.", "info")
    elif prob < 0.6:
        feedback = ("Moderate Risk", "Targeted lifestyle changes can significantly reduce your risk.", "warning")
    elif prob < 0.8:
        feedback = ("High Risk", "Prompt medical attention and lifestyle intervention advised.", "danger")
    else:
        feedback = ("Critical Risk", "Immediate cardiologist consultation required.", "danger")

    return {
        "ranked_actions": ranked_actions, "feedback": feedback,
        "healthy_pattern": healthy_pattern, "diff": diff,
        "n_healthy_neighbors": len(healthy_idx),
    }

# ───────────────────────────────────────────────
# CHART HELPERS — light-theme Plotly
# ───────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(245,247,251,0.8)',
    font=dict(family='Plus Jakarta Sans', color='#4a607a', size=11),
    margin=dict(t=20, b=40, l=10, r=20),
)

def make_gauge(prob: float) -> go.Figure:
    color = "#dc2626" if prob >= 0.6 else ("#d97706" if prob >= 0.35 else "#059669")
    label = "HIGH RISK" if prob >= 0.6 else ("MODERATE" if prob >= 0.35 else "LOW RISK")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 44, 'color': color, 'family': 'Plus Jakarta Sans'}},
        title={'text': f"<b>{label}</b>", 'font': {'size': 15, 'color': color, 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': '#b8c5d6'}, 'tickwidth': 0},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#f5f7fb', 'borderwidth': 0,
            'steps': [
                {'range': [0, 35],   'color': 'rgba(5,150,105,0.07)'},
                {'range': [35, 60],  'color': 'rgba(217,119,6,0.07)'},
                {'range': [60, 100], 'color': 'rgba(220,38,38,0.07)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.85, 'value': prob * 100}
        }
    ))
    fig.update_layout(height=260, **PLOTLY_BASE)
    return fig

def make_risk_bars(inputs: dict) -> go.Figure:
    items = {
        'Disease Score':   min((inputs['cp']+inputs['exang']+inputs['oldpeak']+inputs['ca']) / 10, 1.0),
        'ST Depression':   min(inputs['oldpeak'] / 6, 1.0),
        'Blocked Vessels': inputs['ca'] / 3,
        'Chest Pain':      (inputs['cp'] - 1) / 3,
        'Cholesterol':     min(inputs['chol'] / 400, 1.0),
        'Blood Pressure':  min(inputs['trestbps'] / 200, 1.0),
        'Age Factor':      min(inputs['age'] / 80, 1.0),
        'HR Deficit':      max(0, 1 - inputs['thalach'] / 200),
    }
    labels = list(items.keys())
    vals   = [v * 100 for v in items.values()]
    colors = ['#dc2626' if v >= 60 else ('#d97706' if v >= 35 else '#059669') for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0), opacity=0.78),
        text=[f"{v:.0f}%" for v in vals], textposition='outside',
        textfont=dict(color='#7a95b0', size=10),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 118], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(color='#1a2535', size=11), gridcolor='rgba(0,0,0,0.04)'),
        height=260, **PLOTLY_BASE,
    )
    return fig

def make_shap_chart(shap_vals: np.ndarray, feature_cols: list, X_row: pd.DataFrame) -> go.Figure:
    top_n = 12
    idx  = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    labels, values = [], []
    for i in idx:
        col = feature_cols[i]
        raw = X_row.iloc[0][col]
        labels.append(f"{LABEL_MAP.get(col, col)} = {raw:.3g}")
        values.append(float(shap_vals[i]))
    labels  = labels[::-1]; values = values[::-1]
    colors  = ['#dc2626' if v > 0 else '#2563eb' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0), opacity=0.72),
        text=[f"{v:+.3f}" for v in values], textposition='outside',
        textfont=dict(color='#7a95b0', size=10),
        hovertemplate='%{y}<br>SHAP: %{x:+.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='rgba(0,0,0,0.10)', line_width=1)
    fig.update_layout(
        xaxis=dict(title=dict(text="SHAP impact on risk probability", font=dict(color='#7a95b0', size=11)),
                   tickfont=dict(color='#b8c5d6'), gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        yaxis=dict(tickfont=dict(color='#1a2535', size=10), gridcolor='rgba(0,0,0,0.04)'),
        height=400, **PLOTLY_BASE,
    )
    return fig

def natural_language_shap(shap_vals: np.ndarray, feature_cols: list,
                           X_row: pd.DataFrame, pred: int, prob: float) -> str:
    idx_sorted = np.argsort(np.abs(shap_vals))[::-1]
    top_pos, top_neg = [], []
    for i in idx_sorted:
        col = feature_cols[i]; sv = float(shap_vals[i]); raw = X_row.iloc[0][col]
        entry = (LABEL_MAP.get(col, col), sv, raw)
        if sv > 0.01 and len(top_pos) < 3:  top_pos.append(entry)
        elif sv < -0.01 and len(top_neg) < 3: top_neg.append(entry)
        if len(top_pos) == 3 and len(top_neg) == 3: break

    verdict = "HEART DISEASE DETECTED" if pred == 1 else "NO HEART DISEASE DETECTED"
    conf    = f"{prob*100:.1f}%" if pred == 1 else f"{(1-prob)*100:.1f}%"
    lines   = [f"<b>Prediction: {verdict}</b> (confidence {conf})<br><br>"]

    if top_pos:
        lines.append("<b>Top factors INCREASING risk:</b><br>")
        for name, sv, raw in top_pos:
            lines.append(f"&nbsp;&nbsp;• <b>{name}</b> = {raw:.3g} "
                         f"<span class='shap-positive'>(+{sv:.4f})</span><br>")
    if top_neg:
        lines.append("<br><b>Top factors DECREASING risk:</b><br>")
        for name, sv, raw in top_neg:
            lines.append(f"&nbsp;&nbsp;• <b>{name}</b> = {raw:.3g} "
                         f"<span class='shap-negative'>({sv:.4f})</span><br>")
    return "".join(lines)

def make_dice_chart(query: pd.DataFrame, cf_df: pd.DataFrame,
                    feature_cols_dice: list) -> go.Figure:
    original = query.iloc[0]
    changed  = [
        f for f in feature_cols_dice
        if f in cf_df.columns and bool(np.any(np.abs(cf_df[f].values - original[f]) > 0.01))
    ]

    if not changed:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature changes needed — patient already near the low-risk boundary.",
            x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(color='#7a95b0', size=13)
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(visible=False), yaxis=dict(visible=False), height=260)
        return fig

    n_cfs      = len(cf_df)
    cf_palette = ['#059669', '#2563eb', '#7c3aed']
    fig        = go.Figure()

    fig.add_trace(go.Bar(
        name='Current Patient (High Risk)',
        x=[DICE_LABELS.get(f, f) for f in changed],
        y=[float(original[f]) for f in changed],
        marker=dict(color='#dc2626', line=dict(width=0), opacity=0.75),
        text=[f"{float(original[f]):.2f}" for f in changed],
        textposition='outside',
        textfont=dict(color='#dc2626', size=10),
        hovertemplate='%{x}<br>Current: %{y:.2f}<extra></extra>',
    ))
    for i in range(n_cfs):
        fig.add_trace(go.Bar(
            name=f'Counterfactual {i+1} (Low Risk)',
            x=[DICE_LABELS.get(f, f) for f in changed],
            y=[float(cf_df[f].iloc[i]) for f in changed],
            marker=dict(color=cf_palette[i % len(cf_palette)], line=dict(width=0), opacity=0.75),
            text=[f"{float(cf_df[f].iloc[i]):.2f}" for f in changed],
            textposition='outside',
            textfont=dict(color=cf_palette[i % len(cf_palette)], size=10),
            hovertemplate=f'%{{x}}<br>CF {i+1}: %{{y:.2f}}<extra></extra>',
        ))

    fig.update_layout(
        barmode='group',
        xaxis=dict(tickfont=dict(color='#1a2535', size=11), gridcolor='rgba(0,0,0,0.04)'),
        yaxis=dict(tickfont=dict(color='#7a95b0', size=10), gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        legend=dict(font=dict(color='#4a607a'), orientation='h', y=-0.22, x=0.5, xanchor='center'),
        height=300, **PLOTLY_BASE,
    )
    return fig

def dice_natural_language(query: pd.DataFrame, cf_df: pd.DataFrame, pred: int, prob: float) -> str:
    if pred == 0:
        return ("<b>Prediction: NO HEART DISEASE DETECTED</b> "
                f"(confidence {(1-prob)*100:.1f}%)<br><br>"
                "<i>DiCE counterfactuals are generated only when the model predicts Disease. "
                "Adjust the sidebar to a higher-risk profile to see actionable counterfactuals.</i>")

    original = query.iloc[0]
    n_cfs    = len(cf_df)
    lines    = [
        f"<b>Prediction: HEART DISEASE DETECTED</b> (confidence {prob*100:.1f}%)<br><br>",
        "<i>DiCE finds the <b>minimal changes</b> needed to flip this prediction to "
        "<b>No Disease</b> — concrete, actionable clinical targets.</i><br><br>",
        f"<b>{n_cfs} counterfactual{'s' if n_cfs > 1 else ''} generated:</b><br><br>",
    ]
    for i in range(n_cfs):
        lines.append(f"<b>Counterfactual {i+1}:</b><br>")
        any_change = False
        for f in DICE_LABELS:
            if f not in cf_df.columns: continue
            orig_v = float(original[f])
            cf_v   = float(cf_df[f].iloc[i])
            if abs(cf_v - orig_v) <= 0.01: continue
            any_change = True
            arrow = "↓" if cf_v < orig_v else "↑"
            unit  = DICE_UNITS.get(f, '')
            lines.append(
                f"&nbsp;&nbsp;• <b>{DICE_LABELS[f]}</b>: "
                f"<span class='cf-original'>{orig_v:.1f}{unit}</span> "
                f"<span class='cf-arrow'>→</span> "
                f"<span class='cf-target'>{cf_v:.1f}{unit}</span> "
                f"<span style='color:#8a9db5'>({arrow})</span><br>"
            )
        if not any_change:
            lines.append("&nbsp;&nbsp;<i>(no changes — near decision boundary)</i><br>")
        lines.append("<br>")
    lines.append("<i>These are <b>actionable targets</b>: clinical interventions that move "
                 "these values toward the counterfactual could reduce predicted disease risk.</i>")
    return "".join(lines)

def make_rec_delta_chart(ranked_actions: list) -> go.Figure:
    if not ranked_actions:
        return None
    labels      = [a["title"] for a in ranked_actions]
    user_vals   = [a["user_val"] for a in ranked_actions]
    target_vals = [a["target_val"] for a in ranked_actions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Your Values', x=labels, y=user_vals,
        marker=dict(color='#dc2626', opacity=0.75, line=dict(width=0)),
        text=[f"{v:.1f}" for v in user_vals], textposition='outside',
        textfont=dict(color='#dc2626', size=10),
    ))
    fig.add_trace(go.Bar(
        name='Healthy Target', x=labels, y=target_vals,
        marker=dict(color='#059669', opacity=0.75, line=dict(width=0)),
        text=[f"{v:.1f}" for v in target_vals], textposition='outside',
        textfont=dict(color='#059669', size=10),
    ))
    fig.update_layout(
        barmode='group',
        xaxis=dict(tickfont=dict(color='#1a2535', size=11), gridcolor='rgba(0,0,0,0.04)'),
        yaxis=dict(tickfont=dict(color='#7a95b0', size=10), gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        legend=dict(font=dict(color='#4a607a'), orientation='h', y=-0.18, x=0.5, xanchor='center'),
        height=280, **PLOTLY_BASE,
    )
    return fig

# ───────────────────────────────────────────────
# CHATBOT CORE
# ───────────────────────────────────────────────
def process_and_respond(text: str):
    if not text or not text.strip():
        return
    text = text.strip()
    st.session_state.history.append({"role": "user", "content": text})

    new_lang = detect_language(text, current_lang=st.session_state.lang)
    if new_lang in ("en", "hi", "hi-en"):
        st.session_state.lang = new_lang
    lang = st.session_state.lang

    symptoms = extract_symptoms(text)
    inputs_c = symptoms_to_features(symptoms)

    if MODEL_OK:
        df_c  = build_features(inputs_c, feature_cols)
        pred  = model.predict(df_c)[0]
        prob  = model.predict_proba(df_c)[0][1]
    else:
        pred, prob = 0, 0.0

    reply = generate_response(
        user_input=text,
        prediction=pred,
        probability=prob,
        history=st.session_state.history[-5:],
        lang=lang,
    )
    st.session_state.history.append({"role": "assistant", "content": reply})

# ───────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Patient Profile")
    st.markdown("---")

    st.markdown("### Demographics")
    age = st.slider("Age", 20, 80, 55)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)

    st.markdown("### Cardiac Indicators")
    cp_map  = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal", 4: "Asymptomatic"}
    cp      = st.selectbox("Chest Pain Type", list(cp_map.keys()), format_func=lambda x: f"{x} – {cp_map[x]}")
    thalach = st.slider("Max Heart Rate", 70, 210, 150)
    exang   = st.radio("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    oldpeak = st.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
    slope_m = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    slope   = st.selectbox("ST Slope", list(slope_m.keys()), format_func=lambda x: f"{x} – {slope_m[x]}")

    st.markdown("### Lab Results")
    trestbps = st.slider("Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.radio("Fasting Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                             format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])

    st.markdown("### Additional Tests")
    ca      = st.slider("Major Vessels (0-3)", 0, 3, 0)
    thal_m  = {3: "Normal", 6: "Fixed Defect", 7: "Reversible"}
    thal    = st.selectbox("Thalassemia", list(thal_m.keys()), format_func=lambda x: f"{x} – {thal_m[x]}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history       = []
        st.session_state.lang          = "en"
        st.session_state.last_audio_id = None
        st.session_state.pending_voice = None
        st.rerun()

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.78rem;color:#7a95b0;">Logged in as <b style="color:#1a2535">{st.session_state.get("username","")}</b></div>', unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#8a9db5;line-height:1.6">'
        '⚕️ This tool is for educational purposes only. '
        'Always consult a qualified healthcare professional.</div>',
        unsafe_allow_html=True,
    )

# ───────────────────────────────────────────────
# ASSEMBLE INPUTS + PREDICT
# ───────────────────────────────────────────────
inputs = dict(
    age=float(age), sex=float(sex), cp=float(cp),
    trestbps=float(trestbps), chol=float(chol), fbs=float(fbs),
    restecg=float(restecg), thalach=float(thalach), exang=float(exang),
    oldpeak=float(oldpeak), slope=float(slope), ca=float(ca), thal=float(thal),
)

if MODEL_OK:
    X_input = build_features(inputs, feature_cols)
    prob    = float(model.predict_proba(X_input)[0][1])
    pred    = int(model.predict(X_input)[0])
else:
    prob, pred = 0.35, 0
    X_input    = None

risk_cls  = "high" if prob >= 0.6 else ("medium" if prob >= 0.35 else "low")
risk_lbl  = "High Risk" if prob >= 0.6 else ("Moderate" if prob >= 0.35 else "Low Risk")
ds_val    = inputs['cp'] + inputs['exang'] + inputs['oldpeak'] + inputs['ca']
hr_pct    = int((inputs['thalach'] / 220) * 100)
chol_lbl  = "Normal" if chol < 200 else ("Borderline" if chol < 240 else "High")
chol_cls  = "success" if chol < 200 else ("warning" if chol < 240 else "danger")

# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.markdown(f"""
<div class="hub-banner">
  <div class="banner-badge">🏥 Cardiovascular Intelligence Platform</div>
  <h1><span class="beat">🫀</span> CVD Intelligence Hub</h1>
  <p>Predict · Prevent · Protect &nbsp;·&nbsp; Integrated risk prediction · SHAP explainability · DiCE counterfactuals · AI chatbot · Personalised recommendations</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error("⚠️ Model files not found. Run `python train_save_model.py` first, then restart.")
    st.stop()

# ───────────────────────────────────────────────
# KPI ROW
# ───────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card kpi-{'danger' if prob>=0.6 else ('warning' if prob>=0.35 else 'success')}">
    <div class="kpi-label">Disease Probability</div>
    <div class="kpi-value">{prob*100:.1f}%</div>
    <div class="kpi-sub"><span class="risk-badge {risk_cls}">{risk_lbl}</span></div>
    </div>""", unsafe_allow_html=True)
with k2:
    ds_cls = "danger" if ds_val >= 6 else ("warning" if ds_val >= 3 else "success")
    st.markdown(f"""<div class="kpi-card kpi-{ds_cls}">
    <div class="kpi-label">Disease Score</div>
    <div class="kpi-value">{ds_val:.1f}</div>
    <div class="kpi-sub">CP + Exang + OldPeak + CA</div>
    </div>""", unsafe_allow_html=True)
with k3:
    hr_cls = "success" if hr_pct >= 70 else ("warning" if hr_pct >= 55 else "danger")
    st.markdown(f"""<div class="kpi-card kpi-{hr_cls}">
    <div class="kpi-label">Max HR % of Predicted</div>
    <div class="kpi-value">{hr_pct}%</div>
    <div class="kpi-sub">{int(thalach)} bpm recorded</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card kpi-{chol_cls}">
    <div class="kpi-label">Cholesterol Status</div>
    <div class="kpi-value">{chol_lbl}</div>
    <div class="kpi-sub">{int(chol)} mg/dl</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# THREE TABS
# ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Prediction Dashboard",
    "🤖  CVD Chatbot",
    "💡  Recommendation System",
])

# ══════════════════════════════════════════
# TAB 1 — PREDICTION DASHBOARD
# ══════════════════════════════════════════
with tab1:

    g_col, b_col = st.columns([1, 1.7])

    with g_col:
        st.markdown('<div class="sec-card"><h3>Risk Gauge</h3>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(prob), use_container_width=True, config={'displayModeBar': False})
        box_cls = "warn-box" if pred == 1 else "success-box"
        msg     = (f"<b>Heart Disease Detected</b> — confidence {prob*100:.1f}%. "
                   "Cardiology review strongly recommended."
                   if pred == 1 else
                   f"<b>No Heart Disease Detected</b> — confidence {(1-prob)*100:.1f}%. "
                   "Continue routine monitoring.")
        st.markdown(f'<div class="{box_cls}">{msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b_col:
        st.markdown('<div class="sec-card"><h3>Individual Risk Factor Contributions</h3>', unsafe_allow_html=True)
        st.plotly_chart(make_risk_bars(inputs), use_container_width=True, config={'displayModeBar': False})
        st.markdown('<div class="info-box">🔴 ≥ 60% High &nbsp;·&nbsp; 🟡 35–60% Moderate &nbsp;·&nbsp; 🟢 &lt; 35% Low</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-card"><h3>🔍 Explainable AI — SHAP Feature Impact</h3>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">SHAP values show each feature\'s contribution to the prediction. '
                '<span style="color:#dc2626;font-weight:600">Red = increases risk</span> &nbsp;|&nbsp; '
                '<span style="color:#2563eb;font-weight:600">Blue = decreases risk</span><br>'
                '<span style="color:#8a9db5;font-size:0.8rem">Method: TreeExplainer on LightGBM base learner (fast, exact)</span></div>',
                unsafe_allow_html=True)

    if shap_bundle and X_input is not None:
        with st.spinner("Computing SHAP values…"):
            try:
                if shap_bundle["kind"] == "tree":
                    sv_raw = shap_bundle["explainer"].shap_values(X_input)
                    if isinstance(sv_raw, list):
                        shap_vals = np.array(sv_raw[1]).flatten()
                    else:
                        sv_arr = np.array(sv_raw)
                        shap_vals = sv_arr[0, :, 1] if sv_arr.ndim == 3 else sv_arr.flatten()
                    ev = shap_bundle["explainer"].expected_value
                    base_val = float(ev[1] if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0 else ev)
                else:
                    sv_raw    = shap_bundle["explainer"].shap_values(X_input.values, nsamples=100, silent=True)
                    shap_vals = np.array(sv_raw).flatten()
                    base_val  = float(shap_bundle["explainer"].expected_value)

                sh_left, sh_right = st.columns([1.6, 1.4])
                with sh_left:
                    st.plotly_chart(make_shap_chart(shap_vals, feature_cols, X_input),
                                    use_container_width=True, config={'displayModeBar': False})
                with sh_right:
                    nl_html = natural_language_shap(shap_vals, feature_cols, X_input, pred, prob)
                    box_cls = "nl-explain-risk" if pred == 1 else "nl-explain"
                    st.markdown(f'<div class="{box_cls}">{nl_html}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"SHAP computation skipped: {e}")
    else:
        st.info("SHAP explainer not available (shap_background.pkl missing). Run training script to enable.")

    st.markdown('</div>', unsafe_allow_html=True)

    if not DICE_OK:
        st.markdown('<div class="warn-box">⚠️ DiCE engine not available. Ensure <code>processed.cleveland.data</code> is present and dependencies are installed.</div>',
                    unsafe_allow_html=True)
    elif pred == 0:
        st.markdown("""
        <div class="success-box">
        <b>No Disease Detected</b> — counterfactuals are generated only when the model predicts disease,
        as they show what would need to change to flip a high-risk prediction to low-risk.<br>
        <span style="font-size:0.8rem">Adjust the sidebar to a higher-risk profile
        (higher chest-pain type, exercise angina, blocked vessels, ST depression) to see DiCE in action.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        dice_query = build_dice_query(
            inputs,
            dice_bundle["feature_cols"],
            dice_bundle["categorical_cols"],
            dice_bundle["continuous_cols"],
        )
        dice_pred = int(dice_bundle["model"].predict(dice_query)[0])

        if dice_pred == 0:
            st.markdown("""
            <div class="info-box">
            <b>Note:</b> The raw-feature Stack6 model used by DiCE predicts <b>No Disease</b>
            for this patient (the main engineered-feature model predicted Disease).
            Adjust the sidebar to a higher-risk profile to generate counterfactuals.
            </div>
            """, unsafe_allow_html=True)
        else:
            permitted       = {'age':[20,80], 'trestbps':[80,200], 'chol':[100,400],
                               'thalach':[70,210], 'oldpeak':[0.0,6.0], 'disease_score':[0.0,12.0]}
            features_to_vary = ['trestbps', 'chol', 'thalach', 'oldpeak', 'disease_score']

            cf_result = None
            cf_error  = None
            with st.spinner("🎲 Generating diverse counterfactuals (10–20 s)…"):
                for n_cfs in [3, 2, 1]:
                    try:
                        cf_result = dice_bundle["explainer"].generate_counterfactuals(
                            query_instances  = dice_query,
                            total_CFs        = n_cfs,
                            desired_class    = 0,
                            permitted_range  = permitted,
                            features_to_vary = features_to_vary,
                        )
                        break
                    except Exception as e:
                        cf_error = e
                        continue

            if cf_result is None:
                st.markdown(f"""
                <div class="warn-box">
                <b>DiCE could not generate counterfactuals for this patient.</b><br>
                The patient profile may be too deep inside the high-risk region for any
                small change in the permitted features to flip the prediction.<br>
                <span style="font-size:0.8rem;color:#8a9db5">Last error: {cf_error}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                if 'target' in cf_df.columns:
                    cf_df = cf_df.drop(columns=['target'])

                dice_left, dice_right = st.columns([1.6, 1.4])

                with dice_left:
                    st.markdown("**Current Patient vs Counterfactual Targets**")
                    st.markdown(
                        '<div class="info-box" style="margin-bottom:10px">'
                        '<span style="color:#dc2626;font-weight:600">Red = current values</span> &nbsp;|&nbsp; '
                        '<span style="color:#059669;font-weight:600">Green/Blue/Purple = counterfactual targets</span><br>'
                        'Only features DiCE actually changed are shown.</div>',
                        unsafe_allow_html=True
                    )
                    fig_dice = make_dice_chart(dice_query, cf_df, dice_bundle["feature_cols"])
                    st.plotly_chart(fig_dice, use_container_width=True, config={'displayModeBar': False})

                with dice_right:
                    st.markdown("**Counterfactual Explanation**")
                    dice_nl = dice_natural_language(dice_query, cf_df, pred, prob)
                    st.markdown(f'<div class="nl-explain-risk">{dice_nl}</div>', unsafe_allow_html=True)

                with st.expander("📋 Counterfactual Detail Table — Full Feature Values", expanded=False):
                    display_df = pd.concat(
                        [dice_query.reset_index(drop=True), cf_df.reset_index(drop=True)],
                        ignore_index=True,
                    )
                    display_df.index = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]
                    st.dataframe(display_df.round(2), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-card"><h3>Clinical Summary</h3>', unsafe_allow_html=True)
    cs1, cs2, cs3 = st.columns(3)

    bp_status  = "Normal" if trestbps < 120 else ("Elevated" if trestbps < 130 else ("Stage 1 HTN" if trestbps < 140 else "Stage 2 HTN"))
    ecg_labels = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]

    with cs1:
        st.markdown("**Vital Signs**")
        st.markdown(f"""
| Metric | Value | Status |
|--------|-------|--------|
| Blood Pressure | {int(trestbps)} mmHg | {bp_status} |
| Max Heart Rate | {int(thalach)} bpm | {'Achieved' if thalach >= 150 else 'Low'} |
| ST Depression  | {oldpeak} | {'Abnormal' if oldpeak > 2 else 'Normal'} |
| Resting ECG    | — | {ecg_labels[int(restecg)]} |
""")
    with cs2:
        st.markdown("**Lab Values**")
        chol_label = "Desirable" if chol < 200 else ("Borderline High" if chol < 240 else "High")
        st.markdown(f"""
| Metric | Value | Status |
|--------|-------|--------|
| Cholesterol | {int(chol)} mg/dl | {chol_label} |
| Fasting Sugar | — | {'Elevated' if fbs == 1 else 'Normal'} |
| Vessels (Fluoro) | {int(ca)} | {f'{int(ca)} blocked' if ca > 0 else 'Clear'} |
| Thalassemia | — | {thal_m[int(thal)]} |
""")
    with cs3:
        st.markdown("**Clinical Flags**")
        flags = []
        if prob >= 0.6:    flags.append(("🔴", "HIGH disease probability — urgent review"))
        if trestbps > 140: flags.append(("🟠", "Hypertension detected"))
        if chol > 240:     flags.append(("🟠", "High cholesterol"))
        if oldpeak > 2:    flags.append(("🟡", "Significant ST depression"))
        if exang == 1:     flags.append(("🟡", "Exercise-induced angina"))
        if ca >= 2:        flags.append(("🔴", "Multiple vessel blockages"))
        if thal == 7:      flags.append(("🟠", "Reversible thalassemia defect"))
        if not flags:      flags.append(("✅", "No major risk flags detected"))
        for icon, txt in flags:
            st.markdown(f"{icon} {txt}")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — CHATBOT
# ══════════════════════════════════════════
with tab2:

    st.markdown('<div class="sec-card"><h3>Voice Input</h3>', unsafe_allow_html=True)
    st.markdown("🎤 *Speak your symptoms — the assistant will assess your cardiovascular risk.*")

    audio = st.audio_input("Click the mic icon to record")
    if audio is not None:
        audio_bytes = audio.read()
        audio_id    = hash(audio_bytes)
        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id
            with st.spinner("🎧 Transcribing…"):
                transcribed = transcribe_audio(audio_bytes)
            if transcribed and transcribed.strip():
                st.session_state.pending_voice = transcribed
                process_and_respond(transcribed)
            else:
                st.warning("⚠️ Could not understand. Please speak clearly and try again.")

    if st.session_state.pending_voice:
        st.success(f"✅ Heard: {st.session_state.pending_voice}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-card"><h3>Conversation</h3>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<div class="info-box">👋 Hello! Describe your symptoms and I\'ll assess your cardiovascular risk. You can type or use the microphone above.</div>', unsafe_allow_html=True)
    else:
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="info-box" style="font-size:0.78rem">🌐 Language detected: <b>{st.session_state.lang or "Not set"}</b></div>', unsafe_allow_html=True)

    user_input = st.chat_input("Type your symptoms here…")
    if user_input:
        st.session_state.pending_voice = None
        process_and_respond(user_input)
        st.rerun()

# ══════════════════════════════════════════
# TAB 3 — RECOMMENDATION SYSTEM
# ══════════════════════════════════════════
with tab3:

    rec_result = get_recommendations(inputs, rec_bundle, prob)
    feedback_label, feedback_msg, feedback_type = rec_result["feedback"]
    box_map = {"success": "success-box", "info": "info-box", "warning": "warn-box", "danger": "warn-box"}

    st.markdown(f"""
    <div class="{box_map[feedback_type]}">
    <b>{feedback_label}</b> — {feedback_msg}<br>
    <span style="font-size:0.78rem;opacity:0.7">
    Based on {rec_result['n_healthy_neighbors']} similar healthy patients
    via hybrid content + collaborative filtering.
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r_left, r_right = st.columns([1.3, 1.7])

    with r_left:
        st.markdown('<div class="sec-card"><h3>Personalised Action Plan</h3>', unsafe_allow_html=True)

        if not rec_result["ranked_actions"]:
            st.markdown('<div class="success-box">✅ Your values are close to the healthy patient profile. Maintain your current lifestyle!</div>', unsafe_allow_html=True)
        else:
            for i, action in enumerate(rec_result["ranked_actions"], 1):
                status_color = "#dc2626" if action["status"] == "HIGH" else "#d97706"
                delta_arrow  = "↓ needs reduction" if action["status"] == "HIGH" else "↑ needs improvement"
                st.markdown(f"""
                <div class="rec-card">
                  <div class="rec-header">
                    <div class="rec-icon" style="background:{action['bg']}">{action['icon']}</div>
                    {i}. {action['title']}
                    <span style="font-size:0.72rem;color:{status_color};margin-left:auto;font-weight:600">{delta_arrow}</span>
                  </div>
                  <ul>{''.join(f'<li>{a}</li>' for a in action['actions'])}</ul>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-card"><h3>Medical Guidance</h3>', unsafe_allow_html=True)
        if prob < 0.2:
            st.markdown("""<div class="success-box"><b>Health Maintenance</b><br>
• Continue balanced diet rich in fruits & vegetables<br>
• Maintain regular physical activity and proper sleep<br>
• Stay hydrated and limit junk food<br>
• Annual health checkups recommended</div>""", unsafe_allow_html=True)
        elif prob < 0.6:
            st.markdown("""<div class="info-box"><b>Lifestyle Improvement Recommended</b><br>
• Schedule a cardiac risk assessment with your GP<br>
• Track blood pressure and cholesterol quarterly<br>
• Follow a low-fat, low-sodium, high-fibre diet<br>
• Aim for 150 min moderate exercise per week</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="warn-box"><b>Immediate Medical Attention Required</b><br>
• Schedule urgent cardiology consultation<br>
• Monitor blood pressure, cholesterol & sugar daily<br>
• Strict low-fat, low-salt diet — avoid alcohol & smoking<br>
• Do NOT start high-intensity exercise without clearance<br>
• Adhere strictly to any prescribed medication</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r_right:
        st.markdown('<div class="sec-card"><h3>Your Values vs Healthy Target</h3>', unsafe_allow_html=True)
        fig_delta = make_rec_delta_chart(rec_result["ranked_actions"])
        if fig_delta:
            st.plotly_chart(fig_delta, use_container_width=True, config={'displayModeBar': False})
            st.markdown('<div class="info-box">Red bars = your current values · Green bars = average of similar healthy patients via hybrid recommendation</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">No significant gap found between your values and the healthy target profile. Great work!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-card"><h3>Risk Score Categorisation</h3>', unsafe_allow_html=True)
        risk_data = {
            "Healthy (< 20%)":    "#059669",
            "Low Risk (20–40%)":  "#2563eb",
            "Moderate (40–60%)":  "#d97706",
            "High Risk (60–80%)": "#dc2626",
            "Critical (> 80%)":   "#991b1b",
        }
        active_idx = min(int(prob * 5), 4)
        for idx_r, (lbl, col) in enumerate(zip(risk_data.keys(), risk_data.values())):
            active = idx_r == active_idx
            r_int = int(col[1:3], 16)
            g_int = int(col[3:5], 16)
            b_int = int(col[5:7], 16)
            bg     = f"rgba({r_int},{g_int},{b_int},{'0.12' if active else '0.04'})"
            border = f"1px solid {col}" if active else f"1px solid rgba({r_int},{g_int},{b_int},0.2)"
            weight = "700" if active else "400"
            arrow  = " ◀ You are here" if active else ""
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:8px;
                        padding:8px 14px;margin-bottom:6px;font-size:0.82rem;
                        color:{col};font-weight:{weight}">
              {lbl}{arrow}
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 Full Feature Comparison: You vs Healthy Patients", expanded=False):
        diff_df = rec_result["diff"].reset_index()
        diff_df.columns = ["Feature", "Delta (Healthy − You)"]
        diff_df["Direction"] = diff_df["Delta (Healthy − You)"].apply(
            lambda x: "↑ Needs increase" if x > 0.05 else ("↓ Needs reduction" if x < -0.05 else "✅ On target")
        )
        diff_df["Delta (Healthy − You)"] = diff_df["Delta (Healthy − You)"].round(4)
        diff_df["Feature"] = diff_df["Feature"].apply(lambda x: LABEL_MAP.get(x, x))
        st.dataframe(diff_df, use_container_width=True, hide_index=True)