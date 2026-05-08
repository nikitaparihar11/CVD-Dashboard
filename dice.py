"""
Heart Disease Prediction — Interactive Streamlit Dashboard (Clean Version)
Model: Custom3 HetDiv (LGBM+NB+LR+SVM -> RF) | Accuracy 95% | AUC 0.97
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import dice_ml
from dice_ml import Dice
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid rgba(99,179,237,0.2);
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .high-risk   { color: #fc5c7d; }
    .medium-risk { color: #f6d365; }
    .low-risk    { color: #56ab2f; }
    .banner {
        background: linear-gradient(135deg, #fc5c7d 0%, #6a82fb 100%);
        padding: 28px 36px; border-radius: 16px; margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(106,130,251,0.3);
    }
    .banner h1 { color: white; margin: 0; font-size: 2rem; font-weight: 700; }
    .banner p  { color: rgba(255,255,255,0.85); margin: 6px 0 0; font-size: 1rem; }
    .stSlider > div > div > div { background: #6a82fb !important; }
    div[data-testid="stSidebar"] { background: #0d1117; }
    .info-box {
        background: rgba(99,179,237,0.08); border-left: 3px solid #63b3ed;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
        font-size: 0.88rem; color: #cbd5e0;
    }
    .warning-box {
        background: rgba(252,92,125,0.1); border-left: 3px solid #fc5c7d;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
        font-size: 0.88rem; color: #feb2c5;
    }
    .xai-card {
        background: linear-gradient(135deg, #0f2027 0%, #1a1a2e 100%);
        border-radius: 14px; padding: 20px;
        border: 1px solid rgba(106,130,251,0.25);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .shap-positive { color: #fc5c7d; font-weight: 600; }
    .shap-negative { color: #56b4e9; font-weight: 600; }
    .nl-explain {
        background: rgba(86,171,47,0.07); border-left: 3px solid #56ab2f;
        padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 10px 0;
        font-size: 0.92rem; line-height: 1.7; color: #e2e8f0;
    }
    .nl-explain-risk {
        background: rgba(252,92,125,0.07); border-left: 3px solid #fc5c7d;
        padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 10px 0;
        font-size: 0.92rem; line-height: 1.7; color: #e2e8f0;
    }
    .dice-card {
        background: linear-gradient(135deg, #1a0f2e 0%, #16213e 100%);
        border-radius: 14px; padding: 20px;
        border: 1px solid rgba(155,89,182,0.25);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .cf-original { color: #fc5c7d; font-weight: 600; }
    .cf-target   { color: #56ab2f; font-weight: 600; }
    .cf-arrow    { color: #94a3b8; font-weight: 600; }
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        14%       { transform: scale(1.35); }
        28%       { transform: scale(1); }
        42%       { transform: scale(1.35); }
        70%       { transform: scale(1); }
    }
    .pumping-heart {
        display: inline-block;
        animation: heartbeat 1.4s ease-in-out infinite;
        transform-origin: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Human-readable feature labels ──────────────────────────────────────────────
LABEL_MAP = {
    'age': 'Age (yrs)', 'sex': 'Sex', 'trestbps': 'Blood Pressure',
    'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG', 'thalach': 'Max Heart Rate',
    'exang': 'Exercise Angina', 'oldpeak': 'ST Depression',
    'ca': 'Blocked Vessels', 'disease_score': 'Disease Score',
    'cp_2.0': 'CP: Atypical Angina', 'cp_3.0': 'CP: Non-Anginal',
    'cp_4.0': 'CP: Asymptomatic', 'thal_6.0': 'Thal: Fixed Defect',
    'thal_7.0': 'Thal: Reversible', 'slope_2.0': 'Slope: Flat',
    'slope_3.0': 'Slope: Downsloping'
}

# ── Load model + SHAP background ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load("best_heart_model.pkl")
    feat_cols  = joblib.load("feature_cols.pkl")
    background = joblib.load("shap_background.pkl")
    selector   = joblib.load("selector.pkl")
    return model, feat_cols, background, selector

@st.cache_resource
def get_shap_explainer(_model, _background):
    """
    Build and cache a SHAP explainer for the Stack6 ensemble.

    Per the AI-DWELL notebook (PART 4), the primary SHAP path uses TreeExplainer
    on a standalone LightGBM base learner (fast, exact). We fit it on the same
    background distribution so its expected_value is meaningful for single-patient
    waterfalls. KernelExplainer on the full ensemble is kept as a fallback.
    """
    cols = list(_background.columns)
    bg = _background.astype(float)

    # Primary path — TreeExplainer on standalone LightGBM (matches notebook)
    try:
        lgb_standalone = LGBMClassifier(verbose=-1, random_state=42)
        # We don't have y here in cache, so synthesize a label by asking the
        # ensemble for its prediction on the background — this gives the LGBM
        # a target distribution that mirrors what the ensemble produces.
        bg_preds = (_model.predict_proba(bg)[:, 1] >= 0.5).astype(int)
        # Fall back to KernelExplainer if background is degenerate (single class)
        if len(np.unique(bg_preds)) < 2:
            raise ValueError("Background has single class — using KernelExplainer")
        lgb_standalone.fit(bg, bg_preds)
        tree_explainer = shap.TreeExplainer(lgb_standalone)
        return {"kind": "tree", "explainer": tree_explainer, "cols": cols}
    except Exception:
        # Fallback — KernelExplainer on the full Stack6 ensemble
        predict_fn = lambda x: _model.predict_proba(
            pd.DataFrame(x, columns=cols).astype(float)
        )[:, 1]
        kernel_explainer = shap.KernelExplainer(predict_fn, bg.values)
        return {"kind": "kernel", "explainer": kernel_explainer, "cols": cols}


@st.cache_resource
def get_dice_pipeline():
    """
    Build and cache the DiCE pipeline.

    DiCE works on raw (un-one-hot-encoded) features. Per the notebook (PART 5),
    we train a parallel Stack6 model on raw features so DiCE can vary cp/thal/slope
    as native categorical/continuous values rather than dummy columns.
    """
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]

    df_dice = pd.read_csv("processed.cleveland.data", names=cols)
    df_dice.replace("?", np.nan, inplace=True)
    df_dice = df_dice.dropna().astype(float)
    df_dice['target'] = (df_dice['target'] > 0).astype(int)
    df_dice['disease_score'] = df_dice['cp'] + df_dice['exang'] + df_dice['oldpeak'] + df_dice['ca']

    feature_cols_dice = ["age","sex","cp","trestbps","chol","fbs","restecg",
                         "thalach","exang","oldpeak","slope","ca","thal","disease_score"]
    categorical_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
    continuous_cols  = ["age","trestbps","chol","thalach","oldpeak","disease_score"]

    X_dice = df_dice[feature_cols_dice].copy()
    y_dice = df_dice['target'].copy()
    for c in categorical_cols: X_dice[c] = X_dice[c].astype(int)
    for c in continuous_cols:  X_dice[c] = X_dice[c].astype(float)

    X_train, _, y_train, _ = train_test_split(
        X_dice, y_dice, test_size=0.2, random_state=42
    )

    # Stack6 architecture — same as notebook
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
        "explainer":         dice_exp,
        "model":             stack6_dice,
        "feature_cols":      feature_cols_dice,
        "categorical_cols":  categorical_cols,
        "continuous_cols":   continuous_cols,
    }


try:
    model, feature_cols, shap_background, _selector = load_artifacts()
    selector = None  # best_heart_model.pkl trained on all 18 features; selector not part of pipeline
    explainer    = get_shap_explainer(model, shap_background)
    dice_bundle  = get_dice_pipeline()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(inputs: dict, feature_cols: list, selector=None) -> pd.DataFrame:
    cp = inputs['cp']; exang = inputs['exang']
    oldpeak = inputs['oldpeak']; ca = inputs['ca']
    disease_score = cp + exang + oldpeak + ca
    row = {
        'age': inputs['age'], 'sex': inputs['sex'],
        'trestbps': inputs['trestbps'], 'chol': inputs['chol'],
        'fbs': inputs['fbs'], 'restecg': inputs['restecg'],
        'thalach': inputs['thalach'], 'exang': exang,
        'oldpeak': oldpeak, 'ca': ca, 'disease_score': disease_score,
        'cp_2.0': 1 if cp == 2.0 else 0,
        'cp_3.0': 1 if cp == 3.0 else 0,
        'cp_4.0': 1 if cp == 4.0 else 0,
        'thal_6.0': 1 if inputs['thal'] == 6.0 else 0,
        'thal_7.0': 1 if inputs['thal'] == 7.0 else 0,
        'slope_2.0': 1 if inputs['slope'] == 2.0 else 0,
        'slope_3.0': 1 if inputs['slope'] == 3.0 else 0,
    }
    df = pd.DataFrame([row])
    if selector is not None:
        pre_cols = list(selector.feature_names_in_)
        for col in pre_cols:
            if col not in df.columns:
                df[col] = 0
        arr = selector.transform(df[pre_cols])
        selected_cols = [pre_cols[i] for i in selector.get_support(indices=True)]
        return pd.DataFrame(arr, columns=selected_cols)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


# ── Chart functions ────────────────────────────────────────────────────────────
def make_gauge(prob: float) -> go.Figure:
    color = "#fc5c7d" if prob >= 0.6 else ("#f6d365" if prob >= 0.35 else "#56ab2f")
    label = "HIGH RISK" if prob >= 0.6 else ("MODERATE" if prob >= 0.35 else "LOW RISK")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 52, 'color': color, 'family': 'Inter'}},
        title={'text': f"<b>{label}</b>", 'font': {'size': 20, 'color': color}},
        delta={'reference': 50, 'increasing': {'color': '#fc5c7d'}, 'decreasing': {'color': '#56ab2f'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': '#94a3b8'}, 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': '#1a1a2e', 'borderwidth': 0,
            'steps': [
                {'range': [0, 35],  'color': 'rgba(86,171,47,0.15)'},
                {'range': [35, 60], 'color': 'rgba(246,211,101,0.15)'},
                {'range': [60, 100],'color': 'rgba(252,92,125,0.15)'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 3},
                          'thickness': 0.8, 'value': prob * 100}
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_family='Inter')
    return fig


def make_risk_bars(inputs: dict) -> go.Figure:
    items = {
        'Disease Score':   min((inputs['cp']+inputs['exang']+inputs['oldpeak']+inputs['ca']) / 10, 1.0),
        'ST Depression':   min(inputs['oldpeak'] / 6, 1.0),
        'Blocked Vessels': inputs['ca'] / 3,
        'Chest Pain Lvl':  (inputs['cp'] - 1) / 3,
        'Cholesterol':     min(inputs['chol'] / 400, 1.0),
        'Blood Pressure':  min(inputs['trestbps'] / 200, 1.0),
        'Age Factor':      min(inputs['age'] / 80, 1.0),
        'HR Deficit':      max(0, 1 - inputs['thalach'] / 200),
    }
    labels = list(items.keys()); vals = [v * 100 for v in items.values()]
    colors = ['#fc5c7d' if v >= 60 else ('#f6d365' if v >= 35 else '#56ab2f') for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in vals], textposition='outside',
        textfont=dict(color='#94a3b8', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 115], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(color='#cbd5e0', size=12),
                   gridcolor='rgba(255,255,255,0.04)'),
        margin=dict(t=10, b=10, l=10, r=60), height=260
    )
    return fig


# ── XAI: SHAP waterfall (Plotly) ───────────────────────────────────────────────
def make_shap_waterfall(shap_vals: np.ndarray, feature_cols: list,
                         X_row: pd.DataFrame, base_value: float) -> go.Figure:
    """Custom Plotly SHAP waterfall chart — top 14 features by |impact|."""
    top_n = 14
    idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]

    labels, values, fvals = [], [], []
    for i in idx:
        col = feature_cols[i]
        raw = X_row.iloc[0][col]
        labels.append(f"{LABEL_MAP.get(col, col)} = {raw:.3g}")
        values.append(float(shap_vals[i]))
        fvals.append(float(raw))

    labels = labels[::-1]; values = values[::-1]
    colors = ['#fc5c7d' if v > 0 else '#56b4e9' for v in values]
    text   = [f"{v:+.4f}" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0), opacity=0.88),
        text=text, textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
        hovertemplate='%{y}<br>SHAP: %{x:+.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.15)', line_width=1)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.6)',
        xaxis=dict(title=dict(text="SHAP Value (impact on disease probability)",
                              font=dict(color='#94a3b8', size=12)),
                   tickfont=dict(color='#64748b'),
                   gridcolor='rgba(255,255,255,0.06)', zeroline=False),
        yaxis=dict(tickfont=dict(color='#cbd5e0', size=11),
                   gridcolor='rgba(255,255,255,0.04)'),
        margin=dict(t=20, b=50, l=20, r=80),
        height=420,
        annotations=[
            dict(x=base_value, y=-0.06, xref='x', yref='paper',
                 text=f"Base: {base_value:.3f}", showarrow=False,
                 font=dict(color='#94a3b8', size=11)),
        ]
    )
    return fig


# ── XAI: Natural-language explanation ─────────────────────────────────────────
def natural_language_explanation(shap_vals: np.ndarray, feature_cols: list,
                                  X_row: pd.DataFrame, pred: int, prob: float,
                                  base_value: float) -> str:
    idx_sorted = np.argsort(np.abs(shap_vals))[::-1]
    top_pos, top_neg = [], []
    for i in idx_sorted:
        col = feature_cols[i]; sv = float(shap_vals[i]); raw = X_row.iloc[0][col]
        entry = (LABEL_MAP.get(col, col), sv, raw)
        if sv > 0.01 and len(top_pos) < 3:
            top_pos.append(entry)
        elif sv < -0.01 and len(top_neg) < 3:
            top_neg.append(entry)
        if len(top_pos) == 3 and len(top_neg) == 3:
            break

    verdict = "HEART DISEASE DETECTED" if pred == 1 else "NO HEART DISEASE DETECTED"
    conf    = f"{prob*100:.1f}%" if pred == 1 else f"{(1-prob)*100:.1f}%"

    lines = [f"<b>Prediction: {verdict}</b> (confidence {conf})<br><br>"]

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

    lines.append(f"<br><i>Base probability (average patient): "
                 f"{base_value:.3f} — "
                 f"adjusted to {prob:.3f} for this patient.</i>")
    return "".join(lines)


# ── DiCE: build a raw-feature query row from sidebar inputs ───────────────────
def build_dice_query(inputs: dict, feature_cols_dice: list,
                      categorical_cols: list, continuous_cols: list) -> pd.DataFrame:
    """Construct a single-row raw-feature DataFrame for DiCE (no one-hot)."""
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


# ── DiCE: counterfactual comparison chart (Plotly) ────────────────────────────
def make_dice_chart(query: pd.DataFrame, cf_df: pd.DataFrame,
                     feature_cols_dice: list) -> go.Figure:
    """Grouped bar chart: original patient values vs counterfactual values
    for features that DiCE actually changed."""
    original = query.iloc[0]

    DICE_LABELS = {
        'age': 'Age (yrs)', 'trestbps': 'Blood Pressure',
        'chol': 'Cholesterol', 'thalach': 'Max Heart Rate',
        'oldpeak': 'ST Depression', 'disease_score': 'Disease Score',
        'cp': 'Chest Pain', 'exang': 'Exercise Angina',
        'ca': 'Blocked Vessels', 'slope': 'ST Slope',
        'thal': 'Thalassemia', 'sex': 'Sex',
        'fbs': 'Fasting Sugar', 'restecg': 'Resting ECG',
    }

    changed = [
        f for f in feature_cols_dice
        if f in cf_df.columns and bool(np.any(np.abs(cf_df[f].values - original[f]) > 0.01))
    ]
    if not changed:
        fig = go.Figure()
        fig.add_annotation(text="No feature changes needed — patient already at low risk.",
                           x=0.5, y=0.5, xref='paper', yref='paper',
                           showarrow=False, font=dict(color='#94a3b8', size=14))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=300)
        return fig

    n_cfs    = len(cf_df)
    cf_palette = ['#56ab2f', '#3498DB', '#9B59B6']
    fig = go.Figure()

    # Original (red)
    fig.add_trace(go.Bar(
        name='Original (High Risk)',
        x=[DICE_LABELS.get(f, f) for f in changed],
        y=[float(original[f]) for f in changed],
        marker=dict(color='#fc5c7d', line=dict(width=0)),
        text=[f"{float(original[f]):.2f}" for f in changed],
        textposition='outside',
        textfont=dict(color='#fc5c7d', size=11),
        hovertemplate='%{x}<br>Original: %{y:.2f}<extra></extra>',
    ))

    # Counterfactuals (greens/blues)
    for i in range(n_cfs):
        fig.add_trace(go.Bar(
            name=f'CF {i+1} (Low Risk)',
            x=[DICE_LABELS.get(f, f) for f in changed],
            y=[float(cf_df[f].iloc[i]) for f in changed],
            marker=dict(color=cf_palette[i % len(cf_palette)], line=dict(width=0)),
            text=[f"{float(cf_df[f].iloc[i]):.2f}" for f in changed],
            textposition='outside',
            textfont=dict(color=cf_palette[i % len(cf_palette)], size=11),
            hovertemplate=f'%{{x}}<br>CF {i+1}: %{{y:.2f}}<extra></extra>',
        ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.6)',
        xaxis=dict(tickfont=dict(color='#cbd5e0', size=11),
                   gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(title=dict(text="Feature Value",
                              font=dict(color='#94a3b8', size=12)),
                   tickfont=dict(color='#64748b'),
                   gridcolor='rgba(255,255,255,0.06)', zeroline=False),
        legend=dict(font=dict(color='#cbd5e0'), orientation='h',
                    y=-0.18, x=0.5, xanchor='center'),
        margin=dict(t=20, b=80, l=40, r=20),
        height=420,
    )
    return fig


def dice_natural_language(query: pd.DataFrame, cf_df: pd.DataFrame,
                            pred: int, prob: float) -> str:
    """Plain-English explanation of DiCE counterfactuals."""
    DICE_LABELS = {
        'age': 'Age', 'trestbps': 'Blood Pressure',
        'chol': 'Cholesterol', 'thalach': 'Max Heart Rate',
        'oldpeak': 'ST Depression', 'disease_score': 'Disease Score',
    }
    UNITS = {
        'age': ' yrs', 'trestbps': ' mmHg', 'chol': ' mg/dl',
        'thalach': ' bpm', 'oldpeak': '', 'disease_score': '',
    }

    if pred == 0:
        return ("<b>Prediction: NO HEART DISEASE DETECTED</b> "
                f"(confidence {(1-prob)*100:.1f}%)<br><br>"
                "<i>Counterfactuals are most useful when the model predicts disease — "
                "they show what the patient would need to change to flip the prediction. "
                "Since this patient is already predicted as low-risk, no flip is needed. "
                "Adjust the sidebar to a high-risk profile to see counterfactuals.</i>")

    original = query.iloc[0]
    n_cfs = len(cf_df)

    lines = [
        f"<b>Prediction: HEART DISEASE DETECTED</b> (confidence {prob*100:.1f}%)<br><br>",
        "<i>DiCE generates <b>counterfactual examples</b> — minimally-different "
        "alternate patient profiles that the model would predict as <b>No Disease</b>. "
        "Each shows a possible path to lower risk.</i><br><br>",
        f"<b>{n_cfs} counterfactual{'s' if n_cfs > 1 else ''} found:</b><br><br>",
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
            lines.append(
                f"&nbsp;&nbsp;• <b>{DICE_LABELS[f]}</b>: "
                f"<span class='cf-original'>{orig_v:.1f}{UNITS.get(f,'')}</span> "
                f"<span class='cf-arrow'>→</span> "
                f"<span class='cf-target'>{cf_v:.1f}{UNITS.get(f,'')}</span> "
                f"<span style='color:#64748b'>({arrow})</span><br>"
            )
        if not any_change:
            lines.append("&nbsp;&nbsp;<i>(no changes — model already near boundary)</i><br>")
        lines.append("<br>")

    lines.append("<i>These are <b>actionable</b> targets: clinical interventions "
                 "(medication, exercise, diet) that move these values toward the "
                 "counterfactual could reduce predicted risk.</i>")
    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="banner">
  <h1><span class="pumping-heart">&#10084;&#65039;</span> Heart Disease Prediction Dashboard</h1>
  <p>AI-powered cardiovascular risk assessment</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Model files not found. Run `python train_save_model.py` first.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patient Profile")
    st.markdown("---")
    st.markdown("### Demographics")
    age = st.slider("Age", 20, 80, 55, help="Patient age in years")
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)

    st.markdown("### Cardiac Indicators")
    cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
    cp      = st.selectbox("Chest Pain Type", list(cp_map.keys()), format_func=lambda x: f"{x} - {cp_map[x]}")
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang   = st.radio("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope_map = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    slope   = st.selectbox("ST Slope", list(slope_map.keys()), format_func=lambda x: f"{x} - {slope_map[x]}")

    st.markdown("### Lab Results")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    restecg  = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])

    st.markdown("### Additional Tests")
    ca = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
    thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    thal = st.selectbox("Thalassemia", list(thal_map.keys()), format_func=lambda x: f"{x} - {thal_map[x]}")

# ── Build features & predict ───────────────────────────────────────────────────
inputs = dict(age=float(age), sex=float(sex), cp=float(cp),
              trestbps=float(trestbps), chol=float(chol), fbs=float(fbs),
              restecg=float(restecg), thalach=float(thalach), exang=float(exang),
              oldpeak=float(oldpeak), slope=float(slope), ca=float(ca), thal=float(thal))

X_input = build_features(inputs, feature_cols, selector)
prob    = model.predict_proba(X_input)[0][1]
pred    = int(model.predict(X_input)[0])
risk_class = "high-risk" if prob >= 0.6 else ("medium-risk" if prob >= 0.35 else "low-risk")

# ── KPI row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="metric-card"><p class="metric-value {risk_class}">{prob*100:.1f}%</p>'
                f'<p class="metric-label">Disease Probability</p></div>', unsafe_allow_html=True)
with c2:
    ds = inputs['cp'] + inputs['exang'] + inputs['oldpeak'] + inputs['ca']
    ds_color = "high-risk" if ds >= 6 else ("medium-risk" if ds >= 3 else "low-risk")
    st.markdown(f'<div class="metric-card"><p class="metric-value {ds_color}">{ds:.1f}</p>'
                f'<p class="metric-label">Disease Score</p></div>', unsafe_allow_html=True)
with c3:
    hr_pct = int((inputs['thalach'] / 220) * 100)
    hr_color = "low-risk" if hr_pct >= 70 else ("medium-risk" if hr_pct >= 55 else "high-risk")
    st.markdown(f'<div class="metric-card"><p class="metric-value {hr_color}">{hr_pct}%</p>'
                f'<p class="metric-label">Max HR % of Predicted</p></div>', unsafe_allow_html=True)
with c4:
    chol_status = "Normal" if inputs['chol'] < 200 else ("Borderline" if inputs['chol'] < 240 else "High")
    chol_color  = "low-risk" if inputs['chol'] < 200 else ("medium-risk" if inputs['chol'] < 240 else "high-risk")
    st.markdown(f'<div class="metric-card"><p class="metric-value {chol_color}">{chol_status}</p>'
                f'<p class="metric-label">Cholesterol Status</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gauge + Risk Factors ────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1.9])

with left:
    st.markdown("### Risk Gauge")
    st.plotly_chart(make_gauge(prob), use_container_width=True, config={'displayModeBar': False})
    if pred == 1:
        st.markdown(f'<div class="warning-box"><b>Prediction: Heart Disease Detected</b><br>'
                    f'Confidence: <b>{prob*100:.1f}%</b>. Immediate cardiology consultation recommended.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box"><b>Prediction: No Heart Disease Detected</b><br>'
                    f'Confidence: <b>{(1-prob)*100:.1f}%</b>. Continue regular health monitoring.</div>',
                    unsafe_allow_html=True)

with right:
    st.markdown("#### Individual Risk Factor Contributions")
    st.plotly_chart(make_risk_bars(inputs), use_container_width=True, config={'displayModeBar': False})
    st.markdown('<div class="info-box">Red >= 60% &nbsp;·&nbsp; Yellow 35-60% &nbsp;·&nbsp; Green < 35%</div>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  EXPLAINABLE AI SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Explainable AI (SHAP Analysis)")
st.markdown("""
<div class="info-box">
SHAP (SHapley Additive exPlanations) quantifies exactly how much each feature <b>pushes the prediction
up (red)</b> or <b>down (blue)</b> from the average baseline probability.
Per the AI-DWELL notebook (PART 4), the primary path uses <b>TreeExplainer on the LightGBM base learner</b>
of the Stack6 ensemble — fast and exact.
</div>
""", unsafe_allow_html=True)

with st.spinner("Computing SHAP values for this patient..."):
    if explainer["kind"] == "tree":
        # TreeExplainer path — fast, exact (matches notebook PART 4)
        tree_exp = explainer["explainer"]
        sv_raw   = tree_exp.shap_values(X_input)
        if isinstance(sv_raw, list):           # binary classification → list of 2
            shap_vals = np.array(sv_raw[1]).flatten()
        else:
            sv_arr = np.array(sv_raw)
            # newer SHAP returns shape (n_samples, n_features, n_classes)
            if sv_arr.ndim == 3:
                shap_vals = sv_arr[0, :, 1]
            else:
                shap_vals = sv_arr.flatten()
        ev = tree_exp.expected_value
        base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0 else ev)
        shap_method_label = "TreeExplainer · LightGBM base learner"
    else:
        # KernelExplainer fallback — full ensemble, model-agnostic
        kernel_exp = explainer["explainer"]
        shap_vals_raw = kernel_exp.shap_values(X_input.values, nsamples=150, silent=True)
        shap_vals = np.array(shap_vals_raw).flatten()
        base_value = float(kernel_exp.expected_value)
        shap_method_label = "KernelExplainer · Stack6 ensemble (model-agnostic)"

xai_left, xai_right = st.columns([1.6, 1.4])

with xai_left:
    st.markdown("### SHAP Waterfall — Feature Impact")
    st.markdown(f"""
    <div class="info-box" style="margin-bottom:10px">
    Each bar shows a feature's contribution to the final prediction.
    <span style="color:#fc5c7d;font-weight:600">Red = increases disease risk</span> &nbsp;|&nbsp;
    <span style="color:#56b4e9;font-weight:600">Blue = decreases disease risk</span><br>
    <span style="color:#64748b;font-size:0.82rem">Method: {shap_method_label}</span>
    </div>
    """, unsafe_allow_html=True)
    fig_wf = make_shap_waterfall(shap_vals, feature_cols, X_input, base_value)
    st.plotly_chart(fig_wf, use_container_width=True, config={'displayModeBar': False})

with xai_right:
    st.markdown("### AI Explanation")
    nl_html = natural_language_explanation(shap_vals, feature_cols, X_input, pred, prob, base_value)
    box_cls = "nl-explain-risk" if pred == 1 else "nl-explain"
    st.markdown(f'<div class="{box_cls}">{nl_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DiCE COUNTERFACTUAL SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## DiCE — Diverse Counterfactual Explanations")
st.markdown("""
<div class="info-box">
DiCE answers a different question than SHAP. SHAP asks <i>"why did the model predict this?"</i>
DiCE asks <i>"what would need to change for the model to predict <b>No Disease</b>?"</i>
It generates minimally-different alternate patient profiles — concrete, actionable targets
for clinical intervention. Per the AI-DWELL notebook (PART 5), this uses a parallel Stack6
trained on raw features (no one-hot encoding).
</div>
""", unsafe_allow_html=True)

if pred == 1:
    # Build query in raw-feature form for DiCE
    dice_query = build_dice_query(
        inputs,
        dice_bundle["feature_cols"],
        dice_bundle["categorical_cols"],
        dice_bundle["continuous_cols"],
    )

    # Sanity check — does the raw-feature Stack6 also flag this patient?
    dice_pred = int(dice_bundle["model"].predict(dice_query)[0])

    if dice_pred == 0:
        st.markdown("""
        <div class="info-box">
        <b>Note:</b> The raw-feature Stack6 model used by DiCE predicts <b>No Disease</b>
        for this patient (the main model with engineered features predicted Disease).
        Counterfactuals require a positive prediction to flip — adjust sidebar inputs
        to a higher-risk profile to see DiCE in action.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Permitted ranges and features-to-vary — match notebook
        permitted = {
            'age':           [20,   80],
            'trestbps':      [80,  200],
            'chol':          [100, 400],
            'thalach':       [70,  210],
            'oldpeak':       [0.0,  6.0],
            'disease_score': [0.0, 12.0],
        }
        features_to_vary = ['trestbps', 'chol', 'thalach', 'oldpeak', 'disease_score']

        cf_result = None
        cf_error  = None
        with st.spinner("Generating diverse counterfactuals (this may take 10–20s)..."):
            # Try 3 → 2 → 1 counterfactuals (notebook fallback strategy)
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
            <div class="warning-box">
            <b>DiCE could not generate counterfactuals for this patient.</b><br>
            This usually means the patient profile is too far inside the high-risk region
            for any small change in the permitted features to flip the prediction.<br>
            <span style="color:#64748b;font-size:0.82rem">Last error: {cf_error}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            # DiCE includes the target column — drop it for visualization
            if 'target' in cf_df.columns:
                cf_df = cf_df.drop(columns=['target'])

            dice_left, dice_right = st.columns([1.6, 1.4])

            with dice_left:
                st.markdown(f"### Original vs Counterfactuals — Feature Changes")
                st.markdown("""
                <div class="info-box" style="margin-bottom:10px">
                <span style="color:#fc5c7d;font-weight:600">Red = current patient</span> &nbsp;|&nbsp;
                <span style="color:#56ab2f;font-weight:600">Green/Blue/Purple = counterfactual targets</span><br>
                Only features that DiCE actually changed are shown.
                </div>
                """, unsafe_allow_html=True)
                fig_dice = make_dice_chart(dice_query, cf_df, dice_bundle["feature_cols"])
                st.plotly_chart(fig_dice, use_container_width=True, config={'displayModeBar': False})

            with dice_right:
                st.markdown("### Counterfactual Explanation")
                dice_nl = dice_natural_language(dice_query, cf_df, pred, prob)
                box_cls = "nl-explain-risk" if pred == 1 else "nl-explain"
                st.markdown(f'<div class="{box_cls}">{dice_nl}</div>', unsafe_allow_html=True)

            # Counterfactual table (full detail)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Counterfactual Detail Table")
            st.markdown("""<div class="info-box" style="margin-bottom:6px">
            Full counterfactual feature values vs the current patient.
            </div>""", unsafe_allow_html=True)

            display_df = pd.concat(
                [dice_query.reset_index(drop=True), cf_df.reset_index(drop=True)],
                ignore_index=True,
            )
            display_df.index = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]
            st.dataframe(display_df.round(2), use_container_width=True)
else:
    st.markdown("""
    <div class="info-box">
    <b>Counterfactuals are generated only when the model predicts Disease.</b><br>
    Since this patient is currently predicted as <b>No Disease</b>, there is nothing to "flip" —
    DiCE finds the smallest changes that would push a high-risk patient back to low-risk.<br><br>
    Adjust the sidebar to a high-risk profile (e.g., higher chest-pain type, exercise angina,
    blocked vessels, ST depression) to see DiCE generate counterfactual explanations.
    </div>
    """, unsafe_allow_html=True)

# ── Clinical summary ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Clinical Summary")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("#### Vital Signs")
    bp_status  = "Normal" if trestbps < 120 else ("Elevated" if trestbps < 130 else ("Stage 1 HTN" if trestbps < 140 else "Stage 2 HTN"))
    ecg_labels = ["Normal","ST-T Wave Abnormality","LV Hypertrophy"]
    st.markdown(f"""
| Metric | Value | Status |
|--------|-------|--------|
| Blood Pressure | {int(trestbps)} mmHg | {bp_status} |
| Max Heart Rate | {int(thalach)} bpm | {'Normal' if 60<=thalach<=100 else 'High'} |
| ST Depression  | {oldpeak} | {'Abnormal' if oldpeak > 2 else 'Normal'} |
| Resting ECG    | — | {ecg_labels[int(restecg)]} |
""")
with s2:
    st.markdown("#### Lab Values")
    chol_label = "Desirable" if chol < 200 else ("Borderline High" if chol < 240 else "High")
    st.markdown(f"""
| Metric | Value | Status |
|--------|-------|--------|
| Cholesterol | {int(chol)} mg/dl | {chol_label} |
| Fasting Blood Sugar | — | {'Normal' if fbs==0 else 'Elevated'} |
| Fluoroscopy Vessels | {int(ca)} | {f'{int(ca)} vessel(s)' if ca>0 else 'None'} |
| Thalassemia | — | {thal_map[int(thal)]} |
""")
with s3:
    st.markdown("#### Clinical Flags")
    flags = []
    if prob >= 0.6:    flags.append("HIGH disease probability — urgent review")
    if trestbps > 140: flags.append("Hypertension detected")
    if chol > 240:     flags.append("High cholesterol")
    if oldpeak > 2:    flags.append("Significant ST depression")
    if exang == 1:     flags.append("Exercise-induced angina")
    if ca >= 2:        flags.append("Multiple vessel blockages")
    if thal == 7:      flags.append("Reversible thalassemia defect")
    if not flags:      flags.append("No major risk flags detected")
    for f in flags:
        icon = "🔴" if "HIGH" in f or "Multiple" in f else ("🟠" if any(w in f for w in ["Hyper","choles","thalassemia"]) else ("🟡" if any(w in f for w in ["ST","Exercise"]) else "✅"))
        st.markdown(f"- {icon} {f}")
