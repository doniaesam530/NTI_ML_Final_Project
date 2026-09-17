"""
HealthCare Treatment Outcome Predictor - Streamlit App
Recreates the preprocessing, EDA, and modeling pipeline from HealthCare.ipynb
as an interactive GUI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Pulse | Treatment Outcome AI", layout="wide", page_icon="🩺")

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #12263a 0%, #0b1620 55%, #070d13 100%);
        color: #e8f0fa;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, li, .stMarkdown {
        color: #e8f0fa !important;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #123a5e 0%, #17706f 100%);
        color: #ffffff;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1, .hero p {
        color: #ffffff !important;
    }
    .hero h1 {
        margin-bottom: 0.2rem;
        font-size: 2.1rem;
    }
    .hero p {
        opacity: 0.92;
        font-size: 1.02rem;
        margin: 0;
    }
    .pill {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        font-size: 0.8rem;
        margin-right: 0.4rem;
        color: #ffffff !important;
    }
    div[data-testid="stMetric"] {
        background: #101d2b;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        border: 1px solid #1e3348;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #e8f0fa !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #101d2b;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        border: 1px solid #1e3348;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #cfe0f2 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #123a5e, #17706f);
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
    }
    .stButton>button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(120deg, #17a2b8, #17706f);
        color: white !important;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(23, 162, 184, 0.35);
    }
    section[data-testid="stSidebar"] {
        background: #0a141e;
        border-right: 1px solid #1e3348;
    }
    section[data-testid="stSidebar"] * {
        color: #eaf2fb !important;
    }
    /* Inputs, selects, dataframes, expanders */
    div[data-testid="stFileUploader"],
    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        background-color: #101d2b !important;
        color: #e8f0fa !important;
        border: 1px solid #1e3348 !important;
    }
    div[data-testid="stDataFrame"] {
        background: #101d2b;
        border-radius: 10px;
        border: 1px solid #1e3348;
    }
    div[data-testid="stExpander"] {
        background: #101d2b;
        border: 1px solid #1e3348;
        border-radius: 10px;
    }
    .stAlert {
        background: #101d2b;
        border: 1px solid #1e3348;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Make matplotlib charts match the dark theme
plt.rcParams.update({
    "figure.facecolor": "#0b1620",
    "axes.facecolor": "#101d2b",
    "savefig.facecolor": "#0b1620",
    "axes.edgecolor": "#e8f0fa",
    "axes.labelcolor": "#e8f0fa",
    "text.color": "#e8f0fa",
    "xtick.color": "#e8f0fa",
    "ytick.color": "#e8f0fa",
    "grid.color": "#1e3348",
})

ORDINAL_MAP = {'Low': 0, 'Medium': 1, 'High': 2}

# ---------------------------------------------------------------------------
# Data loading & cleaning (mirrors the notebook)
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)

    for col in ['blood_pressure', 'cholesterol', 'bmi']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if 'alcohol_intake' in df.columns:
        df['alcohol_intake'] = df['alcohol_intake'].fillna('None')

    if 'medication' in df.columns:
        df['medication'] = df['medication'].fillna(df['medication'].mode()[0])

    if 'treatment_start' in df.columns and 'treatment_end' in df.columns:
        df['treatment_end'] = pd.to_datetime(df['treatment_end'])
        df['treatment_start'] = pd.to_datetime(df['treatment_start'])
        df['treatment_start'] = df['treatment_start'].fillna(
            df['treatment_end'] - pd.to_timedelta(df['hospital_stay_days'], unit='D')
        )
        df['treatment_duration_days'] = (df['treatment_end'] - df['treatment_start']).dt.days
        df = df.drop(columns=['treatment_start', 'treatment_end'])

    return df


@st.cache_data
def build_features(df):
    """Reproduces the encoding block from the notebook and returns X, y."""
    data = df.copy()
    y = data['treatment_outcome']
    X = data.drop(columns=['treatment_outcome'])

    X = X.drop(columns=['patient_name', 'diagnosis'], errors='ignore')

    if 'gender' in X.columns:
        X['gender'] = X['gender'].astype(str).str.strip().str.title()
        X['gender'] = X['gender'].map({'Male': 0, 'Female': 1})

    for col in ['infection_risk', 'adherence_level']:
        if col in X.columns:
            X[col] = X[col].map(ORDINAL_MAP)

    if 'alcohol_intake' in X.columns:
        X['alcohol_intake'] = X['alcohol_intake'].replace('None', np.nan).map(ORDINAL_MAP)

    if 'medication' in X.columns:
        X = pd.get_dummies(X, columns=['medication'], drop_first=True)

    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test, use_xgb):
    results = {}
    preds = {}
    fitted = {}

    # Logistic Regression
    log_reg = LogisticRegression(
        class_weight='balanced',
        max_iter=1000, random_state=42
    )
    log_reg.fit(X_train, y_train)
    preds['Logistic Regression'] = log_reg.predict(X_test)
    fitted['Logistic Regression'] = log_reg

    # KNN (scaled)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_scaled, y_train)
    preds['KNN'] = knn.predict(X_test_scaled)
    fitted['KNN'] = (knn, scaler)

    # XGBoost
    if use_xgb:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_enc)
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=len(le.classes_),
            eval_metric='mlogloss', max_depth=6, learning_rate=0.1,
            n_estimators=300, random_state=42
        )
        xgb_model.fit(X_train, y_train_enc, sample_weight=sample_weights)
        y_pred_xgb_enc = xgb_model.predict(X_test)
        preds['XGBoost'] = le.inverse_transform(y_pred_xgb_enc)
        fitted['XGBoost'] = (xgb_model, le)

    # SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    preds['SVM'] = svm_model.predict(X_test)
    fitted['SVM'] = svm_model

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    preds['Random Forest'] = rf_model.predict(X_test)
    fitted['Random Forest'] = rf_model

    # Voting Classifier (LR + SVM + RF, soft voting)
    log_reg_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced',
                                      max_iter=1000, random_state=42))
    ])
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])
    voting_model = VotingClassifier(
        estimators=[('lr', log_reg_pipe), ('svm', svm_pipe), ('rf', rf_model)],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    preds['Voting Classifier'] = voting_model.predict(X_test)
    fitted['Voting Classifier'] = voting_model

    # Metrics table
    rows = []
    for name, p in preds.items():
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, p),
            'Macro F1': f1_score(y_test, p, average='macro'),
            'Macro Precision': precision_score(y_test, p, average='macro'),
            'Macro Recall': recall_score(y_test, p, average='macro'),
        })
    comparison_df = pd.DataFrame(rows).set_index('Model').round(3)

    return fitted, preds, comparison_df


def predict_single(model_name, fitted, row_df):
    obj = fitted[model_name]
    if model_name == 'KNN':
        knn, scaler = obj
        return knn.predict(scaler.transform(row_df))[0]
    if model_name == 'XGBoost':
        xgb_model, le = obj
        pred_enc = xgb_model.predict(row_df)
        return le.inverse_transform(pred_enc)[0]
    return obj.predict(row_df)[0]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <span class="pill">🩺 Clinical ML Studio</span>
        <span class="pill">Multi-model benchmarking</span>
        <span class="pill">Live inference</span>
        <h1>Pulse — Treatment Outcome Predictor</h1>
        <p>Upload patient records, explore the data, race six machine-learning models
        against each other, and get an instant outcome prediction for any new case.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded = st.file_uploader("Upload health_care.csv", type=["csv"])
    st.markdown("---")
    st.caption("The file must contain a `treatment_outcome` column for training to work.")
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Upload your dataset\n"
        "2. Explore it in *Insights*\n"
        "3. Train & compare models\n"
        "4. Predict a new case instantly"
    )

if uploaded is None:
    st.info("👈 Upload your patient dataset from the sidebar to get started.")
    st.stop()

df = load_and_clean(uploaded)

if 'treatment_outcome' not in df.columns:
    st.error("The file must contain a `treatment_outcome` column.")
    st.stop()

tab_overview, tab_eda, tab_models, tab_predict = st.tabs(
    ["📋 Overview", "📊 Insights & EDA", "🤖 Model Arena", "🔮 Predict a Case"]
)

# --- Overview tab -----------------------------------------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("🧾 Rows", df.shape[0])
    c2.metric("📐 Columns", df.shape[1])
    c3.metric("🕳️ Missing Values", int(df.isnull().sum().sum()))

    st.subheader("👀 A Peek at the Data")
    st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Missing Values per Column")
        st.dataframe(df.isnull().sum().rename("Missing Count").to_frame(), use_container_width=True)
    with col_b:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# --- EDA tab -----------------------------------------------------------------
with tab_eda:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        if 'age' in df.columns:
            st.subheader("🎂 Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['age'], kde=True, ax=ax, color="#0f4c81")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        if 'gender' in df.columns:
            st.subheader("🧍 Gender Split")
            fig, ax = plt.subplots()
            sns.countplot(x='gender', data=df, ax=ax, palette=["#0f4c81", "#2d9596"])
            st.pyplot(fig)

        if 'diagnosis' in df.columns:
            st.subheader("🩻 Diagnosis Breakdown")
            fig, ax = plt.subplots()
            df['diagnosis'].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

    with col2:
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(7, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        if 'bmi' in df.columns and 'blood_sugar' in df.columns:
            st.subheader("⚖️ BMI vs. Blood Sugar")
            fig, ax = plt.subplots()
            ax.scatter(df['bmi'], df['blood_sugar'], color="#2d9596", alpha=0.7)
            ax.set_xlabel("BMI")
            ax.set_ylabel("Blood Sugar")
            st.pyplot(fig)

        if 'treatment_outcome' in df.columns and 'medication' in df.columns:
            st.subheader("💊 Outcome by Medication")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.countplot(data=df, x='treatment_outcome', hue='medication', ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    st.subheader("📦 Outlier Check (Boxplots)")
    show_box = st.checkbox("Show boxplots (may take a moment with many columns)")
    if show_box and numeric_cols:
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8, len(numeric_cols) * 2.2))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

# --- Models tab --------------------------------------------------------------
with tab_models:
    st.subheader("⚙️ Training Setup")
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    use_xgb = st.checkbox("Include XGBoost in the race", value=XGB_AVAILABLE, disabled=not XGB_AVAILABLE)
    if not XGB_AVAILABLE:
        st.caption("⚠️ xgboost isn't installed in this environment, so it will be skipped.")

    if st.button("🚀 Train All Models", type="primary"):
        X, y = build_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        fitted, preds, comparison_df = train_models(X_train, y_train, X_test, y_test, use_xgb)

        st.session_state['fitted'] = fitted
        st.session_state['comparison_df'] = comparison_df
        st.session_state['feature_columns'] = X.columns.tolist()
        st.session_state['X_train_columns_dtypes'] = X.dtypes
        st.session_state['y_test'] = y_test
        st.session_state['preds'] = preds
        st.session_state['raw_columns'] = df.drop(columns=['treatment_outcome']).columns.tolist()
        st.success("✅ All models trained successfully!")

    if 'comparison_df' in st.session_state:
        st.subheader("🏆 Model Leaderboard")
        st.dataframe(st.session_state['comparison_df'], use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        st.session_state['comparison_df'].plot(kind='bar', ax=ax, colormap="viridis")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        best_model = st.session_state['comparison_df']['Macro F1'].idxmax()
        st.info(f"🥇 Best model by Macro F1: **{best_model}**")

        model_choice = st.selectbox(
            "Inspect a model's confusion matrix",
            st.session_state['comparison_df'].index.tolist()
        )
        y_test = st.session_state['y_test']
        p = st.session_state['preds'][model_choice]
        labels_order = sorted(y_test.unique())
        cm = confusion_matrix(y_test, p, labels=labels_order)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_order, yticklabels=labels_order, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_choice} — Confusion Matrix')
        st.pyplot(fig)

        with st.expander("📄 Full Classification Report"):
            st.text(classification_report(y_test, p))

# --- Prediction tab -----------------------------------------------------------
with tab_predict:
    if 'fitted' not in st.session_state:
        st.warning("⚠️ Train the models first in the **Model Arena** tab.")
    else:
        st.subheader("🧬 Enter a New Patient Case")

        raw_cols = st.session_state['raw_columns']
        input_data = {}

        cols = st.columns(3)
        for i, col in enumerate(raw_cols):
            target = cols[i % 3]
            if col == 'patient_name':
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                default_val = float(series.median())
                input_data[col] = target.number_input(col, value=default_val)
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                input_data[col] = target.selectbox(col, options)

        model_for_pred = st.selectbox(
            "Choose a model", list(st.session_state['fitted'].keys())
        )

        if st.button("🔮 Predict Outcome"):
            raw_row = pd.DataFrame([input_data])
            # reuse the same cleaning/encoding path
            combined = pd.concat([df.drop(columns=['treatment_outcome']), raw_row], ignore_index=True)
            combined['treatment_outcome'] = list(df['treatment_outcome']) + [df['treatment_outcome'].mode()[0]]
            X_all, _ = build_features(combined)
            row_encoded = X_all.iloc[[-1]]

            # align columns with training features
            feature_columns = st.session_state['feature_columns']
            for c in feature_columns:
                if c not in row_encoded.columns:
                    row_encoded[c] = 0
            row_encoded = row_encoded[feature_columns]

            result = predict_single(model_for_pred, st.session_state['fitted'], row_encoded)
            st.success(f"🎯 Predicted treatment outcome: **{result}**")
