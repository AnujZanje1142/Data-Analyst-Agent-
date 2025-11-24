import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, accuracy_score
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO
from datetime import datetime
import pdfkit
import os
from transformers import pipeline  # For local fallback; pip install transformers torch
import time
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
# ==== 100% FREE LLM – NO API KEY NEEDED (2025 WORKING) ====
# 100% FREE LLM – NO KEY, NO LIMIT, WORKS FOREVER (2025)
# 1. Hugging Face Inference API (free tier, some models support embeddings)
from sentence_transformers import SentenceTransformer, util

# Load model locally (runs offline)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences
sentences = ["This is a test sentence.", "This is another sentence."]

# Encode sentences to embeddings (run locally)
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine similarity locally
cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(f"Similarity score (local): {cosine_scores.item()}")

# Online alternative: Hugging Face Inference API for the same model (runs online)
import requests

def get_online_embedding(text):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    response = requests.post(API_URL, json={"inputs": text})
    return response.json()

online_embedding = get_online_embedding("This is a test sentence.")
print("Embedding from Hugging Face API (online):", online_embedding)

import requests

def get_huggingface_embeddings(text):
    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
    headers = {"Authorization": "Bearer hf_YOUR_ACCESS_TOKEN"}  # Optional, some public queries possible without token
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

embeddings = get_huggingface_embeddings("Example sentence for embedding.")

# 2. OpenTextEmbeddings - Open source API for embedding models (no keys needed)
# https://github.com/Trelent/opentextembeddings
# You can self-host or use their free hosted API endpoint if available.

# 3. Nomic Atlas API for vector search over your own data (free tier, API docs: https://docs.nomic.ai)
# https://api.nomic.ai
# Example: Post text query and get nearest neighbors from dataset

# 4. Google Gemini API for embeddings (free tier available, requires signup)
# https://ai.google.dev/gemini-api/docs/embeddings

# 5. Cohere Embeddings API https://cohere.ai (free tier with signup)
# You get high-quality embeddings for semantic search.

# 6. AssemblyAI (free credits for audio transcription, useful if your data involves audio)

# Open-source libraries you can deploy yourself:
# - Sentence Transformers (Python library)
# - FAISS for vector similarity search
# - Annoy for approximate nearest neighbors search

# Example: Hugging Face inference with no-signup (limited usage)
response = requests.post(
    "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2",
    json={"inputs": "Your text here"}
)
embeddings = response.json()


# YE PURANA FUNCTION HATA DO
def generate_ai_response(prompt, context=""):
    url = "https://api.oyyi.xyz/v1/chat/completions"
    payload = {
        "model": "llama-3.1-70b",
        "messages": [
            {"role": "system", "content": "You are a world-class data analyst. Answer clearly and suggest charts."},
            {"role": "user", "content": f"Data overview:\n{context}\n\nQuestion: {prompt}"}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }
    try:
        r = requests.post(url, json=payload, timeout=25)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return "AI is busy right now, try again in 5 seconds."
    except:
        return "No internet or server down. AI will be back soon."

# YE NAYA FUNCTION DAAL DO (Copy-Paste Kar Do)


  #MAIN START
st.set_page_config(
    page_title="DataForge AI - Advanced ML Data Analysis Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.cleaned_data = None
    st.session_state.view_mode = 'landing'
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'data_transform' not in st.session_state:
    st.session_state.data_transform = {
        'impute_strategy': 'mean',
        'scaling': False,
        'outlier_removal': False
    }

# Dark Theme CSS - Professional, clean dark mode
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

        .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #0f0f0f;
            color: #e0e0e0;
        }

        /* Dark Theme Variables */
        :root {
            --primary-color: #3b82f6; /* Blue for actions */
            --bg-primary: #0f0f0f; /* Dark bg */
            --bg-secondary: #1a1a1a; /* Card bg */
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --card-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            --border-radius: 8px;
            --transition: all 0.2s ease;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg-primary);
            padding: 1rem;
        }

        /* Hero: Dark Header */
        .hero-section {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: var(--card-shadow);
            border: 1px solid #333;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
        }

        /* Feature Cards: Dark Boxes */
        .feature-card {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            text-align: left;
            border: 1px solid #333;
            transition: var(--transition);
            color: var(--text-primary);
        }

        .feature-card:hover {
            box-shadow: 0 4px 8px rgba(59, 130, 246, 0.2);
            border-color: var(--primary-color);
        }

        .feature-icon {
            font-size: 2rem;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .feature-desc {
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* Upload Section: Dark */
        .upload-section {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            text-align: center;
            margin: 2rem 0;
            border: 1px solid #333;
            color: var(--text-primary);
        }

        .upload-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
        }

        .upload-desc {
            color: var(--text-secondary);
            margin-bottom: 1.5rem;
        }

        .upload-area {
            border: 2px dashed #555;
            border-radius: 8px;
            padding: 2rem;
            margin: 1rem 0;
            transition: var(--transition);
            background: #2a2a2a;
            color: var(--text-secondary);
        }

        .upload-area:hover {
            border-color: var(--primary-color);
            background: #1e40af;
        }

        /* Buttons: Dark Blue */
        .cta-button, .stButton > button {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
            color: white;
        }

        .cta-button:hover, .stButton > button:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
        }

        /* Success Box: Dark Green */
        .success-box {
            background: #1f4d30;
            color: #86efac;
            padding: 1rem;
            border-radius: var(--border-radius);
            text-align: center;
            margin: 1.5rem 0;
            border-left: 4px solid #22c55e;
        }

        /* Dashboard Metrics: Dark Cards */
        .dashboard-metric, .metric-card {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            padding: 1rem;
            box-shadow: var(--card-shadow);
            text-align: center;
            border: 1px solid #333;
            transition: var(--transition);
            color: var(--text-primary);
        }

        .dashboard-metric:hover, .metric-card:hover {
            box-shadow: 0 4px 8px rgba(59, 130, 246, 0.2);
        }

        /* Tabs: Dark */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-secondary);
            border: 1px solid #333;
            border-radius: var(--border-radius);
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--primary-color);
            color: white;
        }

        /* DataFrame: Dark Table */
        .stDataFrame {
            background: var(--bg-secondary);
            border-radius: var(--border-radius);
            color: var(--text-primary);
        }

        .dataframe thead th {
            background: #2a2a2a;
            color: var(--text-primary);
            border: 1px solid #333;
        }

        .dataframe tbody td {
            border: 1px solid #333;
            color: var(--text-primary);
        }

        /* Inputs: Dark */
        .stTextInput > div > div > input, .stSelectbox > div > div > select {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid #555;
            border-radius: 6px;
        }

        /* Footer: Dark */
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: var(--text-secondary);
            border-top: 1px solid #333;
            margin-top: 2rem;
        }

        /* ML Section Boxes */
        .ml-box {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            border: 1px solid #333;
            color: var(--text-primary);
        }
    </style>
""", unsafe_allow_html=True)

# Enhanced functions with ML
def analyze_advanced(df):
    analysis = {}
    analysis['stats'] = df.describe(percentiles=[.05, .25, .5, .75, .95])
    analysis['missing_pct'] = (df.isna().sum() / len(df)) * 100
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        analysis['correlation'] = numeric_df.corr()
    weak_points = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            weak_points.append(f"{col}: {df[col].isna().sum()} missing ({analysis['missing_pct'][col]:.1f}%)")
        if df[col].dtype == 'object' and df[col].nunique() == len(df[col]):
            weak_points.append(f"{col}: High cardinality")
    strong_points = []
    if not numeric_df.empty:
        strong_points.append(f"{len(numeric_df.columns)} numeric columns")
    if len(df.columns[df.isna().sum() == 0]) > 0:
        strong_points.append(f"{len(df.columns[df.isna().sum() == 0])} complete columns")
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if date_cols:
        strong_points.append(f"Time series in: {', '.join(date_cols)}")
    analysis['weak_points'] = weak_points
    analysis['strong_points'] = strong_points
    analysis['date_cols'] = date_cols
    return analysis

def detect_anomalies(df, threshold=3):
    numeric_df = df.select_dtypes(include=np.number)
    anomalies = pd.DataFrame()
    for col in numeric_df.columns:
        z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
        anomalies[col] = z_scores > threshold
    return anomalies

def time_series_analysis(df, date_col, value_col):
    df = df.sort_values(date_col).set_index(date_col)
    result = seasonal_decompose(df[value_col], model='additive', period=1)
    return result

def predict_trend(df, x_col, y_col, model_type='linear'):
    X = df[[x_col]].values
    y = df[y_col].values
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    r2 = r2_score(y, pred)
    return pred, r2, model

def cluster_data(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    return clusters, kmeans, scaler

def pca_analysis(df, n_components=2):
    numeric_df = df.select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled)
    return pca_result, pca, scaler

def classify_data(df, target_col):
    X = df.select_dtypes(include=np.number).drop(columns=[target_col]) if target_col in df.select_dtypes(include=np.number).columns else df.select_dtypes(include=np.number)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    feature_importances = model.feature_importances_
    return model, acc, cv_scores.mean(), feature_importances

def create_report(df, analysis):
    report_content = f"""
    <h1>Data Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <h2>Overview</h2>
    <p>Shape: {df.shape[0]} rows × {df.shape[1]} columns</p>
    <h2>Findings</h2>
    <h3>Strengths</h3>
    <ul>
    """
    for point in analysis['strong_points']:
        report_content += f"<li>{point}</li>"
    report_content += """
    </ul>
    <h3>Issues</h3>
    <ul>
    """
    for point in analysis['weak_points']:
        report_content += f"<li>{point}</li>"
    report_content += """
    </ul>
    <h2>Stats</h2>
    """
    try:
        wkhtml_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=wkhtml_path)
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }
        pdf = pdfkit.from_string(report_content, False, options=options, configuration=config)
        return BytesIO(pdf)
    except Exception as e:
        st.error(f"PDF failed: {str(e)}")
        fallback_report = f"Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nShape: {df.shape[0]} x {df.shape[1]}\nStrengths:\n"
        for point in analysis['strong_points']:
            fallback_report += f"- {point}\n"
        fallback_report += "\nIssues:\n"
        for point in analysis['weak_points']:
            fallback_report += f"- {point}\n"
        fallback_report += "\nStats:\n" + analysis['stats'].to_string()
        return BytesIO(fallback_report.encode('utf-8'))

# Landing Page: Dark Tool-like
def render_landing_page(uploaded_file):
    # Hero
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">DataForge AI</h1>
            <p class="hero-subtitle">Advanced ML-Powered Data Analysis Tool</p>
        </div>
    """, unsafe_allow_html=True)

    # Features: Dark Boxes with ML Emphasis
    st.markdown("# Core Tools")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon"><i class="fas fa-upload"></i></div>
                <h3 class="feature-title">Data Upload</h3>
                <p class="feature-desc">Load CSV, Excel, JSON, Parquet. Auto-preview and clean.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon"><i class="fas fa-robot"></i></div>
                <h3 class="feature-title">ML Models</h3>
                <p class="feature-desc">Clustering (KMeans), PCA, RF Regression/Classification, CV.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon"><i class="fas fa-chart-bar"></i></div>
                <h3 class="feature-title">Analyst Tools</h3>
                <p class="feature-desc">TS Decomp, Anomalies, Viz, Exports (PDF/CSV).</p>
            </div>
        """, unsafe_allow_html=True)

    # Upload
    st.markdown("# Start Analysis")
    with st.container():
        st.markdown("""
            <div class="upload-section">
                <h2 class="upload-title">Upload Dataset</h2>
                <p class="upload-desc">Begin with your data file.</p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select file",
            type=["csv", "xlsx", "json", "parquet"]
        )

        if uploaded_file is not None:
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            st.markdown('<div class="success-box">Data loaded successfully.</div>', unsafe_allow_html=True)

            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)

                st.info(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head())

                st.session_state.data = df
                st.session_state.cleaned_data = df
                st.session_state.view_mode = 'dashboard'
                st.rerun()
            except Exception as e:
                st.error(f"Load error: {e}")

        if st.button("Launch Dashboard"):
            if st.session_state.data:
                st.session_state.view_mode = 'dashboard'
                st.rerun()
            else:
                st.warning("Upload data first.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2025 DataForge AI | Powered by Streamlit, Scikit-learn </p>
        </div>
    """, unsafe_allow_html=True)

# Dashboard: With ML Tabs
def render_dashboard():
    st.title("🔧 Advanced Analysis Dashboard")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Reload Data", type=['csv', 'xlsx', 'json', 'parquet'])

        st.header("Preprocessing")
        st.session_state.data_transform['impute_strategy'] = st.selectbox("Impute", ['mean', 'median', 'drop'])
        st.session_state.data_transform['scaling'] = st.checkbox("Scale Features")
        st.session_state.data_transform['outlier_removal'] = st.checkbox("Remove Outliers")

        ai_enabled = st.toggle("AI Assistant", value=True)

        if st.button("← Home"):
            st.session_state.view_mode = 'landing'
            st.session_state.data = None
            st.rerun()

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            st.session_state.data = df
            st.session_state.cleaned_data = preprocess_data(
                strategy=st.session_state.data_transform['impute_strategy'],
                scale=st.session_state.data_transform['scaling'],
                remove_outliers=st.session_state.data_transform['outlier_removal']
            )
            st.session_state.analysis_done = False
            st.session_state.ml_model = None
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.data is not None:
        raw_df = st.session_state.data
        df = st.session_state.cleaned_data

        if not st.session_state.analysis_done:
            st.session_state.analysis = analyze_advanced(df)
            st.session_state.analysis_done = True

        analysis = st.session_state.analysis

        st.info(f"Loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")

        # Metrics Dark Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.metric("Rows", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.metric("Columns", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.metric("Missing Values", df.isna().sum().sum())
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.metric("Numeric Cols", len(df.select_dtypes(include=np.number).columns))
            st.markdown('</div>', unsafe_allow_html=True)

        # Tabs with ML
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Preview", "Basic Analysis", "ML Models", "Advanced Tools", "AI & Export"])

        with tab1:
            st.markdown("### Data Cleaning & Magic")

            # Cool animated toggle
            st.markdown("<h4 style='color:#3b82f6;'>✨ Smart Preprocessing</h4>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                missing_action = st.selectbox(
                    "Missing Values",
                    ["Smart Fill (Auto)", "Fill with Mean", "Fill with Median", "Drop Rows", "Drop Column"],
                    help="Smart Fill = best guess per column"
                )
            with col2:
                outlier_action = st.selectbox(
                    "Outliers",
                    ["Keep All", "Remove Extreme (Z>3)", "Cap at 99th Percentile"],
                    help="Clean noisy data automatically"
                )

            col3, col4 = st.columns(2)
            with col3:
                encode_cats = st.checkbox("Auto Encode Categories", value=True,
                                          help="Convert text columns to numbers for ML")
            with col4:
                scale_nums = st.checkbox("Normalize Numbers", value=False,
                                         help="Scale all numeric columns")

            # Cool one-click magic button
            if st.button("Apply Magic Cleaning ✨", type="primary", use_container_width=True):
                with st.spinner("Applying AI-powered cleaning..."):
                    df_clean = st.session_state.data.copy()

                    # 1. Smart Missing Values
                    for col in df_clean.columns:
                        if df_clean[col].isnull().sum() > 0:
                            if "drop column" in missing_action.lower():
                                df_clean = df_clean.drop(columns=[col])
                                st.success(f"Dropped column {col}")
                            elif "drop rows" in missing_action.lower():
                                df_clean = df_clean.dropna(subset=[col])
                            elif "mean" in missing_action.lower():
                                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                            elif "median" in missing_action.lower():
                                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                            else:  # Smart Fill
                                if df_clean[col].dtype == 'object':
                                    df_clean[col] = df_clean[col].fillna(
                                        df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
                                else:
                                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

                    # 2. Outlier Handling
                    if "remove extreme" in outlier_action.lower():
                        num_cols = df_clean.select_dtypes(include=np.number).columns
                        for col in num_cols:
                            z = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                            df_clean = df_clean[z < 3]
                    elif "cap" in outlier_action.lower():
                        num_cols = df_clean.select_dtypes(include=np.number).columns
                        for col in num_cols:
                            upper = df_clean[col].quantile(0.99)
                            df_clean[col] = df_clean[col].clip(upper=upper)

                    # 3. Encode Categories
                    if encode_cats:
                        cat_cols = df_clean.select_dtypes(include='object').columns
                        for col in cat_cols:
                            if df_clean[col].nunique() < 20:  # Only low-cardinality
                                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                                df_clean = pd.concat([df_clean.drop(col, axis=1), dummies], axis=1)
                            else:
                                # Label encoding for high cardinality
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                df_clean[col] = le.fit_transform(df_clean[col].astype(str))

                    # 4. Scale Numbers
                    if scale_nums:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        num_cols = df_clean.select_dtypes(include=np.number).columns
                        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

                    # Save cleaned data
                    st.session_state.cleaned_data = df_clean
                    st.session_state.analysis_done = False

                st.success("Data cleaned with AI magic! Ready for analysis")
                st.balloons()
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Data Types")
            st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']))

        with tab2:

            st.subheader("Strengths & Issues")
            # LIGHTNING-FAST AUTO CHARTS (2-second load max)
            st.markdown("## Lightning-Fast Auto Charts")

            # Smart sampling for speed
            if len(df) > 5000:
                df_viz = df.sample(5000, random_state=42)
                st.info(f"Showing 5,000 random rows for speed (total rows: {len(df):,})")
            else:
                df_viz = df.copy()

            numeric_cols = df_viz.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df_viz.select_dtypes(include='object').columns.tolist()

            if not numeric_cols:
                st.warning("No numeric columns found for charts.")
            else:
                chart_type = st.selectbox("Choose Chart Type", [
                    "Bar Chart", "Line Chart (Time)", "Pie/Donut", "Scatter Plot",
                    "Box Plot", "Heatmap", "Treemap", "Pareto", "Bubble Chart"
                ])

                if chart_type == "Bar Chart" and cat_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        cat = st.selectbox("Category", cat_cols, key="bar_c")
                    with col2:
                        num = st.selectbox("Value", numeric_cols, key="bar_n")
                    top10 = df_viz.groupby(cat)[num].sum().nlargest(10)
                    fig = px.bar(x=top10.index, y=top10.values, title=f"Top 10 {cat} by {num}")
                    fig.update_layout(height=500, plot_bgcolor="#0f0f0f", paper_bgcolor="#1a1a1a", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Line Chart (Time)":
                    date_col = st.selectbox("Date column", df_viz.columns[df_viz.dtypes == 'datetime64[ns]'], key="date")
                    if date_col:
                        num = st.selectbox("Value", numeric_cols, key="line_n")
                        df_time = df_viz.set_index(date_col).resample('M').sum().reset_index()
                        fig = px.line(df_time, x=date_col, y=num, title=f"Trend of {num}")
                        fig.update_layout(height=500, plot_bgcolor="#0f0f0f", paper_bgcolor="#1a1a1a", font_color="white")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pie/Donut" and cat_cols:
                    cat = st.selectbox("Category", cat_cols, key="pie_c")
                    top8 = df_viz[cat].value_counts().head(8)
                    fig = px.pie(values=top8.values, names=top8.index, hole=0.4, title="Donut Chart")
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Scatter Plot" and len(numeric_cols)>=2:
                    x = st.selectbox("X axis", numeric_cols, key="sc_x")
                    y = st.selectbox("Y axis", numeric_cols, index=1, key="sc_y")
                    fig = px.scatter(df_viz, x=x, y=y, color=cat_cols[0] if cat_cols else None, trendline="ols")
                    fig.update_layout(height=500, plot_bgcolor="#0f0f0f", paper_bgcolor="#1a1a1a", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Box Plot":
                    col = st.selectbox("Column", numeric_cols, key="box_c")
                    fig = px.box(df_viz, y=col, points="outliers")
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Heatmap" and len(numeric_cols)>=3:
                    corr = df_viz[numeric_cols[:10]].corr()
                    fig = px.imshow(corr, text_auto=True, height=600)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Treemap" and cat_cols:
                    cat = st.selectbox("Category", cat_cols, key="tree_c")
                    fig = px.treemap(df_viz, path=[cat], values=numeric_cols[0])
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pareto" and cat_cols:
                    cat = st.selectbox("Category", cat_cols, key="par_c")
                    num = st.selectbox("Value", numeric_cols, key="par_n")
                    temp = df_viz.groupby(cat)[num].sum().sort_values(ascending=False)
                    cum = temp.cumsum()/temp.sum()*100
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=temp.index, y=temp.values))
                    fig.add_trace(go.Scatter(x=temp.index, y=cum, mode='lines+markers', yaxis="y2"))
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Bubble Chart" and len(numeric_cols)>=3:
                    fig = px.scatter(df_viz, x=numeric_cols[0], y=numeric_cols[1], size=numeric_cols[2],
                                   color=cat_cols[0] if cat_cols else None)
                    st.plotly_chart(fig, use_container_width=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                st.write("**Strengths:**")
                for point in analysis['strong_points']:
                    st.success(point)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_s2:
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                st.write("**Issues:**")
                for point in analysis['weak_points']:
                    st.error(point)
                st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Summary Stats")
            st.dataframe(analysis['stats'].round(2))

            if 'correlation' in analysis and not analysis['correlation'].empty:
                st.subheader("Correlation Matrix")
                fig = px.imshow(analysis['correlation'], color_continuous_scale='RdBu_r', title="Correlations")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Machine Learning Models")

            ml_type = st.selectbox("Select ML Task", ["Regression", "Classification", "Clustering", "PCA"])

            if ml_type == "Regression":
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Feature (X)", numeric_cols)
                    y_col = st.selectbox("Target (Y)", [c for c in numeric_cols if c != x_col])
                    model_choice = st.selectbox("Model", ["Linear", "Random Forest"])
                    if st.button("Train Model"):
                        pred, r2, model = predict_trend(df, x_col, y_col, 'rf' if model_choice == "Random Forest" else 'linear')
                        st.session_state.ml_model = {'type': 'regression', 'model': model, 'r2': r2}
                        st.success(f"R² Score: {r2:.3f}")
                        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols" if model_choice == "Linear" else None)
                        st.plotly_chart(fig)
                        if model_choice == "Random Forest":
                            st.bar_chart(model.feature_importances_)
                st.markdown('</div>', unsafe_allow_html=True)

            elif ml_type == "Classification":
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                target_col = st.selectbox("Target Column", df.columns)
                if st.button("Train Classifier"):
                    try:
                        model, acc, cv_mean, importances = classify_data(df, target_col)
                        st.session_state.ml_model = {'type': 'classification', 'model': model, 'acc': acc, 'cv': cv_mean}
                        st.success(f"Accuracy: {acc:.3f}, CV Mean: {cv_mean:.3f}")
                        st.bar_chart(importances)
                    except Exception as e:
                        st.error(f"Error: {e} - Ensure target is categorical/numeric class.")
                st.markdown('</div>', unsafe_allow_html=True)

            elif ml_type == "Clustering":
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                n_clusters = st.slider("Clusters", 2, 10, 3)
                if st.button("Cluster Data"):
                    clusters, kmeans, scaler = cluster_data(df, n_clusters)
                    df['Cluster'] = clusters
                    st.session_state.ml_model = {'type': 'clustering', 'model': kmeans}
                    fig = px.scatter(df, x=df.select_dtypes(np.number).columns[0], y=df.select_dtypes(np.number).columns[1], color='Cluster')
                    st.plotly_chart(fig)
                    st.dataframe(df.head())
                st.markdown('</div>', unsafe_allow_html=True)

            elif ml_type == "PCA":
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                n_comp = st.slider("Components", 2, min(10, len(df.select_dtypes(np.number).columns)), 2)
                if st.button("Run PCA"):
                    pca_result, pca, scaler = pca_analysis(df, n_comp)
                    st.session_state.ml_model = {'type': 'pca', 'model': pca}
                    st.info(f"Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")
                    df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_comp)])
                    fig = px.scatter(df_pca, x='PC1', y='PC2')
                    st.plotly_chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.subheader("Advanced Analyst Tools")

            tool_type = st.selectbox("Tool", ["Time Series", "Anomaly Detection"])

            if tool_type == "Time Series":
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                if analysis['date_cols']:
                    date_col = st.selectbox("Date Col", analysis['date_cols'])
                    value_col = st.selectbox("Value Col", df.select_dtypes(include=np.number).columns)
                    if st.button("Decompose"):
                        result = time_series_analysis(df, date_col, value_col)
                        fig, axes = plt.subplots(4, 1, figsize=(10, 8), facecolor='black')
                        axes[0].set_facecolor('black')
                        axes[1].set_facecolor('black')
                        axes[2].set_facecolor('black')
                        axes[3].set_facecolor('black')
                        result.observed.plot(ax=axes[0], color='white')
                        result.trend.plot(ax=axes[1], color='white')
                        result.seasonal.plot(ax=axes[2], color='white')
                        result.resid.plot(ax=axes[3], color='white')
                        for ax in axes:
                            ax.tick_params(colors='white')
                            ax.spines['bottom'].set_color('white')
                            ax.spines['top'].set_color('white')
                            ax.spines['right'].set_color('white')
                            ax.spines['left'].set_color('white')
                        plt.tight_layout()
                        st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            if tool_type == "Anomaly Detection":
                st.markdown('<div class="ml-box">', unsafe_allow_html=True)
                if st.button("Detect Anomalies"):
                    anomalies = detect_anomalies(df)
                    if anomalies.any().any():
                        st.warning("Anomalies detected.")
                        numeric_cols = df.select_dtypes(np.number).columns
                        col = st.selectbox("View Anomalies in Col", numeric_cols)
                        fig = px.scatter(df, x=df.index, y=col, color=anomalies[col], color_discrete_map={True: 'red', False: 'blue'})
                        st.plotly_chart(fig)
                    else:
                        st.success("No anomalies found.")
                st.markdown('</div>', unsafe_allow_html=True)

        with tab5:
            st.subheader("AI Assistant")
            if ai_enabled:
                context = f"Data: {df.shape}\nColumns: {list(df.columns)}\nHead:\n{df.head().to_string()}\nAnalysis: {analysis['strong_points']}, {analysis['weak_points']}\nML: {st.session_state.ml_model}"
                for msg in st.session_state.conversation:
                    with st.chat_message(msg['role']):
                        st.write(msg['content'])
                if prompt := st.chat_input("Query data/ML..."):
                    st.session_state.conversation.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            response = generate_ai_response(prompt, context)
                            st.write(response)
                            st.session_state.conversation.append({"role": "assistant", "content": response})
            else:
                st.warning("Enable AI.")

            st.subheader("Exports")
            col1, col2 = st.columns(2)
            with col1:


                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import io
                from io import BytesIO
                from PIL import Image
                from datetime import datetime


                st.subheader( "Generate Smart PDF Report with Advanced Features" )

                # Let user choose what to include
                st.write( "### What to Include in PDF" )
                col1 , col2 , col3 = st.columns( 3 )
                with col1:
                    include_summary = st.checkbox( "Executive Summary" , value = True )
                    include_scatter = st.checkbox( "Scatter Plot" , value = True )
                    include_distribution = st.checkbox( "Distribution Chart" , value = True )
                    include_stats_table = st.checkbox( "Statistics Table" , value = True )
                with col2:
                    include_boxplot = st.checkbox( "Outliers (Box Plot)" , value = True )
                    include_correlation = st.checkbox( "Correlation Heatmap" ,
                                                       value = len(
                                                           df.select_dtypes( include = 'number' ).columns ) >= 3 )
                    include_pie = st.checkbox( "Category Breakdown (Pie)" , value = True )
                    include_data_quality = st.checkbox( "Data Quality Score" , value = True )
                with col3:
                    include_duplicates = st.checkbox( "Duplicate Detection" , value = True )
                    include_missing = st.checkbox( "Missing Data Heatmap" , value = True )
                    include_top_categories = st.checkbox( "Top Categories" , value = True )
                    include_completeness = st.checkbox( "Data Completeness" , value = True )

                if st.button( "Generate & Download Enhanced PDF Report" , type = "primary" ,
                              use_container_width = True ):
                    with st.spinner( "Creating your comprehensive PDF..." ):
                        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
                        sample = df.sample( min( 1500 , len( df ) ) , random_state = 42 )
                        num_cols = df.select_dtypes( include = 'number' ).columns.tolist()
                        cat_cols = df.select_dtypes( include = 'object' ).columns.tolist()

                        figs = []

                        # 1. EXECUTIVE SUMMARY
                        if include_summary:
                            missing_pct = (df.isna().sum().sum() / (len( df ) * len( df.columns ))) * 100
                            data_quality = "Excellent" if missing_pct < 5 else "Good" if missing_pct < 20 else "Needs Cleaning"
                            duplicate_count = df.duplicated().sum()

                            summary_text = f"""
                DATAFORGE AI - COMPREHENSIVE REPORT
                Generated: {datetime.now().strftime( '%B %d, %Y - %H:%M' )}

                DATASET OVERVIEW
                Total Rows: {len( df ):,}
                Total Columns: {len( df.columns )}
                Numeric Columns: {len( num_cols )}
                Categorical Columns: {len( cat_cols )}
                Missing Values: {df.isna().sum().sum():,} ({missing_pct:.2f}%)
                Duplicate Rows: {duplicate_count:,}

                KEY METRICS
                Data Quality: {data_quality}
                Data Completeness: {(100 - missing_pct):.2f}%
                Unique Values (First Col): {df[df.columns[0]].nunique():,}
                            """

                            fig_text = go.Figure()
                            fig_text.add_annotation(
                                text = summary_text ,
                                xref = "paper" , yref = "paper" ,
                                x = 0.5 , y = 0.5 , showarrow = False ,
                                font = dict( size = 13 , color = "#1f2937" , family = "Courier New" ) ,
                                align = "left" ,
                                bgcolor = "#f0f9ff" ,
                                bordercolor = "#0ea5e9" ,
                                borderwidth = 3
                            )
                            fig_text.update_layout(
                                template = "plotly_white" ,
                                title = "EXECUTIVE SUMMARY" ,
                                height = 600 ,
                                margin = dict( l = 50 , r = 50 , t = 80 , b = 50 ) ,
                                font = dict( family = "Arial, sans-serif" , size = 12 )
                            )
                            figs.append( fig_text )

                        # 2. DATA QUALITY SCORE (Gauge Chart)
                        if include_data_quality:
                            missing_pct = (df.isna().sum().sum() / (len( df ) * len( df.columns ))) * 100
                            quality_score = max( 0 , 100 - missing_pct )

                            fig = go.Figure( go.Indicator(
                                mode = "gauge+number+delta" ,
                                value = quality_score ,
                                title = {'text': "Data Quality Score"} ,
                                delta = {'reference': 80} ,
                                gauge = {
                                    'axis': {'range': [0 , 100]} ,
                                    'bar': {'color': "#10b981"} ,
                                    'steps': [
                                        {'range': [0 , 50] , 'color': "#fee2e2"} ,
                                        {'range': [50 , 80] , 'color': "#fef3c7"} ,
                                        {'range': [80 , 100] , 'color': "#dcfce7"}
                                    ] ,
                                    'threshold': {
                                        'line': {'color': "red" , 'width': 4} ,
                                        'thickness': 0.75 ,
                                        'value': 90
                                    }
                                }
                            ) )
                            fig.update_layout(
                                template = "plotly_white" ,
                                height = 500 ,
                                font = dict( size = 14 , color = "#1f2937" )
                            )
                            figs.append( fig )

                        # 3. STATISTICS TABLE
                        if include_stats_table and num_cols:
                            stats_data = df[num_cols].describe().round( 2 ).reset_index()
                            stats_data.columns = ['Stat'] + num_cols[:5]

                            fig = go.Figure( data = [go.Table(
                                header = dict(
                                    values = ['<b>' + col + '</b>' for col in stats_data.columns] ,
                                    fill_color = '#0ea5e9' ,
                                    align = 'center' ,
                                    font = dict( color = 'white' , size = 12 )
                                ) ,
                                cells = dict(
                                    values = [stats_data[col] for col in stats_data.columns] ,
                                    fill_color = '#f0f9ff' ,
                                    align = 'center' ,
                                    font = dict( color = '#1f2937' , size = 11 ) ,
                                    height = 30
                                )
                            )] )
                            fig.update_layout(
                                template = "plotly_white" ,
                                title = "Statistical Summary (Numeric Columns)" ,
                                height = 550 ,
                                margin = dict( l = 20 , r = 20 , t = 80 , b = 20 )
                            )
                            figs.append( fig )

                        # 4. SCATTER PLOT
                        if include_scatter and len( num_cols ) >= 2:
                            fig = px.scatter(
                                sample , x = num_cols[0] , y = num_cols[1] ,
                                color = cat_cols[0] if cat_cols else None ,
                                title = f"Relationship: {num_cols[1]} vs {num_cols[0]}" ,
                                color_discrete_sequence = px.colors.qualitative.Pastel ,
                                opacity = 0.7
                            )
                            fig.update_traces( marker = dict( size = 8 ) )
                            fig.update_layout( template = "plotly_white" , height = 550 , font = dict( size = 11 ) )
                            figs.append( fig )

                        # 5. DISTRIBUTION CHART
                        if include_distribution and num_cols:
                            fig = px.histogram(
                                sample , x = num_cols[0] , nbins = 40 ,
                                title = f"Distribution of {num_cols[0]}" ,
                                color_discrete_sequence = ["#06b6d4"]
                            )
                            fig.update_traces( marker_line_color = '#0891b2' , marker_line_width = 1.5 )
                            fig.update_layout( template = "plotly_white" , height = 500 , font = dict( size = 11 ) )
                            figs.append( fig )

                        # 6. BOX PLOT (Outliers)
                        if include_boxplot and num_cols:
                            fig = px.box(
                                df[num_cols[:8]] ,
                                title = "Outliers Detection" ,
                                color_discrete_sequence = px.colors.qualitative.Set2
                            )
                            fig.update_layout( template = "plotly_white" , height = 550 , font = dict( size = 11 ) )
                            figs.append( fig )

                        # 7. CORRELATION HEATMAP
                        if include_correlation and len( num_cols ) >= 3:
                            corr = df[num_cols].corr().round( 2 )
                            fig = px.imshow(
                                corr , text_auto = True ,
                                color_continuous_scale = "Viridis" ,
                                title = "Correlation Matrix" ,
                                labels = dict( color = "Correlation" )
                            )
                            fig.update_layout( template = "plotly_white" , height = 600 , font = dict( size = 11 ) )
                            figs.append( fig )

                        # 8. DUPLICATE DETECTION
                        if include_duplicates:
                            duplicate_count = df.duplicated().sum()
                            unique_count = len( df ) - duplicate_count

                            fig = go.Figure( data = [
                                go.Bar(
                                    x = ['Unique Rows' , 'Duplicate Rows'] ,
                                    y = [unique_count , duplicate_count] ,
                                    marker = dict( color = ['#10b981' , '#ef4444'] ) ,
                                    text = [unique_count , duplicate_count] ,
                                    textposition = 'auto' ,
                                )
                            ] )
                            fig.update_layout(
                                template = "plotly_white" ,
                                title = "Duplicate Records Detection" ,
                                height = 450 ,
                                showlegend = False ,
                                font = dict( size = 12 ) ,
                                xaxis_title = "Record Type" ,
                                yaxis_title = "Count"
                            )
                            figs.append( fig )

                        # 9. MISSING DATA HEATMAP
                        if include_missing:
                            missing_data = df.isnull().sum()
                            missing_pct = (missing_data / len( df ) * 100).round( 2 )

                            fig = go.Figure( data = [go.Bar(
                                x = missing_data.index[:15] ,
                                y = missing_pct[:15] ,
                                marker = dict( color = missing_pct[:15] , colorscale = 'Reds' , showscale = True ) ,
                                text = missing_pct[:15] ,
                                textposition = 'auto'
                            )] )
                            fig.update_layout(
                                template = "plotly_white" ,
                                title = "Missing Data Analysis" ,
                                height = 500 ,
                                xaxis_title = "Columns" ,
                                yaxis_title = "Missing %" ,
                                font = dict( size = 11 )
                            )
                            figs.append( fig )

                        # 10. TOP CATEGORIES
                        if include_top_categories and cat_cols:
                            top_cat = df[cat_cols[0]].value_counts().head( 10 )

                            fig = px.bar(
                                x = top_cat.values , y = top_cat.index ,
                                orientation = 'h' ,
                                title = f"Top 10 Categories in {cat_cols[0]}" ,
                                color = top_cat.values ,
                                color_continuous_scale = "Blues"
                            )
                            fig.update_layout( template = "plotly_white" , height = 500 , font = dict( size = 11 ) )
                            figs.append( fig )

                        # 11. PIE CHART (Category Breakdown)
                        if include_pie and cat_cols:
                            cat_dist = df[cat_cols[0]].value_counts().head( 8 )

                            fig = px.pie(
                                values = cat_dist.values , names = cat_dist.index ,
                                title = f"{cat_cols[0]} Distribution" ,
                                color_discrete_sequence = px.colors.qualitative.Pastel
                            )
                            fig.update_layout( template = "plotly_white" , height = 550 , font = dict( size = 11 ) )
                            figs.append( fig )

                        # 12. DATA COMPLETENESS
                        if include_completeness:
                            completeness = ((len( df ) - df.isnull().sum()) / len( df ) * 100).round( 2 )
                            completeness = completeness[completeness > 0].head( 12 )

                            fig = go.Figure( data = [go.Bar(
                                x = completeness.index ,
                                y = completeness.values ,
                                marker = dict( color = completeness.values , colorscale = 'Greens' ,
                                               showscale = False ) ,
                                text = completeness.values ,
                                textposition = 'auto'
                            )] )
                            fig.update_layout(
                                template = "plotly_white" ,
                                title = "Data Completeness by Column" ,
                                height = 500 ,
                                xaxis_title = "Columns" ,
                                yaxis_title = "Completeness %" ,
                                font = dict( size = 11 ) ,
                                xaxis_tickangle = -45
                            )
                            figs.append( fig )

                        # Convert all figures to PDF
                        pdf_buffer = BytesIO()

                        for i , fig in enumerate( figs ):
                            fig_bytes = fig.to_image( format = "png" , width = 1200 , height = 700 , scale = 2 )
                            img = Image.open( BytesIO( fig_bytes ) )

                            if i == 0:
                                img.save( pdf_buffer , format = 'PDF' )
                            else:
                                img.save( pdf_buffer , format = 'PDF' , append = True )

                        pdf_buffer.seek( 0 )
                        pdf_bytes = pdf_buffer.getvalue()

                        st.success( "Enhanced PDF Report Ready!" )
                        st.balloons()

                        st.download_button(
                            label = "DOWNLOAD YOUR COMPREHENSIVE PDF REPORT" ,
                            data = pdf_bytes ,
                            file_name = f"DataForge_Comprehensive_Report_{datetime.now().strftime( '%Y%m%d_%H%M' )}.pdf" ,
                            mime = "application/pdf" ,
                            use_container_width = True
                        )

                        st.info( f"Report includes {len( figs )} visualizations and analyses" )
def main():
    if st.session_state.view_mode == 'landing' or st.session_state.data is None:
        render_landing_page(None)
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
