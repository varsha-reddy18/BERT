
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="💜",
    layout="wide"
)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR /
    "data" /
    "twitter_processed.csv"
)

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "bert_model"
)

TOKENIZER_DIR = (
    BASE_DIR /
    "models" /
    "tokenizer"
)

ENCODER_PATH = (
    BASE_DIR /
    "models" /
    "label_encoder.pkl"
)

# ==========================================
# LOAD CSS
# ==========================================

with open(
    BASE_DIR / "style.css"
) as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ==========================================
# LOAD DATA
# ==========================================

try:

    df = pd.read_csv(DATA_PATH)

    if "Tweet" not in df.columns:

        df = pd.read_csv(
            DATA_PATH,
            header=None
        )

        df.columns = [
            "ID",
            "Entity",
            "Sentiment",
            "Tweet"
        ]

except Exception as e:

    st.error(
        f"Dataset Loading Error: {e}"
    )

    st.stop()

# Fix data types
df["Entity"] = df["Entity"].astype(str)
df["Sentiment"] = df["Sentiment"].astype(str)
df["Tweet"] = df["Tweet"].astype(str)

# Tweet Length
df["tweet_length"] = (
    df["Tweet"]
    .apply(
        lambda x:
        len(str(x).split())
    )
)
# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_assets():
    if not TOKENIZER_DIR.exists() or not MODEL_DIR.exists() or not ENCODER_PATH.exists():
        return None, None, None

    try:
        tokenizer = BertTokenizer.from_pretrained(
            str(TOKENIZER_DIR)
        )

        model = TFBertForSequenceClassification.from_pretrained(
            str(MODEL_DIR)
        )

        with open(
            ENCODER_PATH,
            "rb"
        ) as f:
            encoder = pickle.load(f)

        return (
            tokenizer,
            model,
            encoder
        )
    except Exception:
        return None, None, None

tokenizer, model, encoder = load_assets()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown(
    """
    <h1 style='text-align:center;color:#a855f7'>
    </h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.title(
    "💜 BERT Dashboard"
)

# Sidebar model status indicator
if tokenizer is None or model is None or encoder is None:
    st.sidebar.markdown(
        """
        <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <span style="color: #fca5a5; font-size: 0.85rem; font-weight: bold;">⚠️ MODEL MISSING</span><br/>
            <span style="color: #cbd5e1; font-size: 0.75rem;">Run train_model.py to initialize.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <span style="color: #a7f3d0; font-size: 0.85rem; font-weight: bold;">✅ MODEL READY</span><br/>
            <span style="color: #cbd5e1; font-size: 0.75rem;">BERT weights & tokenizer loaded.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Analysis",
        "Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Model: BERT Base

Architecture:
Transformer Encoder

Task:
Multi-Class Sentiment Analysis
"""
)

# ==========================================
# HOME
# ==========================================

if page == "Home":

    st.markdown(
    """
    <div class='main-header'>
    <h1>💜 BERT Sentiment Analysis Dashboard</h1>
    <p>Transformer Encoder Based NLP System</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Tweets",
            len(df)
        )

    with col2:
        st.metric(
            "Classes",
            df["Sentiment"].nunique()
        )

    with col3:
        st.metric(
            "Model",
            "BERT"
        )

    with col4:
        st.metric(
            "Architecture",
            "Encoder"
        )

    st.info(
        """
        BERT uses Bidirectional Attention
        to understand text context and
        classify sentiment accurately.
        """
    )

# ==========================================
# DATASET
# ==========================================
# ==========================================
# DATASET
# ==========================================

elif page == "Dataset":

    st.title(
        "📊 Dataset Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            len(df)
        )

    with col2:
        st.metric(
            "Unique Entities",
            df["Entity"].nunique()
        )

    with col3:
        st.metric(
            "Sentiment Classes",
            df["Sentiment"].nunique()
        )

    st.markdown("---")

    st.subheader(
        "Dataset Preview"
    )

    preview_df = (
        df[
            [
                "Entity",
                "Sentiment",
                "Tweet"
            ]
        ]
        .head(20)
        .copy()
    )

    st.dataframe(
        preview_df,
        width="stretch"
    )

    st.markdown("---")

    st.subheader(
        "Dataset Information"
    )

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ]
        }
    )

    st.dataframe(
        info_df,
        width="stretch"
    )

# ==========================================
# ANALYSIS
# ==========================================
# ==========================================
# ANALYSIS
# ==========================================

elif page == "Analysis":

    st.title(
        "📈 Sentiment Analytics"
    )

    st.subheader(
        "Sentiment Distribution"
    )

    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
    )

    st.bar_chart(
        sentiment_counts
    )

    st.markdown("---")

    st.subheader(
        "Top Entities"
    )

    entity_counts = (
        df["Entity"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        entity_counts
    )

    st.markdown("---")

    st.subheader(
        "Tweet Length Distribution"
    )

    st.line_chart(
        df["tweet_length"]
    )
# ==========================================
# PREDICTION
# ==========================================

elif page == "Prediction":

    st.title(
        "🔮 Sentiment Prediction"
    )

    if model is None or tokenizer is None or encoder is None:
        st.markdown(
            """
            <div class="result-card" style="border-left: 6px solid #ef4444;">
                <h3 style="color: #f87171 !important; margin-top: 0;">⚠️ BERT Model Assets Missing</h3>
                <p>To use the Sentiment Prediction interface, you first need to train the model to generate the weights and files.</p>
                <ol style="margin-bottom: 20px;">
                    <li>Open your terminal/command prompt.</li>
                    <li>Execute the training script to run on a fast subset of the data:
                        <pre style="background: rgba(0,0,0,0.4); padding: 10px; border-radius: 8px; font-family: monospace; color: #a78bfa; margin-top: 5px;">python implementation/train_model.py</pre>
                    </li>
                    <li>After the script prints <code>"BERT Model Saved"</code>, refresh this dashboard!</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        text = st.text_area(
            "Enter Tweet / Text to Analyze",
            placeholder="Type your tweet or sentence here...",
            height=150
        )

        if st.button(
            "Predict Sentiment"
        ):

            if text.strip() == "":

                st.warning(
                    "Please enter some text."
                )

            else:

                encoded = tokenizer(
                    text,
                    return_tensors="tf",
                    truncation=True,
                    padding=True,
                    max_length=128
                )

                outputs = model(
                    encoded
                )

                probs = tf.nn.softmax(
                    outputs.logits,
                    axis=1
                )

                pred = tf.argmax(
                    probs,
                    axis=1
                ).numpy()[0]

                sentiment = (
                    encoder
                    .inverse_transform(
                        [pred]
                    )[0]
                )

                confidence = float(
                    np.max(
                        probs.numpy()
                    )
                )

                # Visual badge mapping
                badge_colors = {
                    "Positive": "background: linear-gradient(135deg, #10b981, #059669); color: white;",
                    "Negative": "background: linear-gradient(135deg, #f43f5e, #e11d48); color: white;",
                    "Neutral": "background: linear-gradient(135deg, #f59e0b, #d97706); color: white;",
                    "Irrelevant": "background: linear-gradient(135deg, #64748b, #475569); color: white;"
                }
                badge_style = badge_colors.get(sentiment, "background: #a855f7; color: white;")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="font-size: 0.95rem; opacity: 0.7; margin-bottom: 8px;">PREDICTION RESULT</div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 15px; flex-wrap: wrap;">
                            <span class="sentiment-badge" style="{badge_style}">{sentiment}</span>
                            <div style="text-align: right;">
                                <div style="font-size: 1.8rem; font-weight: 800; color: #c084fc; line-height: 1;">{confidence:.2%}</div>
                                <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 4px;">Confidence Score</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Class probabilities visualization
                st.subheader(
                    "Class Probabilities"
                )

                prob_dict = {
                    encoder.classes_[i]: float(probs.numpy()[0][i])
                    for i in range(len(encoder.classes_))
                }

                # Display custom progress bars
                for cls, val in prob_dict.items():
                    bar_color = "#3b82f6"
                    if cls == "Positive":
                        bar_color = "#10b981"
                    elif cls == "Negative":
                        bar_color = "#ef4444"
                    elif cls == "Neutral":
                        bar_color = "#f59e0b"
                    elif cls == "Irrelevant":
                        bar_color = "#6b7280"

                    st.markdown(
                        f"""
                        <div style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.9rem;">
                                <span>{cls}</span>
                                <strong>{val:.2%}</strong>
                            </div>
                            <div style="background: rgba(255,255,255,0.05); width:100%; height:12px; border-radius:6px; overflow:hidden; border:1px solid rgba(255,255,255,0.05);">
                                <div style="background: {bar_color}; width: {val*100}%; height: 100%; border-radius:6px; transition: width 0.5s ease-in-out;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )