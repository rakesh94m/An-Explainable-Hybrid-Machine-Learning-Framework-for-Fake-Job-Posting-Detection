import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import shap
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from pathlib import Path
import spacy
from spellchecker import SpellChecker

st.set_page_config(
    page_title="Fake Job Detection",
    layout="wide"
)

ARTIFACTS = Path("artifacts")
FRAUD_THRESHOLD = 0.40   

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS / "model_xgb_full.joblib")
    tfidf = joblib.load(ARTIFACTS / "final_tfidf.joblib")
    meta_cols = joblib.load(ARTIFACTS / "meta_cols.joblib")
    return model, tfidf, meta_cols

model, tfidf, meta_cols = load_artifacts()

# ✅ Create combined feature names AFTER loading artifacts
tfidf_feature_names = tfidf.get_feature_names_out().tolist()
meta_feature_names = meta_cols
all_feature_names = tfidf_feature_names + meta_feature_names

@st.cache_resource
def load_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    spell = SpellChecker()
    return nlp, spell

nlp, spell = load_nlp()
STOPWORDS = nlp.Defaults.stop_words

def clean_text_basic(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOPWORDS and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

SUSPICIOUS_KEYWORDS = [
    "earn", "easy money", "quick money", "paid daily", "paid weekly",
    "work from home", "urgent hiring", "apply immediately",
    "no experience", "no interview", "anyone can apply",
    "bitcoin", "crypto", "wallet", "upi", "gift card",
    "congratulations", "selected", "limited offer", "act now"
]

def suspicious_keyword_score(text):
    t = text.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in t)

def grammar_error_ratio(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    miss = sum(1 for t in tokens if t not in spell)
    return miss / len(tokens)

def build_meta_features(raw_text, clean_text):
    data = {
        "text_length": len(raw_text),
        "word_count": len(raw_text.split()),
        "clean_text_length": len(clean_text),
        "suspicious_keyword_count": suspicious_keyword_score(raw_text),
        "has_suspicious_keyword": int(suspicious_keyword_score(raw_text) > 0),
        "grammar_error_ratio": grammar_error_ratio(raw_text),
        "has_salary": 0,
        "salary_range_length": 0,
        "salary_anomaly": 0,
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0
    }
    return pd.DataFrame([data])[meta_cols]

st.title("Fake Job Posting Detection")
st.write(
    "Paste a job description below. The system will detect whether the job is **fraudulent or legitimate** "
    "and explain the decision using **Explainable AI (SHAP)**."
)

text_input = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the complete job posting text here..."
)

if st.button("Detect Fraud"):
    if not text_input.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Analyzing job posting..."):
            clean = clean_text_basic(text_input)

            X_text = tfidf.transform([clean])
            X_meta = csr_matrix(build_meta_features(text_input, clean).values)
            X_final = hstack([X_text, X_meta])

            prob_fraud = model.predict_proba(X_final)[0][1]
            pred = int(prob_fraud >= FRAUD_THRESHOLD)

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"FRAUDULENT JOB\n\nFraud Probability: **{prob_fraud:.2f}**")
        else:
            st.success(f"LEGITIMATE JOB\n\nConfidence: **{1 - prob_fraud:.2f}**")

        st.subheader("Explanation (Why this decision?)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_final)

        # Handle binary output safely
        if shap_values.values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        # Convert sparse matrix safely
        data_array = X_final.toarray()[0] if hasattr(X_final, "toarray") else X_final[0]

        # Create SHAP Explanation object with feature names
        sv = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=data_array,
            feature_names=all_feature_names
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(sv, max_display=15, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Positive features push the prediction toward **fraud**, "
            "negative features push it toward **legitimate**."
        )
        
