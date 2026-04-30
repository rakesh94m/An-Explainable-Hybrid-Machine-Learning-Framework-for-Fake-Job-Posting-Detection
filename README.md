# Fake Job Posting Detection using Hybrid Machine Learning & Explainable AI

## 📌 Overview

This project presents a hybrid machine learning framework to detect fraudulent job postings using both textual and credibility-based features. The system combines TF-IDF-based text representation with structured metadata features such as grammar quality, suspicious keywords, salary transparency, and company information.

In addition to prediction, the system integrates Explainable AI (SHAP) to provide transparent and interpretable results, helping users understand why a job posting is classified as fake or legitimate.

---

## 🚀 Features

* Hybrid feature engineering (Text + Metadata)
* TF-IDF vectorization for text representation
* Credibility features:

  * Grammar Error Ratio (GER)
  * Suspicious keyword detection
  * Salary transparency indicator
  * Company-related features
* Machine Learning models (trained and evaluated):

  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost
  * LightGBM
  * Stacking Ensemble
* **XGBoost used for final deployment**
* Explainable AI using SHAP
* Real-time prediction using Streamlit web app

---

## 📂 Project Structure

```
.
├── app.py
├── training.ipynb
├── requirements.txt
├── fake_job_posting_detection.pdf
├── fake_job_postings.zip
│
├── artifacts/
│   ├── final_tfidf.joblib
│   ├── meta_cols.joblib
│   ├── model_xgb_full.joblib
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset is included as a compressed file:

```
fake_job_postings.zip
```

👉 Extract the dataset before training:

```bash
unzip fake_job_postings.zip
```

Dataset source: Kaggle Fake Job Postings Dataset

---

## ▶️ Running the Application

### Run Streamlit app:

```bash
streamlit run app.py
```

---

## 🧠 Model & Explainability

* The deployed model is **XGBoost**, stored in the `artifacts/` directory.
* SHAP is used to explain predictions:

  * Global feature importance
  * Local instance-level explanations
* Enables transparent and interpretable fraud detection decisions.

---

## 📈 Results

* Achieved high classification accuracy (>97%)
* Stacking model achieved highest ROC-AUC
* **XGBoost selected for deployment due to balanced precision–recall performance and interpretability**
* Significant reduction in false negatives using hybrid features

---

## 🌐 Real-Time Deployment

The application allows users to:

* Input job posting details
* Get fraud probability score
* View SHAP-based explanation of predictions

---

## 📄 Research Paper

The detailed methodology and results are documented in:

```
fake_job_posting_detection.pdf
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Streamlit
* Pandas, NumPy

---

## 👨‍💻 Author

Rakesh Meesa
B.Tech CSE (AI)

---

## ⭐ Note

Pre-trained model and artifacts are included for direct execution. No need to retrain unless modifying the pipeline.
