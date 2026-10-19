# 📊 Telecom Customer Churn Prediction & Retention Analytics

An end-to-end Machine Learning web application and data analysis project built to predict customer churn risk, analyze customer lifecycle patterns, and recommend targeted retention interventions for telecommunication providers.

![Customer Churn Analytics](https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=1200&auto=format&fit=crop&q=80)

---

## 🌟 Features
- **Live Churn Risk Predictor:** Enter customer parameters (contract type, tenure, monthly spend, service addons) to receive an immediate churn risk percentage and classification (Low / Moderate / High Risk).
- **Automated Retention Guidance:** Generates strategic interventions based on feature risk drivers (e.g. promoting annual contracts, bundle incentives).
- **Interactive EDA Dashboard:** Explore churn patterns across payment methods, internet technologies, and tenure cohorts.
- **Production Pipeline:** End-to-end scikit-learn preprocessing and Random Forest classifier achieving **0.842 ROC-AUC** and **80% recall** on churning subscribers.
- **Deployable:** Ready for instant deployment on Streamlit Cloud, Render, or Docker containers.

---

## 🚀 Quickstart

### 1. Local Run
```bash
# Clone repository
git clone https://github.com/ayusprasad/churn-rate-analysis.git
cd churn-rate-analysis

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain model
python train.py

# Launch Streamlit web app
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 2. Run with Docker
```bash
# Build container
docker build -t churn-predictor .

# Run container
docker run -p 8501:8501 churn-predictor
```
Open [http://localhost:8501](http://localhost:8501).

---

## 📁 Project Structure
```text
churn-rate-analysis/
├── app.py                     # Streamlit application dashboard & predictor
├── train.py                   # Model training and pipeline export script
├── telco.csv                  # Dataset (7,043 customer records)
├── churn_model.pkl            # Pretrained Random Forest pipeline artifact
├── churn rate.ipynb           # Original exploratory analysis notebook
├── some more upgrade.ipynb    # Modeling & upgrades notebook
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Production Docker configuration
└── README.md                  # Project documentation
```
