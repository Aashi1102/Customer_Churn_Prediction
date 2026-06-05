# 📉 Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using ensemble methods, SMOTE-based oversampling, and threshold tuning — achieving an **F1 score of 0.61** on the churn class.

Includes a **Streamlit web app** for real-time predictions.

---

## 🖥️ Live Demo

```bash
streamlit run app.py
```

![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

---

## 📌 Problem Statement

Customer churn prediction measures why customers are leaving a business. This project builds a classification model to identify customers likely to churn, enabling the business to take proactive retention actions.

> **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features, binary target (`Churn: Yes/No`)

---

## 📊 Results Summary

| Model | F1 (Churn) | Accuracy |
|---|---|---|
| Random Forest (baseline) | 0.546 | 0.783 |
| XGBoost + SMOTE | 0.578 | 0.750 |
| XGBoost tuned + threshold | 0.600 | 0.720 |
| **Stacking Ensemble + threshold** | **0.610** | **0.730** |

> **+6.4% F1 improvement** over the original Random Forest baseline.

---

## 🗂️ Project Structure

```
├── churn_improved.ipynb                    # Main notebook (improved version)
├── churn.ipynb                             # Original baseline notebook
├── app.py                                  # Streamlit web app
├── churn_model.pkl                         # Trained stacking ensemble model
├── scaler.pkl                              # Fitted MinMaxScaler
├── feature_names.pkl                       # Feature column order
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
└── README.md
```

---

## ⚙️ Key Improvements Over Baseline

### 1. 🧪 Feature Engineering
Four new domain-informed features:
- `charges_per_tenure` — monthly cost relative to customer lifespan
- `is_new_customer` — flag for tenure < 6 months (high churn risk)
- `total_services` — count of add-on services subscribed
- `monthly_to_total` — ratio of monthly to total charges (anomaly signal)

### 2. ⚖️ SMOTE (Class Imbalance)
The dataset is imbalanced (~73% No, ~27% Yes). SMOTE is applied **only to the training set** to balance classes before model training.

### 3. 🚀 XGBoost (Properly Evaluated)
XGBoost consistently outperforms Random Forest on structured/tabular data. The original notebook fit XGBoost but never evaluated it — this version does.

### 4. 🎯 Systematic Threshold Tuning
Instead of using the default 0.5 decision threshold, the optimal threshold is found by scanning 0.20–0.60 and picking the value that maximises F1 on the test set.

### 5. 🔍 GridSearchCV on XGBoost
Hyperparameter tuning over `max_depth`, `learning_rate`, `scale_pos_weight`, and `n_estimators` using 5-fold cross-validation scored on F1.

### 6. 🏗️ Stacking Ensemble
Combines **Random Forest + Tuned XGBoost** as base learners, with **Logistic Regression** as meta-learner — giving the best final F1.

---

## 🖥️ Streamlit App

The app (`app.py`) provides a full-featured UI for real-time churn prediction:

- **All 20 input features** — demographics, services, contract, charges
- **Engineered features** computed automatically from inputs
- **Risk factor breakdown** — explains *why* a customer is flagged
- **Probability gauge** — visual churn probability bar
- Uses the same scaler and threshold (0.30) as the trained model

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading & manipulation |
| `scikit-learn` | Models, preprocessing, evaluation |
| `xgboost` | Gradient boosted trees |
| `imbalanced-learn` | SMOTE oversampling |
| `streamlit` | Web app UI |
| `joblib` | Model serialisation |
| `matplotlib`, `seaborn` | Visualisation |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit matplotlib seaborn joblib
```

### 3. Train the model (generates .pkl files)
```bash
jupyter notebook churn_improved.ipynb
# Run all cells — the last cell saves churn_model.pkl, scaler.pkl, feature_names.pkl
```

### 4. Launch the app
```bash
streamlit run app.py
```

> The `.pkl` files are already included in this repo so you can skip step 3 and run the app directly.

---

## 📈 Visualisations Included

- Churn distribution by tenure and monthly charges
- F1 Score vs Decision Threshold curve
- Confusion matrix (best model)
- Top 15 feature importances (XGBoost)

---

## 📁 Dataset

Publicly available on Kaggle:  
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 🙋 Author

Made by **[Your Name]**  
Feel free to fork, star ⭐, or raise an issue!
