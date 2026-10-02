# Pulse — HealthCare Treatment Outcome Predictor

An interactive **Streamlit** GUI built on top of the `HealthCare.ipynb` notebook. It takes the full notebook pipeline — data cleaning, exploratory analysis, model training, and evaluation — and turns it into a point‑and‑click application, with one added capability the notebook doesn't have: live prediction for a brand-new, unseen patient.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-GUI-FF4B4B)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

The dataset models hospital patients and the goal is to predict `treatment_outcome` from a set of clinical and demographic features (age, gender, vitals, lab results, lifestyle factors, diagnosis, medication, etc.).

This app wraps that whole workflow in four screens:

| Tab | What it does |
|---|---|
| **Overview** | Upload a CSV, see row/column counts, missing values, and summary statistics. |
| **Insights & EDA** | All the exploratory plots from the notebook — age distribution, gender split, diagnosis breakdown, correlation heatmap, BMI vs. blood sugar, outcome by medication, and outlier boxplots. |
| **Model Arena** | Trains and compares 6 models from the notebook side by side, with a leaderboard, bar chart, confusion matrix, and classification report. |
| **Predict a Case** | *New addition.* A form to enter a new patient's data and get an instant outcome prediction from any trained model — something the original notebook doesn't support, since it only evaluates models on its own test split. |

## Features

- 📁 **CSV upload** — bring your own `health_care.csv`; no hardcoded file path.
- 🧹 **Faithful preprocessing** — reproduces the notebook's exact cleaning steps (median/mode imputation, ordinal encoding for risk/adherence/alcohol levels, one-hot encoding for medication, derived `treatment_duration_days`).
- 📊 **Full EDA suite** — every chart from the notebook, rendered live for whatever dataset is uploaded.
- 🤖 **Six trained models** — Logistic Regression, KNN, XGBoost, SVM, Random Forest, and a soft-voting ensemble of the three best performers.
- 🏆 **Model comparison** — Accuracy, Macro F1, Macro Precision, and Macro Recall for every model, plus confusion matrices and classification reports on demand.
- 🔮 **Live inference** — predict an outcome for a patient who isn't in the dataset, using the same preprocessing pipeline the models were trained with.
- 🌙 **Dark, polished UI** — a custom dark theme with styled tabs, metrics, and charts.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **ML:** scikit-learn (Logistic Regression, KNN, SVM, Random Forest, Voting Classifier), XGBoost
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Project Structure

```
.
├── app.py              # Streamlit application (all logic lives here)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- A copy of `health_care.csv` (the dataset used in `HealthCare.ipynb`), with at least a `treatment_outcome` column

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (by default at `http://localhost:8501`). From there:

1. Upload your `health_care.csv` from the sidebar.
2. Browse the **Overview** and **Insights & EDA** tabs to explore the data.
3. Go to **Model Arena** and click **Train All Models** to fit and compare all six models.
4. Switch to **Predict a Case**, fill in a new patient's data, pick a model, and get a prediction.

## Expected Data Format

The uploaded CSV should contain (at minimum) the columns used during training, including a `treatment_outcome` target column. Columns such as `patient_name` and `diagnosis` are automatically dropped before modeling, and missing values are imputed the same way the notebook does.

## Notes & Limitations

- XGBoost is optional — if it isn't installed, the app skips it and trains the remaining five models.
- Model training runs in-session (via Streamlit's caching) and isn't persisted between restarts; retraining is quick but not instantaneous on large datasets.
- This app is for educational/demonstrative purposes and is **not** a certified clinical decision-support tool.

## License

This project is licensed under the MIT License — feel free to use and adapt it for your own coursework or demos.
