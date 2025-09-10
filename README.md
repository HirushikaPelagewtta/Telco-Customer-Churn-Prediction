# Customer Churn Prediction (Telco) — Production-Ready ML

A complete, business-focused churn prediction system built on the Kaggle Telco Customer Churn dataset. The repo includes exploratory data analysis, robust preprocessing, multiple model families (incl. CatBoost), hyperparameter tuning, threshold optimization, and a reproducible pipeline layout suitable for production.

> **Headline results:** Best model is **CatBoost** with **F1 ≈ 64%**, **Recall ≈ 71%**, **Precision ≈ 58%**, **Accuracy ≈ 79%** . High churn risk is driven by **month-to-month contracts**, **electronic check payments**, **short tenure**, and **higher monthly charges** .

---

## 📂 Repository Structure

```
.
├── Artifacts/                         # Saved models, encoders/scalers, reports
├── Data/                              # Raw / interim / processed data (see data/raw and temp files)
├── catboost_info/                     # CatBoost training logs/metadata
├── artifacts/                         # (pipeline run outputs – splits, models, metrics)
├── data/
│   └── raw/                           # Raw CSV(s)
├── pipelines/                         # Orchestrated steps for ingestion/splitting/training
├── src/                               # Core library code used by pipelines
├── utils/                             # Helpers (I/O, metrics, config, etc.)
├── 0_Handling_Missing_Values.ipynb    # EDA & data cleaning
├── 1_Handling_Outliers.ipynb
├── 2_Feature_Binning.ipynb
├── 3_Encoding_And_Scaling.ipynb
├── 1_Base_Model_Training.ipynb        # Baselines + initial eval
├── 2_kfold_validation.ipynb
├── 3_Multi_model_training.ipynb       # RF/XGB/CatBoost training
├── 4_Hyperparam_tuning.ipynb
├── 5_Threshold_optimization.ipynb
├── 6_Using_the_Best_Model.ipynb       # Inference/serving demo
├── config.yaml                        # Centralized configuration
├── requirements.txt                   # Python dependencies
├── Makefile                           # Optional convenience commands
├── .env                               # Local environment variables (optional)
└── temp_imputed.csv                   # Intermediate artifact from preprocessing
```

> The Jupyter notebooks show the full **data → features → model → tune → threshold → use** workflow, while `pipelines/`, `src/`, and `utils/` contain reusable code used to make the process repeatable.

---

## 🎯 Problem & Business Framing

* **Goal:** Predict which customers will churn so the business can target retention actions.
* **Dataset:** Telco Customer Churn (≈7,043 rows, 21 features; label: `Churn`)&#x20;
* **Deliverables:** Cleaned features, multiple trained models (LogReg/DT/RF/XGBoost/CatBoost), and evaluation using class-imbalance-aware metrics (Precision, Recall, F1, Accuracy) .

### Why it matters

At the chosen decision threshold, **\~71%** of at-risk customers can be captured (Recall), and **\~58%** of contacted customers are true churners (Precision), improving campaign ROI .
A cost snapshot in the report shows a positive **net expected ROI** when using these predictions to drive targeted offers .

---

## ✅ Results (Summary)

| Model               | F1 (test)   |
| ------------------- | ----------- |
| Logistic Regression | \~79.9%     |
| Decision Tree       | \~80.0%     |
| Random Forest       | \~84.7%     |
| XGBoost             | \~84.8%     |
| **CatBoost**        | **\~85.6%** |

*As reported in the executive summary comparison table* .

**Key churn drivers:** month-to-month contracts, electronic check payments, short tenure, high monthly charges .
**Recommended actions:** incentives to move to annual contracts, value-add bundles (security/tech support), prioritize outreach to top decile risk, tailor channel by payment method .

---

## 🛠️ Setup

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) set environment variables
cp .env.example .env  # if provided

# 4) Verify
python -c "import sklearn, xgboost, catboost; print('ok')"
```

---

## 🚀 How to Run

### Option A — Reproduce the workflow with notebooks

Open the notebooks in this order:

1. `0_Handling_Missing_Values.ipynb` → cleaning & imputations
2. `1_Handling_Outliers.ipynb`
3. `2_Feature_Binning.ipynb`
4. `3_Encoding_And_Scaling.ipynb`
5. `1_Base_Model_Training.ipynb`
6. `2_kfold_validation.ipynb`
7. `3_Multi_model_training.ipynb`
8. `4_Hyperparam_tuning.ipynb`
9. `5_Threshold_optimization.ipynb`
10. `6_Using_the_Best_Model.ipynb` → load best model, run inference

Intermediate outputs and trained models are saved under `Artifacts/` or `artifacts/`.

### Option B — Pipeline scripts (if provided)

If the repo includes CLI entry points for the modules in `pipelines/` and `src/`, you can wire them with the `Makefile`:

```bash
# Examples (adjust targets to what's implemented in your Makefile)
make ingest         # read raw data into data/raw
make split          # train/valid/test split
make train          # train baseline and ensemble models
make tune           # hyperparameter search
make evaluate       # metrics & confusion matrix
make predict        # run inference with the best model
```

Configuration such as paths, feature lists, and model params is centralized in **`config.yaml`**.

---

## 📊 Evaluation & Thresholding

* **Metrics:** Precision, Recall, F1, Accuracy (stratified CV and test set).
* **Class imbalance:** addressed via threshold tuning and evaluation emphasis on Recall / F1.
* **Threshold optimization:** `5_Threshold_optimization.ipynb` chooses an operating point that balances business cost/benefit; see the ROI table and recommendations in the report .

---

## 🧠 Feature Engineering (Highlights)

* Tenure binning (New / Established / Loyal), categorical encoding, numeric scaling .
* Payment type, contract type, and service add-ons captured to expose churn signals (drivers listed above).

---

## 🧪 Reusing the Best Model

Use `6_Using_the_Best_Model.ipynb` to load the exported pipeline and run batch or single-row predictions. For programmatic usage, import the packaged pipeline from `src/` (e.g., `from src.pipeline import load_model, predict` if present) and pass a dataframe with the same schema as the training data.

---

## 📈 Business Playbook (from the report)

* **Convert** month-to-month → annual with targeted incentives.
* **Bundle** tech-support/security add-ons for at-risk segments.
* **Prioritize** top-decile churn probability for maximum ROI.
* **Personalize** outreach channel based on payment method (e.g., electronic check → email reminders/offers).
  All are summarized and supported in the executive summary .

---

## 🔒 Reproducibility & Hygiene

* All transformations are embedded into scikit-learn/CatBoost pipelines for **train/test parity** and **one-line inference** .
* Parameters and paths live in `config.yaml`.
* Deterministic splits via fixed seeds (where applicable).
* Artifacts versioned under `Artifacts/` or `artifacts/`.

---

## 🧭 Roadmap

* Add experiment tracking (MLflow or Weights & Biases).
* Export a lightweight FastAPI service for real-time scoring.
* Implement calibration and cost-sensitive training variants.
* Add unit tests (pytest) for `src/` and schema checks (pandera).

---

## 📜 Citation

If you use or reference the results/figures in this repo, please cite the accompanying **Executive Summary Report** for metrics, drivers, and business analysis:   .

---

**Author:** H. P. Pelagewatta
**Project:** Mini Project 0 — Advanced Telco: Customer Churn Prediction (Production-Ready ML)&#x20;
