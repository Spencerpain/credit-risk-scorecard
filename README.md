# Credit Risk Scorecard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-scorecard-r7ue6drwtpg72hlobfde5u.streamlit.app/)

**Live App:** https://credit-risk-scorecard-r7ue6drwtpg72hlobfde5u.streamlit.app/

Predict probability of default (PD) and compute expected loss metrics using the [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) dataset.

**Models:**
- Logistic Regression Scorecard (WoE/IV binning + PDO scaling)
- XGBoost (raw features, SHAP explainability)

**Outputs:** AUC-ROC, KS Statistic, Gini Coefficient, Scorecard Points Table, Expected Loss by risk band.

---

## Setup

```bash
git clone https://github.com/Spencerpain/credit-risk-scorecard.git
cd credit-risk-scorecard
pip install -r requirements.txt
```

## Get the Data

See [`data/README.md`](data/README.md) for download instructions.

Quick version:
```bash
pip install kaggle
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/
```

## Run

```bash
jupyter notebook notebooks/credit_risk_scorecard.ipynb
```

---

## Methodology

### 1. WoE Binning & IV
Features are binned using monotonic optimal binning. Each bin is assigned a Weight of Evidence (WoE) score, and Information Value (IV) is used to select predictive features.

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | Useless |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3 – 0.5 | Strong |

### 2. Logistic Regression Scorecard
WoE-transformed features are fed into logistic regression. Coefficients are converted to integer scorecard points using the standard PDO scaling formula:

```
Factor = PDO / ln(2)
Offset = Base_Score - Factor × ln(Base_Odds)
Points = -(coef × WoE + intercept/n) × Factor
```

Default settings: **PDO=20, Base Score=600, Base Odds=19:1**

### 3. XGBoost
Trained on raw features with `scale_pos_weight` to handle the ~6.7% default rate. Early stopping prevents overfitting.

### 4. Expected Loss
```
EL = PD × LGD × EAD
```
- **PD**: model output
- **LGD**: 0.45 (Basel II unsecured retail assumption)
- **EAD**: unit exposure (loan balance not available in dataset)

---

## Project Structure

```
credit-risk-scorecard/
├── notebooks/
│   └── credit_risk_scorecard.ipynb   ← Main analysis
├── src/
│   ├── preprocessing.py              ← Data loading & cleaning
│   ├── binning.py                    ← WoE/IV binning
│   ├── scorecard.py                  ← LR scorecard & points table
│   ├── xgboost_model.py              ← XGBoost + SHAP
│   └── evaluation.py                 ← AUC, KS, Gini, plots
├── data/
│   └── README.md                     ← Download instructions
├── outputs/                          ← Generated charts & tables
└── requirements.txt
```
