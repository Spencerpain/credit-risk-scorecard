import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, confusion_matrix

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")
st.title("Credit Risk Scorecard")

@st.cache_data
def load_data():
    import os, zipfile, subprocess
    if not os.path.exists("data/cs-training.csv"):
        os.makedirs("data", exist_ok=True)
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
        if not os.path.exists(kaggle_json):
            username = st.secrets["KAGGLE_USERNAME"]
            key = st.secrets["KAGGLE_KEY"]
            with open(kaggle_json, "w") as f:
                import json
                json.dump({"username": username, "token": key}, f)
            os.chmod(kaggle_json, 0o600)
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "GiveMeSomeCredit", "-p", "data"],
            check=True
        )
        with zipfile.ZipFile("data/GiveMeSomeCredit.zip", "r") as z:
            z.extractall("data")
    df = pd.read_csv("data/cs-training.csv", index_col=0)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X, X_test, y_test

def prob_to_score(prob, pdo=20, base_score=600, base_odds=50):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    score = offset - factor * np.log(prob / (1 - prob + 1e-10))
    return np.clip(score, 300, 850)

df = load_data()
model, X, X_test, y_test = train_model(df)
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

tab1, tab2, tab3 = st.tabs(["Dataset", "Model Performance", "Score Calculator"])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['SeriousDlqin2yrs'].mean():.1%}")
    col3.metric("Features", len(df.columns) - 1)

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature", X.columns)
    fig, ax = plt.subplots()
    df[feature].hist(bins=50, ax=ax)
    ax.set_title(feature)
    st.pyplot(fig)

with tab2:
    st.subheader("Model Performance")
    st.metric("AUC Score", f"{auc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Score Distribution**")
        scores = prob_to_score(y_pred)
        fig, ax = plt.subplots()
        ax.hist(scores, bins=50, edgecolor="black", color="steelblue")
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.write("**Feature Importance**")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        }).sort_values("Coefficient", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(data=importance, x="Coefficient", y="Feature", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.subheader("Credit Score Calculator")
    st.write("Enter applicant details to get a credit score.")

    col1, col2 = st.columns(2)
    with col1:
        utilization = st.slider("Revolving Utilization", 0.0, 1.0, 0.3)
        age = st.number_input("Age", 18, 100, 40)
        late_30_59 = st.number_input("Times 30-59 Days Late", 0, 20, 0)
        debt_ratio = st.number_input("Debt Ratio", 0.0, 10.0, 0.3)
        monthly_income = st.number_input("Monthly Income", 0, 100000, 5000)
    with col2:
        open_credit = st.number_input("Open Credit Lines", 0, 30, 5)
        late_90 = st.number_input("Times 90+ Days Late", 0, 20, 0)
        real_estate = st.number_input("Real Estate Loans", 0, 10, 1)
        late_60_89 = st.number_input("Times 60-89 Days Late", 0, 20, 0)
        dependents = st.number_input("Number of Dependents", 0, 10, 0)

    if st.button("Calculate Score"):
        input_data = np.array([[utilization, age, late_30_59, debt_ratio,
                                 monthly_income, open_credit, late_90,
                                 real_estate, late_60_89, dependents]])
        prob = model.predict_proba(input_data)[0][1]
        score = int(prob_to_score(prob))

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Credit Score", score)
        col2.metric("Default Probability", f"{prob:.1%}")

        if score >= 700:
            st.success("Low Risk")
        elif score >= 600:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")
