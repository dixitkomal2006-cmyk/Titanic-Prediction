import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("🚢 Titanic Survival Prediction")

file = st.file_uploader("Upload Titanic CSV", type="csv")

if file is not None:

    df = pd.read_csv(file)

    # ---------------- Data Preprocessing ----------------
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare",
        "Sex_male", "Embarked_Q", "Embarked_S"
    ]

    X = df[features]
    y = df["Survived"]

    # ---------------- Train/Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------- Model ----------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ---------------- Accuracy ----------------
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: **{accuracy:.2f}**")

    # ---------------- Confusion Matrix ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted No", "Predicted Yes"],
        index=["Actual No", "Actual Yes"]
    )

    st.dataframe(cm_df)

    # ---------------- Survival Distribution ----------------
    st.subheader("Survival Distribution")

    st.bar_chart(df["Survived"].value_counts())
