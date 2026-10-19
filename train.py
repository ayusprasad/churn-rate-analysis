import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_churn_model(csv_path="telco.csv", output_path="churn_model.pkl"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist.")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean TotalCharges (spaces to NaN, then numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Map target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    drop_cols = ["customerID", "Churn"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    print("Training Random Forest Churn Classifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Model ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    model_payload = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "roc_auc": roc_auc,
    }

    joblib.dump(model_payload, output_path)
    print(f"Saved trained pipeline to {output_path} ({os.path.getsize(output_path)/1024:.1f} KB).")
    return model_payload


if __name__ == "__main__":
    train_churn_model()
