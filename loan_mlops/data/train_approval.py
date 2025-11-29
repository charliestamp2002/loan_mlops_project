from typing import Dict, Any

import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from loan_mlops.data.load_data import load_approval_data
from loan_mlops.data.preprocess import (
    build_preprocessing_pipeline,
    split_features_target)
from loan_mlops.utils.paths import APPROVAL_MODEL_FILE


def load_config(config_path: str = "loan_mlops/config/base_config.yaml") -> Dict[str, Any]:
    
    """Load YAML configuration file."""
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_baseline_approval() -> None: 
    "Train a baseline Logistic Regression model on loan approval data."


    config = load_config()
    data_cfg = config["data"]["approval"]
    model_cfg = config["model"]["approval"]

    df = load_approval_data()

    cols_needed = ( 
        data_cfg["numeric_features"] + data_cfg["categorical_features"] + [data_cfg["target_column"]]
    )

    df = df[cols_needed].copy()

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    X, y = split_features_target(df, data_cfg["target_column"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg["test_size"], random_state=model_cfg["random_state"], stratify=y
    )

    preprocessor = build_preprocessing_pipeline(
        numeric_features=data_cfg["numeric_features"],
        categorical_features=data_cfg["categorical_features"],
    )

    clf = LogisticRegression(max_iter = 1000)

    pipe = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    pipe.fit(X_train,y_train)

    joblib.dump(pipe, APPROVAL_MODEL_FILE)
    print(f"Saved baseline approval model to {APPROVAL_MODEL_FILE}")

    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    auc = roc_auc_score(y_test, y_prob)
    print(f"Baseline Logistic Regression ROC-AUC: {auc:.4f}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline Logistic Regression Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, pipe.predict(X_test)))

if __name__ == "__main__": 
    train_baseline_approval()





