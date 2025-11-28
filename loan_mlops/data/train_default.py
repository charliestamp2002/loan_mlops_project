from typing import Dict, Any

import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from loan_mlops.data.load_data import load_default_data
from loan_mlops.data.preprocess import (
    build_preprocessing_pipeline,
    split_features_target)

def load_config(config_path: str = "loan_mlops/config/base_config.yaml") -> Dict[str, Any]:
    
    """Load YAML configuration file."""
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def process_pipeline(type_of_pipeline: str, model_cfg: Dict[str, Any],
                     X_train: pd.DataFrame, y_train: pd.Series, 
                     preprocessor: Any, clf: Any) -> ImbPipeline: 
    if type_of_pipeline == "smote": 
        smote = SMOTE(
            sampling_strategy = "auto",
            random_state=model_cfg["random_state"]
            )
        
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", smote),               
                ("classifier", clf),
            ]
        )
    else: 
        pipe = ImbPipeline(
            steps = [
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
    return pipe


def train_baseline_default() -> None: 
    "Train a baseline Logistic Regression model on loan default data."


    config = load_config()
    data_cfg = config["data"]["default"]
    model_cfg = config["model"]["default"]

    df = load_default_data()

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

    print("Class distribution in training data (Before SMOTE):")
    print(y_train.value_counts())
    print("\nClass proportions (Before SMOTE):")
    print(y_train.value_counts(normalize=True))

    preprocessor = build_preprocessing_pipeline(
        numeric_features=data_cfg["numeric_features"],
        categorical_features=data_cfg["categorical_features"],
    )

    clf = LogisticRegression(max_iter = 1000,
                             class_weight='balanced'
            )
    
    pipe = process_pipeline(
        type_of_pipeline = "smote",
        model_cfg = model_cfg,
        X_train = X_train,
        y_train = y_train,
        preprocessor = preprocessor,
        clf = clf
    )

    # print("Class distribution in training data (After SMOTE):")
    # print(y_train.value_counts())
    # print("\nClass proportions (After SMOTE):")
    # print(y_train.value_counts(normalize=True))
    
    # pipe = Pipeline(
    #     steps = [
    #         ("preprocessor", preprocessor),
    #         ("classifier", clf),
    #     ]
    # )

    pipe.fit(X_train,y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Baseline Logistic Regression ROC-AUC: {auc:.4f}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline Logistic Regression Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))



if __name__ == "__main__": 
    train_baseline_default()
