from typing import Tuple, List, Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline(
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> ColumnTransformer:
    """
    Build a Column Transformer which scales 
    numeric features and one-hot encodes 
    categorical features.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", 
             OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                )
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into (X, y) given target column name.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

