from typing import Any, Dict

import joblib
import pandas as pd

from loan_mlops.utils.paths import DEFAULT_MODEL_FILE

_default_model = None


def load_default_model() -> Any:
    """Load and cache the loan default model."""
    global _default_model
    if _default_model is None:
        _default_model = joblib.load(DEFAULT_MODEL_FILE)
    return _default_model

def predict_default_proba(applicant_features: Dict[str, Any]) -> float:
    """
    Predict probability of default for a single applicant.

    Parameters
    ----------
    applicant_features : dict
        Keys should match feature names used during training
        (both numeric and categorical).

    Returns
    -------
    float
        Probability of default (class 1).
    """
    model = load_default_model()

    # Convert to DataFrame with a single row
    X = pd.DataFrame([applicant_features])

    proba = model.predict_proba(X)[0, 1]
    return float(proba)


