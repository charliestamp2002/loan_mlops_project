
from typing import Any, Dict

import joblib
import pandas as pd

from loan_mlops.utils.paths import APPROVAL_MODEL_FILE

_approval_model = None


def load_approval_model() -> Any:
    """Load and cache the loan approval model."""
    global _approval_model
    if _approval_model is None:
        _approval_model = joblib.load(APPROVAL_MODEL_FILE)
    return _approval_model

def predict_approval_proba(applicant_features: Dict[str, Any]) -> float:
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
    model = load_approval_model()

    # Convert to DataFrame with a single row
    X = pd.DataFrame([applicant_features])

    proba = model.predict_proba(X)[0, 1]
    return float(proba)