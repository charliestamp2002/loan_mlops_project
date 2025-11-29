from dataclasses import dataclass
from typing import Any, Dict, Literal

import yaml

from loan_mlops.data.approval_inference import predict_approval_proba
from loan_mlops.data.default_inference import predict_default_proba

DecisionType = Literal["APPROVE", "REJECT", "MANUAL_REVIEW"]

@dataclass
class DecisionResult:
    decision: DecisionType
    approval_proba: float
    default_proba: float
    reasons: Dict[str, Any]

def load_decision_config(path: str = "loan_mlops/config/base_config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["decision"]

def make_decision(app_features_approval: Dict[str, Any],
                  app_features_default: Dict[str, Any]) -> DecisionResult:
    
    decision_cfg = load_decision_config()
    approval_p = predict_approval_proba(app_features_approval)
    default_p = predict_default_proba(app_features_default)

    approval_thresh = decision_cfg["approval_threshold"]
    high_default_thresh = decision_cfg["high_risk_default_threshold"]
    low_default_thresh = decision_cfg["low_risk_default_threshold"]

    if approval_p < approval_thresh:
        decision: DecisionType = "REJECT"
        reason = "Low approval probability"
    elif default_p >= high_default_thresh:
        decision = "REJECT"
        reason = "High default risk"
    elif default_p <= low_default_thresh and approval_p >= approval_thresh:
        decision = "APPROVE"
        reason = "Low default risk and sufficient approval probability"
    else: 
        decision = "MANUAL_REVIEW"
        reason = "Uncertain Risk Profile"
    
    reasons = {
        "rule_reason": reason,
        "approval_threshold": approval_thresh,
        "high_risk_default_threshold": high_default_thresh,
        "low_risk_default_threshold": low_default_thresh,
        }

    return DecisionResult( 
        decision=decision,
        approval_proba=approval_p,
        default_proba=default_p,
        reasons=reasons,
    )

    




