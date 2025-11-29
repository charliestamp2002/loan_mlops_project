from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from loan_mlops.decision.engine import make_decision, DecisionResult

app = FastAPI(title="Loan Decision Service")

class ApprovalFeatures(BaseModel):
    Age: int
    AnnualIncome: float
    CreditScore: int
    EmploymentStatus: str
    MaritalStatus: str
    EducationLevel: str

class DefaultFeatures(BaseModel):
    Age: int
    LoanAmount: float
    CreditScore: int
    Education: str
    EmploymentType: str
    MaritalStatus: str

class DecisionResponse(BaseModel):
    decision: str
    approval_proba: float
    default_proba: float
    reasons: Dict[str, Any]

@app.get("/health")
def health(): 
    return {"status": "ok"}

@app.post("/score_application", response_model=DecisionResponse)
def score_application(approval: ApprovalFeatures, default: DefaultFeatures):
    """
    Takes two feature blocks:
    - approval: features expected by approval model
    - default:  features expected by default model
    """

    result: DecisionResult = make_decision(
        app_features_approval=approval.dict(),
        app_features_default=default.dict(),
    )

    return DecisionResponse(
        decision=result.decision,
        approval_proba=result.approval_proba,
        default_proba=result.default_proba,
        reasons=result.reasons,
    )