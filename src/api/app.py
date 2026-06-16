import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import pandas as pd
from src.policy.decision_engine import DecisionEngine

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize the decision engine
cate_map = config['data']['true_cate']
default_risk_increase = config.get('policy', {}).get('default_risk_increase', 0.02)
intervention_cost = config.get('policy', {}).get('intervention_cost', 50)
avg_loan_loss = config.get('policy', {}).get('avg_loan_loss', 500)

engine = DecisionEngine(
    cate_map=cate_map,
    intervention_cost=intervention_cost,
    avg_loan_loss=avg_loan_loss,
    default_risk_increase=default_risk_increase
)

app = FastAPI(
    title="Credit Limit Optimizer API",
    description="Causal decision engine for credit limit increases",
    version="1.0.0"
)

# Request/response models
class UserFeatures(BaseModel):
    income_quintile: str

class DecisionResponse(BaseModel):
    user_id: str
    income_quintile: str
    recommended_treatment: bool
    expected_niv: float
    cate: float
    default_loss_expected: float

@app.get("/")
def root():
    return {"message": "Credit Limit Optimizer API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/decide")
def decide(user: UserFeatures):
    """
    Get a decision for a single user based on their income quintile.
    """
    try:
        result = engine.recommend(user.income_quintile)
        return DecisionResponse(
            user_id="single_user",
            income_quintile=user.income_quintile,
            recommended_treatment=result['recommend_treatment'],
            expected_niv=result['expected_niv'],
            cate=result['cate'],
            default_loss_expected=result['default_loss_expected']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/decide_batch")
def decide_batch(users: list[UserFeatures]):
    """
    Get decisions for a batch of users.
    """
    try:
        df = pd.DataFrame([u.dict() for u in users])
        df_policy = engine.apply_policy(df)
        return df_policy[['income_quintile', 'recommended_treatment', 'expected_niv']].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))