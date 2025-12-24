"""
FastAPI Application for Financial Risk Prediction
Aligned with actual dataset: Age, Income, Credit Score, ... Risk Rating
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Financial Risk Prediction API",
    description="Predict financial risk (Low/Medium/High) based on customer and loan features.",
    version="1.0",
)

# Model paths
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# Global model objects
model = None
label_encoder = None
scaler = None
feature_cols = None


class LoanApplication(BaseModel):
    """Input schema for loan risk prediction."""

    age: int = Field(..., ge=18, le=100, description="Applicant's age")
    gender: str = Field(..., description="Gender (Male/Female)")
    education_level: str = Field(..., description="Education level (e.g., Bachelor's, PhD)")
    marital_status: str = Field(..., description="Marital status (e.g., Single, Married)")
    income: float = Field(..., ge=0, description="Annual income")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of loan (e.g., Auto, Business)")
    employment_status: str = Field(
        ..., description="Employment status (e.g., Employed, Unemployed)"
    )
    years_at_current_job: float = Field(..., ge=0, description="Years at current job")
    payment_history: str = Field(..., description="Payment history (Good/Fair/Poor)")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    assets_value: float = Field(..., ge=0, description="Total assets value")
    number_of_dependents: int = Field(..., ge=0, description="Number of dependents")
    previous_defaults: float = Field(..., ge=0, description="Number of previous defaults")
    marital_status_change: int = Field(0, ge=0, description="Marital status change indicator")

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "gender": "Male",
                "education_level": "Bachelor's",
                "marital_status": "Married",
                "income": 75000.0,
                "credit_score": 720.0,
                "loan_amount": 25000.0,
                "loan_purpose": "Auto",
                "employment_status": "Employed",
                "years_at_current_job": 5.0,
                "payment_history": "Good",
                "debt_to_income_ratio": 0.25,
                "assets_value": 150000.0,
                "number_of_dependents": 2,
                "previous_defaults": 0.0,
                "marital_status_change": 0,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for predictions."""

    risk_class: str
    risk_probabilities: dict
    confidence: float


@app.on_event("startup")
def load_model():
    """Load model and preprocessing artifacts on startup."""
    global model, label_encoder, scaler, feature_cols

    print("Loading model artifacts...")

    try:
        if os.path.exists(MODEL_PATH):
            model = xgb.Booster()
            model.load_model(MODEL_PATH)
            print(f"  - Loaded model from {MODEL_PATH}")
        else:
            print(f"  - WARNING: Model file not found at {MODEL_PATH}")

        if os.path.exists(LABEL_ENCODER_PATH):
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            print("  - Loaded label encoder")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("  - Loaded scaler")

        if os.path.exists(FEATURE_COLS_PATH):
            feature_cols = joblib.load(FEATURE_COLS_PATH)
            print(f"  - Loaded feature columns ({len(feature_cols)} features)")

    except Exception as e:
        print(f"Error loading model: {e}")


def prepare_features(application: LoanApplication) -> pd.DataFrame:
    """Convert input to feature DataFrame matching training format."""
    # Create base DataFrame
    data = {
        "age": [application.age],
        "income": [application.income],
        "credit_score": [application.credit_score],
        "loan_amount": [application.loan_amount],
        "years_at_current_job": [application.years_at_current_job],
        "debt_to_income_ratio": [application.debt_to_income_ratio],
        "assets_value": [application.assets_value],
        "number_of_dependents": [application.number_of_dependents],
        "previous_defaults": [application.previous_defaults],
        "marital_status_change": [application.marital_status_change],
        "gender": [application.gender],
        "education_level": [application.education_level],
        "marital_status": [application.marital_status],
        "loan_purpose": [application.loan_purpose],
        "employment_status": [application.employment_status],
        "payment_history": [application.payment_history],
    }

    df = pd.DataFrame(data)

    # === CREATE DOMAIN-SPECIFIC FEATURES (must match training) ===
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)
    df["asset_coverage_ratio"] = df["assets_value"] / (df["loan_amount"] + 1)
    df["net_worth_estimate"] = df["assets_value"] - df["loan_amount"]
    df["high_dti_flag"] = (df["debt_to_income_ratio"] > 0.43).astype(int)
    df["poor_credit_flag"] = (df["credit_score"] < 580).astype(int)
    df["fair_credit_flag"] = ((df["credit_score"] >= 580) & (df["credit_score"] < 670)).astype(int)
    df["good_credit_flag"] = (df["credit_score"] >= 670).astype(int)
    df["has_previous_defaults"] = (df["previous_defaults"] > 0).astype(int)
    df["job_stability"] = df["years_at_current_job"] / (df["age"] - 18 + 1)
    df["income_per_dependent"] = df["income"] / (df["number_of_dependents"] + 1)
    df["credit_score_normalized"] = (df["credit_score"] - 300) / 550
    df["affordability_index"] = (df["income"] - df["loan_amount"] * 0.05) / (df["income"] + 1)
    df["credit_income_interaction"] = df["credit_score_normalized"] * np.log1p(df["income"])
    df["risk_composite"] = (
        df["high_dti_flag"] + df["poor_credit_flag"] + df["has_previous_defaults"]
    )
    df["young_borrower"] = (df["age"] < 25).astype(int)
    df["senior_borrower"] = (df["age"] > 55).astype(int)

    # One-hot encode categoricals
    categorical_features = [
        "gender",
        "education_level",
        "marital_status",
        "loan_purpose",
        "employment_status",
        "payment_history",
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Ensure all training features exist (add missing with 0)
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder to match training
    df_encoded = df_encoded[feature_cols]

    # Scale numerical features (including domain features)
    numerical_features = [
        "age",
        "income",
        "credit_score",
        "loan_amount",
        "years_at_current_job",
        "debt_to_income_ratio",
        "assets_value",
        "number_of_dependents",
        "previous_defaults",
        "marital_status_change",
        "loan_to_income_ratio",
        "asset_coverage_ratio",
        "net_worth_estimate",
        "high_dti_flag",
        "poor_credit_flag",
        "fair_credit_flag",
        "good_credit_flag",
        "has_previous_defaults",
        "job_stability",
        "income_per_dependent",
        "credit_score_normalized",
        "affordability_index",
        "credit_income_interaction",
        "risk_composite",
        "young_borrower",
        "senior_borrower",
    ]

    if scaler is not None:
        cols_to_scale = [c for c in numerical_features if c in df_encoded.columns]
        df_encoded[cols_to_scale] = scaler.transform(df_encoded[cols_to_scale])

    return df_encoded


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(application: LoanApplication):
    """Predict financial risk for a loan application."""
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    # Prepare features
    try:
        features = prepare_features(application)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature preparation error: {str(e)}")

    # Create DMatrix
    dmatrix = xgb.DMatrix(features, feature_names=feature_cols)

    # Predict
    probabilities = model.predict(dmatrix)[0]
    predicted_class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class_idx])

    # Decode class
    if label_encoder is not None:
        risk_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        class_names = list(label_encoder.classes_)
    else:
        risk_class = ["Low", "Medium", "High"][predicted_class_idx]
        class_names = ["Low", "Medium", "High"]

    prob_dict = {name: float(p) for name, p in zip(class_names, probabilities)}

    return PredictionResponse(
        risk_class=risk_class, risk_probabilities=prob_dict, confidence=confidence
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features_count": len(feature_cols) if feature_cols else 0,
    }


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Financial Risk Prediction API", "docs": "/docs", "health": "/health"}
