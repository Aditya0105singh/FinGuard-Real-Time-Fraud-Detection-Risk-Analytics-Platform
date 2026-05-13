"""Pydantic request/response models for the FinGuard API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionInput(BaseModel):
    """A single transaction to score."""

    amount: float = Field(..., ge=0, description="Transaction amount (EUR)",
                          examples=[149.62])
    transaction_hour: int = Field(..., ge=0, le=23,
                                  description="Hour of day (0-23)", examples=[2])
    is_night: int = Field(..., ge=0, le=1,
                          description="1 if the transaction occurred between 00:00-06:59")
    amount_zscore: float = Field(..., description="Z-score normalised amount",
                                 examples=[0.24])
    v1_to_v10: list[float] = Field(
        ...,
        description="Top-10 PCA components (V1..V10) from the upstream pipeline",
        examples=[[-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09]],
    )

    @field_validator("v1_to_v10")
    @classmethod
    def must_have_ten_components(cls, v: list[float]) -> list[float]:
        if len(v) != 10:
            raise ValueError(f"v1_to_v10 must contain exactly 10 values, got {len(v)}")
        return v


class FeatureContribution(BaseModel):
    """One feature's SHAP contribution to a prediction (log-odds space)."""

    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionOutput(BaseModel):
    """Fraud score and recommended action for a transaction."""

    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(..., ge=0, le=1)
    recommendation: str
    top_factors: list[FeatureContribution] | None = Field(
        None,
        description="Top SHAP factors driving this score (single predictions only)",
    )


class BatchPredictionInput(BaseModel):
    """A batch of transactions to score in one call."""

    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=1000)


class BatchPredictionOutput(BaseModel):
    count: int
    predictions: list[PredictionOutput]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str


class StatsResponse(BaseModel):
    """Model performance metrics from the last training run."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    average_precision: float
    optimal_threshold: float | None = Field(
        None, description="Cost-optimal decision threshold (if optimizer has run)"
    )
