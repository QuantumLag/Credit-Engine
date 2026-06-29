from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    actual_value: float
    human_readable: str
    direction: Literal["increases_risk", "decreases_risk"]
    magnitude: Literal["high", "medium", "low"]


class ThinFileExplanation(BaseModel):
    alt_credit_score: float
    thin_file_confidence: float
    alt_score_shap: float
    note: str


class Counterfactual(BaseModel):
    action: str
    feature: Optional[str] = None
    current_value_human: Optional[str] = None
    suggested_value_human: Optional[str] = None
    estimated_probability_reduction: Optional[float] = None
    new_predicted_probability: Optional[float] = None
    new_risk_label: Optional[Literal["Low", "Medium", "High"]] = None
    feasibility: Optional[Literal["easy", "medium", "hard"]] = None
    note: Optional[str] = None


class ApplicantExplanation(BaseModel):
    applicant_id: str
    is_thin_file: bool
    predicted_default_probability: float
    predicted_risk_label: Literal["Low", "Medium", "High"]
    class_probabilities: dict[str, float]
    baseline_default_probability: float
    top_risk_drivers: list[FeatureContribution] = Field(min_length=5, max_length=5)
    top_risk_mitigants: list[FeatureContribution] = Field(min_length=5, max_length=5)
    thin_file_explanation: Optional[ThinFileExplanation]
    feature_group_contributions: dict[str, float]
    counterfactuals: Optional[list[Counterfactual]] = None
