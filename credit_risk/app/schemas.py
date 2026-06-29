from typing import Literal, Optional

from pydantic import BaseModel


class NarrativeRequest(BaseModel):
    applicant_id: str
    force: bool = False


class ApplicantSummary(BaseModel):
    applicant_id: str
    risk_label: str
    default_probability: float
    is_thin_file: bool


class ApplicantListItem(BaseModel):
    applicant_id: str
    risk_label: str
    is_thin_file: bool


class ModelPerformance(BaseModel):
    regression_r2: float
    classification_accuracy: float
    auc_roc: float


class FairnessFlags(BaseModel):
    regression: bool
    classification: bool


class DashboardOverview(BaseModel):
    total_applicants: int
    risk_distribution: dict[str, int]
    thin_file_count: int
    thin_file_pct: float
    mean_default_probability: float
    model_performance: ModelPerformance
    fairness_flags: FairnessFlags


class NarrativeResponse(BaseModel):
    applicant_id: str
    narrative: str
    recommendation: Literal[
        "APPROVE", "APPROVE WITH CONDITIONS", "REFER FOR REVIEW", "DECLINE"
    ]
    generated_at: str
