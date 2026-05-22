"""
API Response Schemas
=====================
Pydantic models for FastAPI response serialization.

NeuralScoreBase contains:
  - Legacy dimension scores (for radar chart): visual, auditory, emotional, etc.
  - Model-based predictions (for CTR ranking): predicted_ctr, tier, confidence
"""

from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class NeuralScoreBase(BaseModel):
    """Neural analysis scores for a video.

    Two types of scores are included:
    - Legacy dimension scores (0-100, for radar chart breakdown)
    - Model-based CTR predictions (from trained XGBoost model)
    """

    # Model-based overall score: predicted_proba * 100
    # This is the PRIMARY metric for comparing videos.
    overall_score: Optional[float] = None

    # Legacy dimension scores (0-100 scale, for radar chart display)
    visual_score: Optional[float] = None
    auditory_score: Optional[float] = None
    emotional_score: Optional[float] = None
    attention_score: Optional[float] = None
    language_score: Optional[float] = None

    # Model-based CTR predictions
    predicted_ctr: Optional[float] = None           # Exact predicted CTR %
    predicted_class: Optional[str] = None           # "High" or "Low"
    predicted_proba: Optional[float] = None         # P(High CTR), 0.0–1.0
    prediction_tier: Optional[str] = None           # e.g. "Strong High"

    # Quantile confidence bounds (if available)
    ctr_lower_bound: Optional[float] = None         # 10th percentile CTR
    ctr_upper_bound: Optional[float] = None         # 90th percentile CTR


class VideoResponse(BaseModel):
    id: UUID
    filename: str
    original_name: str
    upload_time: datetime
    status: str
    job_id: Optional[str] = None
    scores: Optional[NeuralScoreBase] = None

    class Config:
        from_attributes = True


# ── Creative Insight Schemas ──────────────────────────────────────────────

class FeatureAnalysis(BaseModel):
    """Per-feature analysis from the LLM."""
    feature_name: str
    value: float
    rating: str           # "excellent", "good", "average", "poor"
    interpretation: str   # What this value means for this ad

class InsightStrength(BaseModel):
    """A strength identified in the ad's neural response."""
    title: str
    description: str
    impact: str           # "high", "medium", "low"

class InsightWeakness(BaseModel):
    """A weakness identified in the ad's neural response."""
    title: str
    description: str
    impact: str           # "high", "medium", "low"

class InsightRecommendation(BaseModel):
    """An actionable recommendation to improve the ad."""
    title: str
    description: str
    priority: str         # "high", "medium", "low"
    expected_impact: str  # Brief description of expected improvement

class CreativeInsightResponse(BaseModel):
    """Full structured creative insight response."""
    id: str
    video_id: str
    summary: str
    strengths: list[InsightStrength]
    weaknesses: list[InsightWeakness]
    recommendations: list[InsightRecommendation]
    feature_analysis: list[FeatureAnalysis]
    model_used: str
    generated_at: Optional[str] = None
