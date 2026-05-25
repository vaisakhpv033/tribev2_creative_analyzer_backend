"""
Database Models
================
SQLAlchemy ORM models for the Creative Quality Analyzer.

NeuralScore stores two types of data:

1. LEGACY DIMENSION SCORES (visual_score, auditory_score, etc.)
   - Used for the radar-chart breakdown display
   - Scaled to 0-100 with a linear transform
   - Low correlation with real CTR (kept for backward compatibility)

2. MODEL-BASED PREDICTIONS (predicted_ctr, predicted_class, etc.)
   - Produced by the trained XGBoost CTR predictor
   - Uses 6 selected brain features with strong CTR correlation
   - overall_score = predicted_proba * 100 (actionable 0-100 metric)
"""

import uuid
import datetime
import enum
from sqlalchemy import Column, String, Float, Text, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROVISIONING_GPU = "PROVISIONING_GPU"
    BOOTING_GPU = "BOOTING_GPU"
    UPLOADING = "UPLOADING"
    INFERENCE = "INFERENCE"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, index=True)
    original_name = Column(String)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    job_id = Column(String, nullable=True)  # RunPod Job ID
    npz_path = Column(String, nullable=True)

    scores = relationship("NeuralScore", back_populates="video", uselist=False)
    insights = relationship("CreativeInsight", back_populates="video", uselist=False)


class NeuralScore(Base):
    """Neural analysis results for a video.

    Contains both legacy dimension scores and model-based CTR predictions.
    """
    __tablename__ = "neural_scores"

    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), primary_key=True)

    # ── Model-based overall score ─────────────────────────────────────────
    # overall_score = predicted_proba * 100
    # This is the PRIMARY actionable metric: "How confident are we this
    # ad will perform above average?" (0-100 scale, correlated with CTR)
    overall_score = Column(Float, nullable=True)

    # ── Legacy dimension scores (for radar chart) ─────────────────────────
    # These are kept for backward compatibility and dimension-level insights.
    # Computed as: (raw_brain_activation + 0.5) * 100, clamped to [0, 100].
    # NOTE: These individually have LOW correlation with real CTR.
    visual_score = Column(Float, nullable=True)
    auditory_score = Column(Float, nullable=True)
    emotional_score = Column(Float, nullable=True)
    attention_score = Column(Float, nullable=True)
    language_score = Column(Float, nullable=True)

    # ── Model-based CTR predictions ───────────────────────────────────────
    # Produced by the trained XGBoost model (6-feature input).
    # These are the actual CTR prediction outputs.
    predicted_ctr = Column(Float, nullable=True)           # Exact predicted CTR %
    predicted_class = Column(String, nullable=True)        # "High" or "Low"
    predicted_proba = Column(Float, nullable=True)         # P(High CTR), 0.0–1.0
    prediction_tier = Column(String, nullable=True)        # "Strong High", "Likely High", etc.

    # ── Quantile confidence bounds (optional) ─────────────────────────────
    # P10/P90 bounds from quantile regression models (if available).
    ctr_lower_bound = Column(Float, nullable=True)         # 10th percentile CTR
    ctr_upper_bound = Column(Float, nullable=True)         # 90th percentile CTR

    # ── Raw model feature values (audit trail) ────────────────────────────
    # The 6 brain features fed into the XGBoost model. Stored for debugging,
    # reproducibility, and potential re-training.
    longest_sustained_above_mean = Column(Float, nullable=True)
    orbital_mean = Column(Float, nullable=True)
    visual_std = Column(Float, nullable=True)
    insula_short_mean = Column(Float, nullable=True)
    attention_onset_second = Column(Float, nullable=True)
    # Note: emotional_mean is not stored separately because it's identical
    # to the raw value behind emotional_score (before 0-100 scaling).

    video = relationship("Video", back_populates="scores")


class BaselineCalibration(Base):
    """Legacy calibration table for 0-100 dimension score normalization."""
    __tablename__ = "baseline_calibration"

    dimension = Column(String, primary_key=True)
    min_value = Column(Float)
    max_value = Column(Float)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow,
                        onupdate=datetime.datetime.utcnow)


class CreativeInsight(Base):
    """AI-generated creative analysis for a video.

    Produced by calling Gemini LLM with the video's brain feature data.
    Structured output contains executive summary, strengths, weaknesses,
    recommendations, and per-feature analysis.

    Generated on-demand when the user clicks 'Generate Insights' on the
    report page. Overwritten on regeneration (one insight per video).
    """
    __tablename__ = "creative_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), index=True)

    # Structured insight sections
    summary = Column(Text, nullable=True)             # Executive summary
    strengths = Column(Text, nullable=True)            # JSON array of strength objects
    weaknesses = Column(Text, nullable=True)           # JSON array of weakness objects
    recommendations = Column(Text, nullable=True)      # JSON array of recommendation objects
    feature_analysis = Column(Text, nullable=True)     # JSON array of per-feature analysis

    # Metadata
    model_used = Column(String, default="gemini-2.5-flash")
    generated_at = Column(DateTime, default=datetime.datetime.utcnow)

    video = relationship("Video", back_populates="insights")
