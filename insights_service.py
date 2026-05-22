"""
Creative Insights Service
===========================
Orchestrates the generation of AI-powered creative insights.

Flow:
  1. Load NeuralScore data from the database
  2. Build the prompt using prompts.py
  3. Call Gemini via llm_client.py
  4. Validate and save structured insights to CreativeInsight model

This module is the only one that touches the database AND the LLM.
prompts.py and llm_client.py are pure functions / API wrappers.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from sqlalchemy.orm import Session

import models
from prompts import SYSTEM_PROMPT, format_analysis_data
from llm_client import GeminiClient, LLMError

logger = logging.getLogger("insights_service")

# Lazily initialized singleton — created on first call
_gemini_client: GeminiClient | None = None


def _get_client() -> GeminiClient:
    """Get or create the Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def generate_insights(video_id: UUID, db: Session) -> models.CreativeInsight:
    """Generate structured creative insights for a video.

    Args:
        video_id: UUID of the video to analyze.
        db: SQLAlchemy database session.

    Returns:
        The saved CreativeInsight model instance.

    Raises:
        ValueError: If the video is not found or not yet analyzed.
        LLMError: If the Gemini call fails after retries.
    """
    # ── Step 1: Load video and scores ──────────────────────────────────
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise ValueError(f"Video not found: {video_id}")

    if video.status != models.JobStatus.COMPLETED:
        raise ValueError(
            f"Video analysis not complete (status: {video.status.value}). "
            f"Insights can only be generated after analysis is finished."
        )

    scores = video.scores
    if not scores:
        raise ValueError(f"No neural scores found for video {video_id}")

    if scores.predicted_ctr is None:
        raise ValueError(
            f"Video {video_id} has no CTR prediction. "
            f"Re-run analysis with the updated pipeline."
        )

    # ── Step 2: Build prompt ───────────────────────────────────────────
    logger.info(f"[{video_id}] Building insights prompt...")

    # Reconstruct emotional_mean from the raw emotional score
    # (emotional_score was scaled: (raw + 0.5) * 100, so raw = emotional_score / 100 - 0.5)
    emotional_mean_raw = (scores.emotional_score / 100.0) - 0.5 if scores.emotional_score else 0.0

    feature_values = {
        "longest_sustained_above_mean": scores.longest_sustained_above_mean or 0.0,
        "emotional_mean": emotional_mean_raw,
        "orbital_mean": scores.orbital_mean or 0.0,
        "visual_std": scores.visual_std or 0.0,
        "insula_short_mean": scores.insula_short_mean or 0.0,
        "attention_onset_second": scores.attention_onset_second or 0.0,
    }

    dimension_scores = {
        "visual": scores.visual_score or 0.0,
        "auditory": scores.auditory_score or 0.0,
        "emotional": scores.emotional_score or 0.0,
        "attention": scores.attention_score or 0.0,
        "language": scores.language_score or 0.0,
    }

    user_message = format_analysis_data(
        video_name=video.original_name or video.filename,
        predicted_ctr=scores.predicted_ctr,
        predicted_class=scores.predicted_class or "Unknown",
        predicted_proba=scores.predicted_proba or 0.0,
        prediction_tier=scores.prediction_tier or "Unknown",
        overall_score=scores.overall_score or 0.0,
        feature_values=feature_values,
        ctr_lower_bound=scores.ctr_lower_bound,
        ctr_upper_bound=scores.ctr_upper_bound,
        dimension_scores=dimension_scores,
    )

    # ── Step 3: Call Gemini ─────────────────────────────────────────────
    logger.info(f"[{video_id}] Calling Gemini for insights...")
    client = _get_client()
    result = client.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.7,
    )

    # ── Step 4: Validate structure ─────────────────────────────────────
    _validate_insight_response(result)

    # ── Step 5: Save to database ───────────────────────────────────────
    logger.info(f"[{video_id}] Saving insights to database...")

    # Delete existing insight (overwrite policy)
    existing = db.query(models.CreativeInsight).filter(
        models.CreativeInsight.video_id == video_id
    ).first()
    if existing:
        db.delete(existing)
        db.flush()

    insight = models.CreativeInsight(
        video_id=video_id,
        summary=result.get("summary", ""),
        strengths=json.dumps(result.get("strengths", [])),
        weaknesses=json.dumps(result.get("weaknesses", [])),
        recommendations=json.dumps(result.get("recommendations", [])),
        feature_analysis=json.dumps(result.get("feature_analysis", [])),
        model_used=client._model,
    )
    db.add(insight)
    db.commit()
    db.refresh(insight)

    logger.info(f"[{video_id}] Insights saved successfully (id={insight.id})")
    return insight


def get_insights(video_id: UUID, db: Session) -> models.CreativeInsight | None:
    """Fetch existing insights for a video.

    Returns None if no insights have been generated yet.
    """
    return db.query(models.CreativeInsight).filter(
        models.CreativeInsight.video_id == video_id
    ).first()


def format_insight_response(insight: models.CreativeInsight) -> dict:
    """Convert a CreativeInsight DB model to an API-ready dict.

    Parses JSON strings back into Python objects for the response.
    """
    return {
        "id": str(insight.id),
        "video_id": str(insight.video_id),
        "summary": insight.summary,
        "strengths": json.loads(insight.strengths) if insight.strengths else [],
        "weaknesses": json.loads(insight.weaknesses) if insight.weaknesses else [],
        "recommendations": json.loads(insight.recommendations) if insight.recommendations else [],
        "feature_analysis": json.loads(insight.feature_analysis) if insight.feature_analysis else [],
        "model_used": insight.model_used,
        "generated_at": insight.generated_at.isoformat() if insight.generated_at else None,
    }


def _validate_insight_response(data: dict) -> None:
    """Validate the structure of the LLM response.

    Raises ValueError if required fields are missing.
    """
    required_keys = ["summary", "strengths", "weaknesses", "recommendations", "feature_analysis"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}")

    if not isinstance(data["strengths"], list):
        raise ValueError("'strengths' must be a list")
    if not isinstance(data["weaknesses"], list):
        raise ValueError("'weaknesses' must be a list")
    if not isinstance(data["recommendations"], list):
        raise ValueError("'recommendations' must be a list")
    if not isinstance(data["feature_analysis"], list):
        raise ValueError("'feature_analysis' must be a list")
