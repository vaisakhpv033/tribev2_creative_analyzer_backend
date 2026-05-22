"""
API Routes
===========
FastAPI router for the Creative Quality Analyzer.

Endpoints:
  POST /api/v1/videos/upload     — Upload .mp4 or .npz file for analysis
  GET  /api/v1/videos/{id}/status — Check processing status
  GET  /api/v1/videos             — List all videos
  GET  /api/v1/videos/{id}/report — Full analysis report with scores + predictions
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from uuid import UUID
import shutil
import os
import schemas, models, database
from worker import process_video_task

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

STORAGE_DIR = os.getenv("STORAGE_DIR", "./uploads")
os.makedirs(STORAGE_DIR, exist_ok=True)


@router.post("/upload", response_model=schemas.VideoResponse)
def upload_file(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    is_npz = file.filename.endswith(".npz")
    is_mp4 = file.filename.endswith(".mp4")

    if not (is_mp4 or is_npz):
        raise HTTPException(status_code=400, detail="Only .mp4 or .npz files are supported.")

    db_video = models.Video(filename=file.filename, original_name=file.filename)
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    # Save file
    file_location = f"{STORAGE_DIR}/{db_video.id}_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    db_video.filename = f"{db_video.id}_{file.filename}"
    if is_npz:
        db_video.npz_path = file_location

    db.commit()

    # Trigger Celery task
    process_video_task.delay(str(db_video.id), file_location, is_npz)

    return db_video


@router.get("/{video_id}/status", response_model=schemas.VideoResponse)
def get_video_status(video_id: UUID, db: Session = Depends(database.get_db)):
    db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
    return db_video


@router.get("", response_model=list[schemas.VideoResponse])
def list_videos(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    videos = db.query(models.Video).order_by(
        models.Video.upload_time.desc()
    ).offset(skip).limit(limit).all()
    return videos


@router.get("/{video_id}/report")
def get_video_report(video_id: UUID, db: Session = Depends(database.get_db)):
    """Full analysis report for a processed video.

    Returns:
      - video: Basic metadata (id, filename)
      - scores: Legacy dimension scores (0-100) for radar chart
      - predictions: Model-based CTR predictions and confidence tier
      - model_features: The 6 brain features used by the XGBoost model
      - timeseries: Per-second activation curves for each brain dimension
      - global_mean: Per-second whole-brain average activation
    """
    db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")

    if db_video.status != models.JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video analysis is not yet completed")

    scores = db_video.scores

    # Re-extract timeseries from .npz for the response
    # (timeseries are not stored in DB — only computed on demand)
    timeseries_data = {}
    global_mean_data = []
    if db_video.npz_path and os.path.exists(db_video.npz_path):
        from analyzer import analyzer as brain_analyzer
        analysis_result = brain_analyzer.analyze(db_video.npz_path)
        timeseries_data = analysis_result.get('timeseries', {})
        global_mean_data = analysis_result.get('global_mean', [])

    # Build response
    response = {
        "video": {
            "id": db_video.id,
            "filename": db_video.filename,
            "original_name": db_video.original_name,
        },

        # Legacy dimension scores (for radar chart display)
        "scores": {
            "overall": scores.overall_score,
            "visual": scores.visual_score,
            "auditory": scores.auditory_score,
            "emotional": scores.emotional_score,
            "attention": scores.attention_score,
            "language": scores.language_score,
        },

        # Model-based CTR predictions (from trained XGBoost model)
        "predictions": {
            "predicted_ctr": scores.predicted_ctr,
            "predicted_class": scores.predicted_class,
            "confidence": scores.predicted_proba,
            "tier": scores.prediction_tier,
        },

        # Raw model feature values (the 6 features fed to XGBoost)
        "model_features": {
            "longest_sustained_above_mean": scores.longest_sustained_above_mean,
            "emotional_mean": scores.emotional_score,  # raw value before 0-100 scaling
            "orbital_mean": scores.orbital_mean,
            "visual_std": scores.visual_std,
            "insula_short_mean": scores.insula_short_mean,
            "attention_onset_second": scores.attention_onset_second,
        },

        # Time-series data (for charts)
        "timeseries": timeseries_data,
        "global_mean": global_mean_data,
    }

    # Add quantile confidence bounds if available
    if scores.ctr_lower_bound is not None:
        response["predictions"]["ctr_lower_bound"] = scores.ctr_lower_bound
        response["predictions"]["ctr_upper_bound"] = scores.ctr_upper_bound

    return response
