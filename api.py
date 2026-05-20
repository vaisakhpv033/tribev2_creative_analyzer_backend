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
def upload_video(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
        
    db_video = models.Video(filename=file.filename, original_name=file.filename)
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    # Save file
    file_location = f"{STORAGE_DIR}/{db_video.id}_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    db_video.filename = f"{db_video.id}_{file.filename}"
    db.commit()

    # Trigger Celery task
    process_video_task.delay(str(db_video.id), file_location)

    return db_video

@router.get("/{video_id}/status", response_model=schemas.VideoResponse)
def get_video_status(video_id: UUID, db: Session = Depends(database.get_db)):
    db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
    return db_video

@router.get("", response_model=list[schemas.VideoResponse])
def list_videos(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    videos = db.query(models.Video).order_by(models.Video.upload_time.desc()).offset(skip).limit(limit).all()
    return videos

@router.get("/{video_id}/report")
def get_video_report(video_id: UUID, db: Session = Depends(database.get_db)):
    db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    if db_video.status != models.JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video analysis is not yet completed")
        
    scores = db_video.scores
    
    # Load timeseries from .npz
    timeseries_data = {}
    if db_video.npz_path and os.path.exists(db_video.npz_path):
        from analyzer import analyzer as brain_analyzer
        analysis_result = brain_analyzer.analyze(db_video.npz_path)
        timeseries_data = analysis_result.get('timeseries', {})
        
    return {
        "video": {
            "id": db_video.id,
            "filename": db_video.filename,
            "original_name": db_video.original_name,
        },
        "scores": {
            "overall": scores.overall_score,
            "visual": scores.visual_score,
            "auditory": scores.auditory_score,
            "emotional": scores.emotional_score,
            "attention": scores.attention_score,
            "language": scores.language_score,
        },
        "timeseries": timeseries_data
    }
