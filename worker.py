import os
import time
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
TRIBEV2_API_BASE_URL = os.getenv("TRIBEV2_API_BASE_URL", "http://localhost:8000") # Replace with runpod URL

celery_app = Celery(
    "creative_quality_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

import requests
from database import SessionLocal
import models
from analyzer import analyzer as brain_analyzer

@celery_app.task(name="process_video_task", bind=True, max_retries=3)
def process_video_task(self, video_id: str, video_path: str, is_npz: bool = False):
    db = SessionLocal()
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        db.close()
        return {"error": "Video not found"}
        
    try:
        if not is_npz:
            # 1. Submit to TRIBEv2
            video.status = models.JobStatus.INFERENCE
            db.commit()
            
            url = f"{TRIBEV2_API_BASE_URL}/api/v1/jobs/analyze"
            with open(video_path, "rb") as f:
                resp = requests.post(url, files={"video": (os.path.basename(video_path), f, "video/mp4")}, timeout=300)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            
            video.job_id = job_id
            db.commit()

            # 2. Poll for completion
            status_url = f"{TRIBEV2_API_BASE_URL}/api/v1/jobs/{job_id}/status"
            while True:
                status_resp = requests.get(status_url, timeout=30)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                api_status = status_data.get("status", "UNKNOWN")
                
                if api_status == "COMPLETED":
                    break
                elif api_status in ["FAILED", "DELETED"]:
                    raise Exception(f"TRIBEv2 API returned terminal status: {api_status}")
                
                time.sleep(10) # Poll every 10 seconds
                
            # 3. Download .npz
            result_url = f"{TRIBEV2_API_BASE_URL}/api/v1/jobs/{job_id}/result"
            npz_dest = f"{os.path.splitext(video_path)[0]}.npz"
            with requests.get(result_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(npz_dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            video.npz_path = npz_dest
        else:
            npz_dest = video_path

        video.status = models.JobStatus.ANALYZING
        db.commit()

        # 4. Extract features
        analysis_result = brain_analyzer.analyze(npz_dest)
        raw_scores = analysis_result['raw_scores']
        
        # 5. Calculate scores (Normalization placeholder)
        # Ideally, we would fetch min/max from models.BaselineCalibration.
        # For this MVP, we map the raw scores directly to a dummy 0-100 scale.
        def scale(val):
            # very rough estimate scaling
            scaled = (val + 0.5) * 100 
            return max(0, min(100, scaled))
            
        visual_score = scale(raw_scores['visual'])
        auditory_score = scale(raw_scores['auditory'])
        emotional_score = scale(raw_scores['emotional'])
        language_score = scale(raw_scores['language'])
        attention_score = min(100, max(0, raw_scores['attention'] * 50)) # ratio scaling
        
        overall_score = (visual_score * 0.25) + (auditory_score * 0.15) + \
                        (emotional_score * 0.30) + (attention_score * 0.20) + \
                        (language_score * 0.10)

        # 6. Update DB
        score_record = models.NeuralScore(
            video_id=video_id,
            overall_score=overall_score,
            visual_score=visual_score,
            auditory_score=auditory_score,
            emotional_score=emotional_score,
            attention_score=attention_score,
            language_score=language_score
        )
        db.add(score_record)
        video.status = models.JobStatus.COMPLETED
        db.commit()
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        video.status = models.JobStatus.FAILED
        db.commit()
    finally:
        db.close()
        
    return {"status": "finished", "video_id": video_id}
