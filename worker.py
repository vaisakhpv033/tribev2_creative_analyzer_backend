import os
import time
import requests
from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

from database import SessionLocal
import models
from analyzer import analyzer as brain_analyzer

load_dotenv()

# Setup Celery logger
logger = get_task_logger(__name__)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
TRIBEV2_API_BASE_URL = os.getenv("TRIBEV2_API_BASE_URL", "http://localhost:8000")

celery_app = Celery(
    "creative_quality_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Production-ready Celery broker and task resiliency settings
celery_app.conf.update(
    # Re-establish connections on startup if broker is down
    broker_connection_retry_on_startup=True,
    
    # Retry connecting to broker if the connection is lost (retry indefinitely)
    broker_connection_max_retries=None,
    
    # Task execution safety settings
    task_acks_late=True,                 # Acknowledge task after execution so it isn't lost if worker dies
    task_reject_on_worker_lost=True,     # Requeue task if worker crashes during execution
    
    # Task time limits to prevent hung processes
    task_time_limit=3600,                # 1 hour hard limit
    task_soft_time_limit=3300,           # 55 minutes soft limit (allows clean shutdown/retries)
    
    # Prefetch limits
    worker_prefetch_multiplier=1,        # Only fetch 1 task at a time per worker process
)

class PermanentTaskFailure(Exception):
    """Exception raised for errors that cannot be resolved by retrying the task."""
    pass

def _update_video_status(video_id: str, status: models.JobStatus, job_id: str = None, npz_path: str = None):
    """Safely updates a video's status, job_id, and npz_path in the database with strict session management."""
    db = SessionLocal()
    try:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video:
            video.status = status
            if job_id is not None:
                video.job_id = job_id
            if npz_path is not None:
                video.npz_path = npz_path
            db.commit()
            logger.info(f"[{video_id}] Database status updated to {status.value} (job_id: {job_id}, npz_path: {npz_path})")
        else:
            logger.warning(f"[{video_id}] Database update requested, but video record not found.")
    except Exception as e:
        logger.error(f"[{video_id}] Database update failed: {e}")
        db.rollback()
    finally:
        db.close()

@celery_app.task(name="process_video_task", bind=True, max_retries=5)
def process_video_task(self, video_id: str, video_path: str, is_npz: bool = False):
    current_attempt = self.request.retries + 1
    logger.info(f"[{video_id}] Starting task. Attempt {current_attempt}/{self.max_retries + 1}")

    # Step 0: Pre-flight checks
    if not os.path.exists(video_path):
        error_msg = f"Video file not found at path: {video_path}"
        logger.error(f"[{video_id}] {error_msg}")
        _update_video_status(video_id, models.JobStatus.FAILED)
        return {"error": error_msg}

    db = SessionLocal()
    try:
        # Fetch the video record
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            logger.error(f"[{video_id}] Video record not found in database.")
            db.close()
            return {"error": "Video not found"}

        if not is_npz:
            job_id = video.job_id
            
            # Step 1: Submit to TRIBEv2 (if not already submitted in a previous attempt)
            if not job_id:
                logger.info(f"[{video_id}] Submitting video to TRIBEv2...")
                video.status = models.JobStatus.INFERENCE
                db.commit()

                url = f"{TRIBEV2_API_BASE_URL}/api/v1/jobs/analyze"
                try:
                    with open(video_path, "rb") as f:
                        resp = requests.post(
                            url, 
                            files={"video": (os.path.basename(video_path), f, "video/mp4")}, 
                            timeout=300
                        )
                    resp.raise_for_status()
                except requests.HTTPError as he:
                    status_code = he.response.status_code if he.response is not None else 500
                    # For client-side mistakes (4xx errors, excluding 429), fail permanently
                    if status_code in [400, 401, 403, 415, 422]:
                        raise PermanentTaskFailure(f"TRIBEv2 API submission failed with client error {status_code}: {he}")
                    raise
                
                job_id = resp.json()["job_id"]
                video.job_id = job_id
                db.commit()
                logger.info(f"[{video_id}] Video successfully submitted. Job ID assigned: {job_id}")
            else:
                logger.info(f"[{video_id}] Found existing Job ID: {job_id}. Skipping submission phase, resuming polling.")

            # Step 2: Poll for completion
            status_url = f"{TRIBEV2_API_BASE_URL}/api/v1/jobs/{job_id}/status"
            logger.info(f"[{video_id}] Polling TRIBEv2 job status for {job_id}...")

            poll_count = 0
            max_polls = 120  # 120 * 15 seconds = 30 minutes polling timeout limit
            while True:
                try:
                    status_resp = requests.get(status_url, timeout=30)
                    # Self-healing: if remote job expired or server reset, trigger re-submission by resetting job_id
                    if status_resp.status_code == 404:
                        logger.warning(f"[{video_id}] Job {job_id} not found on TRIBEv2 server (HTTP 404). Resetting job_id to trigger re-submission.")
                        video.job_id = None
                        db.commit()
                        raise Exception(f"TRIBEv2 Job {job_id} not found on server (404). job_id reset. Retrying task.")
                    
                    status_resp.raise_for_status()
                except requests.HTTPError as he:
                    status_code = he.response.status_code if he.response is not None else 500
                    if status_code in [400, 401, 403, 422]:
                        raise PermanentTaskFailure(f"TRIBEv2 API status check returned permanent client error {status_code}: {he}")
                    raise

                status_data = status_resp.json()
                api_status = status_data.get("status", "UNKNOWN")
                logger.info(f"[{video_id}] Polled status: {api_status}")

                if api_status == "COMPLETED":
                    break
                elif api_status in ["FAILED", "DELETED"]:
                    raise PermanentTaskFailure(f"TRIBEv2 API returned terminal failure status: {api_status}")

                poll_count += 1
                if poll_count >= max_polls:
                    raise Exception(f"TRIBEv2 polling timed out after 30 minutes for Job {job_id}")

                time.sleep(15)

            # Step 3: Download .npz atomically to avoid partial corrupt files
            npz_dest = f"{os.path.splitext(video_path)[0]}.npz"
            npz_tmp = f"{npz_dest}.tmp"
            result_url = f"{TRIBEV2_API_BASE_URL}/api/v1/jobs/{job_id}/result"
            
            logger.info(f"[{video_id}] Downloading .npz result atomically to {npz_dest}...")
            try:
                with requests.get(result_url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(npz_tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                os.replace(npz_tmp, npz_dest)
            except Exception as de:
                # Cleanup temp file on failure
                if os.path.exists(npz_tmp):
                    try:
                        os.remove(npz_tmp)
                    except OSError:
                        pass
                raise de

            video.npz_path = npz_dest
            db.commit()
        else:
            npz_dest = video_path

        # Step 4: Extract features
        logger.info(f"[{video_id}] Analyzing neural features from {npz_dest}...")
        video.status = models.JobStatus.ANALYZING
        db.commit()

        analysis_result = brain_analyzer.analyze(npz_dest)
        raw_scores = analysis_result['raw_scores']

        # Step 5: Calculate scores (Normalization)
        def scale(val):
            scaled = (val + 0.5) * 100
            return max(0, min(100, scaled))

        visual_score = scale(raw_scores['visual'])
        auditory_score = scale(raw_scores['auditory'])
        emotional_score = scale(raw_scores['emotional'])
        language_score = scale(raw_scores['language'])
        attention_score = min(100, max(0, raw_scores['attention'] * 50))

        overall_score = (visual_score * 0.25) + (auditory_score * 0.15) + \
                        (emotional_score * 0.30) + (attention_score * 0.20) + \
                        (language_score * 0.10)

        # Step 6: Upsert database neural scores (avoid key duplicates)
        score_record = db.query(models.NeuralScore).filter(models.NeuralScore.video_id == video_id).first()
        if score_record:
            logger.info(f"[{video_id}] Updating existing neural score record.")
            score_record.overall_score = overall_score
            score_record.visual_score = visual_score
            score_record.auditory_score = auditory_score
            score_record.emotional_score = emotional_score
            score_record.attention_score = attention_score
            score_record.language_score = language_score
        else:
            logger.info(f"[{video_id}] Creating new neural score record.")
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
        logger.info(f"[{video_id}] Task completed successfully.")

    except PermanentTaskFailure as ptf:
        logger.error(f"[{video_id}] Permanent task failure encountered: {ptf}")
        try:
            db.rollback()
            # Reload video within database session context to update status to FAILED
            video = db.query(models.Video).filter(models.Video.id == video_id).first()
            if video:
                video.status = models.JobStatus.FAILED
                db.commit()
        except Exception as dbe:
            logger.error(f"[{video_id}] Failed to set video status to FAILED on permanent failure: {dbe}")
            db.rollback()
        raise ptf

    except Exception as exc:
        retries = self.request.retries
        max_retries = self.max_retries
        countdown = (2 ** retries) * 30

        if retries >= max_retries:
            logger.error(f"[{video_id}] Max retries ({max_retries}) reached. Failing task. Last error: {exc}")
            try:
                db.rollback()
                video = db.query(models.Video).filter(models.Video.id == video_id).first()
                if video:
                    video.status = models.JobStatus.FAILED
                    db.commit()
            except Exception as dbe:
                logger.error(f"[{video_id}] Failed to set video status to FAILED on max retries limit: {dbe}")
                db.rollback()
            raise exc
        else:
            logger.warning(f"[{video_id}] Transient error: {exc}. Retrying task in {countdown}s (Attempt {retries + 1}/{max_retries}).")
            db.rollback()
            db.close() # Avoid connection leaks before retrying
            raise self.retry(exc=exc, countdown=countdown)

    finally:
        db.close()

    return {"status": "finished", "video_id": video_id}
