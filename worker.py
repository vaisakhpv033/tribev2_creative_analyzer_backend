"""
Creative Quality Worker — Celery Task
=======================================
Orchestrates the full video analysis pipeline:

1. Upload video to TRIBEv2 GPU API → get .npz brain predictions
2. Extract brain features (legacy dimension scores + model features)
3. Run trained XGBoost CTR predictor for performance prediction
4. Save all results to the database

Two types of scores are saved:

LEGACY SCORES (for radar-chart display):
  - visual_score, auditory_score, emotional_score, attention_score, language_score
  - Computed from dimension-grouped brain region means
  - Scaled to 0-100 with a simple linear transform: (raw + 0.5) * 100
  - overall_score uses weighted formula: V*0.25 + A*0.15 + E*0.30 + At*0.20 + L*0.10
  - NOTE: These have low correlation with real CTR (ρ≈0.07). They are kept
    for backward compatibility and dimension-level insights only.

MODEL-BASED PREDICTIONS (for CTR prediction):
  - predicted_ctr: Exact CTR % predicted by XGBoost regressor
  - predicted_class: "High" or "Low" CTR (above/below median)
  - predicted_proba: Probability of being High CTR (0.0–1.0)
  - prediction_tier: Human-readable confidence label
  - overall_score is NOW set to predicted_proba * 100, giving a 0-100
    score that is directly actionable and correlated with real CTR.

The model uses these 6 brain features (selected by Spearman correlation with CTR):
  1. longest_sustained_above_mean  (ρ=0.587) — sustained engagement
  2. emotional_mean                (ρ=0.584) — emotional activation
  3. orbital_mean                  (ρ=0.588) — reward center
  4. visual_std                    (ρ=-0.508) — visual rhythm
  5. insula_short_mean             (ρ=0.490) — gut feelings
  6. attention_onset_second        (ρ=0.520) — engagement build-up speed
"""

import json
import os
import time
import logging
from pathlib import Path

import numpy as np
import requests
import xgboost as xgb
from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

from database import SessionLocal
import models
from analyzer import analyzer as brain_analyzer
from runpod_manager import get_runpod_manager, PodProvisioningError

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logger = get_task_logger(__name__)

# ── Celery Setup ─────────────────────────────────────────────────────────────
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
TRIBEV2_API_BASE_URL = os.getenv("TRIBEV2_API_BASE_URL", "")
RUNPOD_IDLE_TIMEOUT_SECONDS = int(os.getenv("RUNPOD_IDLE_TIMEOUT_SECONDS", "120"))

celery_app = Celery(
    "creative_quality_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    worker_prefetch_multiplier=1,
)

# ── Celery Beat Schedule (for watchdog cleanup) ──────────────────────────────
celery_app.conf.beat_schedule = {
    'watchdog-cleanup-every-3-min': {
        'task': 'watchdog_cleanup_pod',
        'schedule': 180.0,
    },
}
celery_app.conf.timezone = 'UTC'


# ── XGBoost Model Loading ───────────────────────────────────────────────────
# Models are loaded ONCE at module import time and reused for all predictions.
# These are the trained CTR predictor models from model_training/results/best_model.

ML_MODEL_DIR = Path(__file__).parent / "ml_models"

def _load_ml_models():
    """Load XGBoost models and feature list from ml_models/ directory.

    Returns:
        Tuple of (regressor, classifier, feature_list, p10_model, p90_model).
        Quantile models may be None if not available.
    """
    feat_path = ML_MODEL_DIR / "selected_features.json"
    reg_path  = ML_MODEL_DIR / "xgb_regressor.json"
    clf_path  = ML_MODEL_DIR / "xgb_classifier.json"
    p10_path  = ML_MODEL_DIR / "xgb_quantile_p10.json"
    p90_path  = ML_MODEL_DIR / "xgb_quantile_p90.json"

    for p in [feat_path, reg_path, clf_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ML model file: {p}\n"
                f"Copy model files from model_training/results/best_model/ to {ML_MODEL_DIR}/"
            )

    with open(feat_path) as f:
        feature_list = json.load(f)

    reg = xgb.XGBRegressor()
    reg.load_model(str(reg_path))

    clf = xgb.XGBClassifier()
    clf.load_model(str(clf_path))

    # Quantile models (optional — for confidence bounds)
    p10, p90 = None, None
    if p10_path.exists() and p90_path.exists():
        p10 = xgb.XGBRegressor()
        p10.load_model(str(p10_path))
        p90 = xgb.XGBRegressor()
        p90.load_model(str(p90_path))
        logging.info("Quantile models (P10/P90) loaded for confidence bounds.")

    logging.info(f"XGBoost models loaded. Features: {feature_list}")
    return reg, clf, feature_list, p10, p90


# Load models lazily per-worker to avoid OpenMP deadlocks after Celery fork.
_ml_models_cache = None

def get_ml_models():
    global _ml_models_cache
    if _ml_models_cache is None:
        _ml_models_cache = _load_ml_models()
    return _ml_models_cache


def _classify_tier(proba: float) -> str:
    """Map predicted probability to a human-readable tier label."""
    if proba >= 0.80:
        return "Strong High"
    elif proba >= 0.60:
        return "Likely High"
    elif proba >= 0.40:
        return "Borderline"
    elif proba >= 0.20:
        return "Likely Low"
    else:
        return "Strong Low"


# ── Exceptions ───────────────────────────────────────────────────────────────

class PermanentTaskFailure(Exception):
    """Exception raised for errors that cannot be resolved by retrying the task."""
    pass


# ── Database Helpers ─────────────────────────────────────────────────────────

def _update_video_status(video_id: str, status: models.JobStatus,
                         job_id: str = None, npz_path: str = None):
    """Safely updates a video's status, job_id, and npz_path in the database."""
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
            logger.info(f"[{video_id}] DB status → {status.value} (job_id: {job_id})")
        else:
            logger.warning(f"[{video_id}] DB update: video record not found.")
    except Exception as e:
        logger.error(f"[{video_id}] DB update failed: {e}")
        db.rollback()
    finally:
        db.close()


# ── RunPod Cleanup Tasks ─────────────────────────────────────────────────────

@celery_app.task(name="maybe_cleanup_pod")
def maybe_cleanup_pod():
    """Delayed task: delete the pod if idle for longer than the timeout."""
    mgr = get_runpod_manager()
    if mgr:
        try:
            mgr.cleanup_if_idle()
        except Exception as e:
            logger.error(f"maybe_cleanup_pod failed: {e}")


@celery_app.task(name="watchdog_cleanup_pod")
def watchdog_cleanup_pod():
    """Periodic safety net: find and delete orphaned pods."""
    mgr = get_runpod_manager()
    if mgr:
        try:
            mgr.force_cleanup_all()
        except Exception as e:
            logger.error(f"watchdog_cleanup_pod failed: {e}")


# ── Main Task ────────────────────────────────────────────────────────────────

@celery_app.task(name="process_video_task", bind=True, max_retries=5)
def process_video_task(self, video_id: str, video_path: str, is_npz: bool = False):
    """Full pipeline: video → TRIBEv2 → features → CTR prediction → DB."""
    current_attempt = self.request.retries + 1
    logger.info(f"[{video_id}] Starting task. Attempt {current_attempt}/{self.max_retries + 1}")

    # Step 0: Pre-flight checks
    if not os.path.exists(video_path):
        error_msg = f"Video file not found at path: {video_path}"
        logger.error(f"[{video_id}] {error_msg}")
        _update_video_status(video_id, models.JobStatus.FAILED)
        return {"error": error_msg}

    # Determine if we're using managed RunPod mode
    runpod_mgr = get_runpod_manager()
    managed_mode = runpod_mgr is not None

    if managed_mode:
        runpod_mgr.increment_active()

    db = SessionLocal()
    try:
        # Fetch the video record
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            logger.error(f"[{video_id}] Video record not found in database.")
            db.close()
            return {"error": "Video not found"}

        if not is_npz:
            # ── Step 1: Ensure GPU pod is ready ───────────────────────────
            if managed_mode:
                def status_callback(status_str: str):
                    """Update video status during pod lifecycle."""
                    try:
                        status_enum = models.JobStatus(status_str)
                        _update_video_status(video_id, status_enum)
                    except ValueError:
                        logger.warning(f"[{video_id}] Unknown status: {status_str}")

                logger.info(f"[{video_id}] Ensuring GPU pod is ready (managed mode)...")
                base_url = runpod_mgr.ensure_pod_ready(status_callback=status_callback)
                logger.info(f"[{video_id}] GPU pod ready at {base_url}")
            else:
                if not TRIBEV2_API_BASE_URL:
                    raise PermanentTaskFailure(
                        "No GPU endpoint configured. Set TRIBEV2_API_BASE_URL or "
                        "enable RunPod managed mode (RUNPOD_API_KEY + RUNPOD_TEMPLATE_ID)."
                    )
                base_url = TRIBEV2_API_BASE_URL
                logger.info(f"[{video_id}] Using static GPU endpoint: {base_url}")

            job_id = video.job_id

            # ── Step 2: Submit to TRIBEv2 ────────────────────────────────
            if not job_id:
                logger.info(f"[{video_id}] Submitting video to TRIBEv2...")
                video.status = models.JobStatus.INFERENCE
                db.commit()

                url = f"{base_url}/api/v1/jobs/analyze"
                try:
                    with open(video_path, "rb") as f:
                        resp = requests.post(
                            url,
                            files={"video": (os.path.basename(video_path), f, "video/mp4")},
                            timeout=300,
                        )
                    resp.raise_for_status()
                except requests.HTTPError as he:
                    status_code = he.response.status_code if he.response is not None else 500
                    if status_code in [400, 401, 403, 415, 422]:
                        raise PermanentTaskFailure(
                            f"TRIBEv2 API submission failed with client error {status_code}: {he}"
                        )
                    raise

                job_id = resp.json()["job_id"]
                video.job_id = job_id
                db.commit()
                logger.info(f"[{video_id}] Submitted. Job ID: {job_id}")
            else:
                logger.info(f"[{video_id}] Found existing Job ID: {job_id}. Resuming polling.")

            # ── Step 3: Poll for completion ───────────────────────────────
            status_url = f"{base_url}/api/v1/jobs/{job_id}/status"
            logger.info(f"[{video_id}] Polling TRIBEv2 job status...")

            poll_count = 0
            max_polls = 120  # 120 * 15s = 30 min
            while True:
                try:
                    status_resp = requests.get(status_url, timeout=30)
                    if status_resp.status_code == 404:
                        logger.warning(f"[{video_id}] Job {job_id} not found (404). Resetting.")
                        video.job_id = None
                        db.commit()
                        raise Exception(f"TRIBEv2 Job {job_id} not found. Retrying.")
                    status_resp.raise_for_status()
                except requests.HTTPError as he:
                    status_code = he.response.status_code if he.response is not None else 500
                    if status_code in [400, 401, 403, 422]:
                        raise PermanentTaskFailure(
                            f"TRIBEv2 status check permanent error {status_code}: {he}"
                        )
                    raise

                api_status = status_resp.json().get("status", "UNKNOWN")
                logger.info(f"[{video_id}] Poll status: {api_status}")

                if api_status == "COMPLETED":
                    break
                elif api_status in ["FAILED", "DELETED"]:
                    raise PermanentTaskFailure(f"TRIBEv2 terminal status: {api_status}")

                poll_count += 1
                if poll_count >= max_polls:
                    raise Exception(f"Polling timed out after 30 min for Job {job_id}")
                time.sleep(15)

            # ── Step 4: Download .npz atomically ─────────────────────────
            npz_dest = f"{os.path.splitext(video_path)[0]}.npz"
            npz_tmp = f"{npz_dest}.tmp"
            result_url = f"{base_url}/api/v1/jobs/{job_id}/result"

            logger.info(f"[{video_id}] Downloading .npz → {npz_dest}")
            try:
                with requests.get(result_url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(npz_tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                os.replace(npz_tmp, npz_dest)
            except Exception as de:
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

        # ── Step 5: Extract brain features ───────────────────────────────
        logger.info(f"[{video_id}] Analyzing neural features from {npz_dest}...")
        video.status = models.JobStatus.ANALYZING
        db.commit()

        analysis_result = brain_analyzer.analyze(npz_dest)
        raw_scores = analysis_result['raw_scores']
        model_features = analysis_result['model_features']

        # ── Step 6a: Compute LEGACY dimension scores (0-100 scale) ───────
        # These are the radar-chart scores. Kept for backward compatibility.
        # NOTE: These use a simple linear transform and have low CTR correlation.

        def scale_dimension(val):
            """Scale raw brain activation to 0-100 range."""
            scaled = (val + 0.5) * 100
            return max(0.0, min(100.0, scaled))

        visual_score    = scale_dimension(raw_scores['visual'])
        auditory_score  = scale_dimension(raw_scores['auditory'])
        emotional_score = scale_dimension(raw_scores['emotional'])
        language_score  = scale_dimension(raw_scores['language'])
        attention_score = min(100.0, max(0.0, raw_scores['attention'] * 50))

        # ── Step 6b: Run XGBoost CTR prediction (MODEL-BASED) ────────────
        # These are the real CTR predictions from the trained model.

        _reg_model, _clf_model, _selected_features, _p10_model, _p90_model = get_ml_models()

        X = np.array([[model_features[f] for f in _selected_features]])

        # Regression: predict exact CTR %
        predicted_log_ctr = _reg_model.predict(X)[0]
        predicted_ctr = float(np.expm1(predicted_log_ctr))

        # Classification: predict High/Low CTR
        predicted_class_int = int(_clf_model.predict(X)[0])
        predicted_class = "High" if predicted_class_int == 1 else "Low"
        predicted_proba = float(_clf_model.predict_proba(X)[0][1])
        prediction_tier = _classify_tier(predicted_proba)

        # Quantile confidence bounds (if models available)
        ctr_lower = None
        ctr_upper = None
        if _p10_model is not None and _p90_model is not None:
            p10_log = _p10_model.predict(X)[0]
            p90_log = _p90_model.predict(X)[0]
            ctr_lower = float(np.expm1(p10_log))
            ctr_upper = float(np.expm1(p90_log))
            ctr_lower = min(ctr_lower, predicted_ctr)
            ctr_upper = max(ctr_upper, predicted_ctr)

        # Overall score = predicted_proba * 100
        # This gives a 0-100 score that IS correlated with real CTR,
        # unlike the legacy weighted formula.
        overall_score = predicted_proba * 100.0

        logger.info(
            f"[{video_id}] Prediction: CTR={predicted_ctr:.2f}%, "
            f"class={predicted_class}, proba={predicted_proba:.3f}, "
            f"tier={prediction_tier}, overall_score={overall_score:.1f}"
        )

        # ── Step 7: Save to database ─────────────────────────────────────
        score_record = db.query(models.NeuralScore).filter(
            models.NeuralScore.video_id == video_id
        ).first()

        score_data = dict(
            # Model-based overall score (replaces legacy weighted formula)
            overall_score=overall_score,

            # Legacy dimension scores (for radar chart display)
            visual_score=visual_score,
            auditory_score=auditory_score,
            emotional_score=emotional_score,
            attention_score=attention_score,
            language_score=language_score,

            # Model-based CTR predictions
            predicted_ctr=round(predicted_ctr, 4),
            predicted_class=predicted_class,
            predicted_proba=round(predicted_proba, 4),
            prediction_tier=prediction_tier,

            # Raw model feature values (for debugging/audit trail)
            longest_sustained_above_mean=model_features['longest_sustained_above_mean'],
            orbital_mean=model_features['orbital_mean'],
            visual_std=model_features['visual_std'],
            insula_short_mean=model_features['insula_short_mean'],
            attention_onset_second=model_features['attention_onset_second'],
        )

        # Quantile bounds
        if ctr_lower is not None:
            score_data['ctr_lower_bound'] = round(ctr_lower, 4)
            score_data['ctr_upper_bound'] = round(ctr_upper, 4)

        if score_record:
            logger.info(f"[{video_id}] Updating existing neural score record.")
            for key, val in score_data.items():
                setattr(score_record, key, val)
        else:
            logger.info(f"[{video_id}] Creating new neural score record.")
            score_record = models.NeuralScore(video_id=video_id, **score_data)
            db.add(score_record)

        video.status = models.JobStatus.COMPLETED
        db.commit()
        logger.info(f"[{video_id}] Task completed successfully.")

    except PermanentTaskFailure as ptf:
        logger.error(f"[{video_id}] Permanent failure: {ptf}")
        try:
            db.rollback()
            video = db.query(models.Video).filter(models.Video.id == video_id).first()
            if video:
                video.status = models.JobStatus.FAILED
                db.commit()
        except Exception as dbe:
            logger.error(f"[{video_id}] Failed to set FAILED status: {dbe}")
            db.rollback()
        raise ptf

    except PodProvisioningError as ppe:
        # Pod provisioning failed — this is retryable
        logger.error(f"[{video_id}] Pod provisioning error: {ppe}")
        retries = self.request.retries
        max_retries = self.max_retries
        countdown = (2 ** retries) * 30

        if retries >= max_retries:
            logger.error(f"[{video_id}] Max retries ({max_retries}) reached for pod provisioning.")
            try:
                db.rollback()
                video = db.query(models.Video).filter(models.Video.id == video_id).first()
                if video:
                    video.status = models.JobStatus.FAILED
                    db.commit()
            except Exception as dbe:
                logger.error(f"[{video_id}] Failed to set FAILED status: {dbe}")
                db.rollback()
            raise ppe
        else:
            logger.warning(
                f"[{video_id}] Pod provisioning error. "
                f"Retrying in {countdown}s (Attempt {retries + 1}/{max_retries})."
            )
            db.rollback()
            db.close()
            raise self.retry(exc=ppe, countdown=countdown)

    except Exception as exc:
        retries = self.request.retries
        max_retries = self.max_retries
        countdown = (2 ** retries) * 30

        if retries >= max_retries:
            logger.error(f"[{video_id}] Max retries ({max_retries}) reached. Error: {exc}")
            try:
                db.rollback()
                video = db.query(models.Video).filter(models.Video.id == video_id).first()
                if video:
                    video.status = models.JobStatus.FAILED
                    db.commit()
            except Exception as dbe:
                logger.error(f"[{video_id}] Failed to set FAILED status: {dbe}")
                db.rollback()
            raise exc
        else:
            logger.warning(
                f"[{video_id}] Transient error: {exc}. "
                f"Retrying in {countdown}s (Attempt {retries + 1}/{max_retries})."
            )
            db.rollback()
            db.close()
            raise self.retry(exc=exc, countdown=countdown)

    finally:
        # ALWAYS decrement active count and schedule cleanup
        if managed_mode:
            try:
                runpod_mgr.decrement_active()
                # Schedule cleanup check after idle timeout
                maybe_cleanup_pod.apply_async(countdown=RUNPOD_IDLE_TIMEOUT_SECONDS)
                logger.info(f"[{video_id}] Scheduled pod cleanup check in {RUNPOD_IDLE_TIMEOUT_SECONDS}s.")
            except Exception as ce:
                logger.error(f"[{video_id}] Failed to schedule cleanup: {ce}")
        db.close()

    return {"status": "finished", "video_id": video_id}
