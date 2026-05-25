"""
RunPod Pod Lifecycle Manager
=============================
Manages the full lifecycle of RunPod GPU pods:
- On-demand creation when a video analysis is requested
- Health checking and readiness polling
- Automatic cleanup after an idle timeout
- Watchdog safety net for orphaned pods

State is stored in Redis (shared across all Celery workers):
  runpod:pod_id         — Current active pod ID
  runpod:pod_url        — Current pod proxy base URL
  runpod:active_count   — Number of in-flight analyses
  runpod:last_idle_at   — Timestamp when active_count dropped to 0
  runpod:create_lock    — Distributed lock for pod creation
"""

import os
import time
import logging
from typing import Optional, Callable

import redis
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_TEMPLATE_ID = os.getenv("RUNPOD_TEMPLATE_ID", "")
RUNPOD_GPU_TYPES = [
    g.strip()
    for g in os.getenv("RUNPOD_GPU_TYPES", "NVIDIA RTX A4000,NVIDIA RTX A5000,NVIDIA A40").split(",")
    if g.strip()
]
RUNPOD_IDLE_TIMEOUT_SECONDS = int(os.getenv("RUNPOD_IDLE_TIMEOUT_SECONDS", "120"))
RUNPOD_API_BASE = "https://rest.runpod.io/v1"

# Timeouts
POD_CREATION_TIMEOUT = 600      # 10 min max for pod to reach RUNNING
POD_HEALTH_TIMEOUT = 300        # 5 min max for port 8000 to respond
CREATE_LOCK_TTL = 600           # 10 min lock TTL for pod creation
HEALTH_CHECK_INTERVAL = 10     # seconds between health check polls
POD_STATUS_POLL_INTERVAL = 10  # seconds between pod status polls


# ── Exceptions ───────────────────────────────────────────────────────────────

class PodProvisioningError(Exception):
    """Raised when pod creation or readiness check fails."""
    pass


class PodCleanupError(Exception):
    """Raised when pod deletion fails."""
    pass


# ── RunPod Manager ───────────────────────────────────────────────────────────

class RunPodManager:
    """Manages RunPod GPU pod lifecycle with Redis-backed state."""

    REDIS_POD_ID = "runpod:pod_id"
    REDIS_POD_URL = "runpod:pod_url"
    REDIS_ACTIVE_COUNT = "runpod:active_count"
    REDIS_LAST_IDLE_AT = "runpod:last_idle_at"
    REDIS_CREATE_LOCK = "runpod:create_lock"

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.api_key = RUNPOD_API_KEY
        self.template_id = RUNPOD_TEMPLATE_ID
        self.gpu_types = RUNPOD_GPU_TYPES
        self.idle_timeout = RUNPOD_IDLE_TIMEOUT_SECONDS

        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required for managed pod mode")
        if not self.template_id:
            raise ValueError("RUNPOD_TEMPLATE_ID is required for managed pod mode")

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Public API ───────────────────────────────────────────────────────

    def ensure_pod_ready(self, status_callback: Optional[Callable] = None) -> str:
        """Ensure a running, healthy pod exists. Returns the base URL.

        If a pod already exists and is healthy, returns immediately.
        If no pod exists, creates one and waits for it to become ready.
        Uses a distributed lock to prevent multiple workers from creating
        pods simultaneously.

        Args:
            status_callback: Optional function called with status strings
                           ('PROVISIONING_GPU', 'BOOTING_GPU') for progress updates.

        Returns:
            Base URL of the ready pod (e.g., 'https://abc123-8000.proxy.runpod.net')

        Raises:
            PodProvisioningError: If pod creation or readiness check fails.
        """
        # 1. Check for an existing healthy pod
        existing_url = self._get_cached_pod_url()
        if existing_url:
            logger.info(f"Found cached pod URL: {existing_url}")
            if self._health_check(existing_url):
                logger.info("Cached pod is healthy. Reusing.")
                return existing_url
            else:
                logger.warning("Cached pod URL is not healthy. Checking pod status...")
                pod_id = self.redis.get(self.REDIS_POD_ID)
                if pod_id:
                    pod_id = pod_id.decode() if isinstance(pod_id, bytes) else pod_id
                    pod_info = self._get_pod(pod_id)
                    if pod_info and pod_info.get("desiredStatus") == "RUNNING":
                        # Pod is running but service not ready yet — wait for health
                        if status_callback:
                            status_callback("BOOTING_GPU")
                        logger.info(f"Pod {pod_id} is RUNNING, waiting for health...")
                        self._wait_for_health(existing_url)
                        return existing_url
                    else:
                        # Pod is gone or stopped — clear cache
                        logger.warning(f"Pod {pod_id} is not running. Clearing cache.")
                        self._clear_pod_cache()

        # 2. Acquire creation lock
        lock_acquired = self.redis.set(
            self.REDIS_CREATE_LOCK, "locked", nx=True, ex=CREATE_LOCK_TTL
        )

        if not lock_acquired:
            # Another worker is creating a pod — wait for it
            logger.info("Another worker is creating a pod. Waiting for pod URL...")
            return self._wait_for_cached_pod(status_callback)

        try:
            # 3. Double-check after acquiring lock (another worker may have finished)
            existing_url = self._get_cached_pod_url()
            if existing_url and self._health_check(existing_url):
                logger.info("Pod became available while acquiring lock.")
                return existing_url

            # 4. Create a new pod
            if status_callback:
                status_callback("PROVISIONING_GPU")
            logger.info("Creating new RunPod pod...")
            pod_info = self._create_pod()
            pod_id = pod_info["id"]
            logger.info(f"Pod created: {pod_id} (name: {pod_info.get('name', 'unknown')})")

            # Store pod_id immediately (even before RUNNING)
            self.redis.set(self.REDIS_POD_ID, pod_id)

            # 5. Wait for pod to reach RUNNING status
            self._wait_for_running(pod_id)

            # 6. Wait for port 8000 health check
            if status_callback:
                status_callback("BOOTING_GPU")
            base_url = self._get_proxy_url(pod_id)
            logger.info(f"Pod {pod_id} is RUNNING. Waiting for health at {base_url}...")
            self._wait_for_health(base_url)

            # 7. Cache the URL
            self.redis.set(self.REDIS_POD_URL, base_url)
            logger.info(f"Pod {pod_id} is fully ready at {base_url}")
            return base_url

        except Exception as e:
            # Clean up on failure
            logger.error(f"Pod provisioning failed: {e}")
            pod_id_raw = self.redis.get(self.REDIS_POD_ID)
            if pod_id_raw:
                pod_id = pod_id_raw.decode() if isinstance(pod_id_raw, bytes) else pod_id_raw
                try:
                    self._delete_pod(pod_id)
                except Exception as de:
                    logger.error(f"Failed to cleanup pod {pod_id} after error: {de}")
            self._clear_pod_cache()
            raise PodProvisioningError(f"Failed to provision GPU pod: {e}") from e

        finally:
            self.redis.delete(self.REDIS_CREATE_LOCK)

    def increment_active(self):
        """Atomically increment the active analysis count."""
        count = self.redis.incr(self.REDIS_ACTIVE_COUNT)
        logger.info(f"Active analysis count incremented to {count}")
        return count

    def decrement_active(self):
        """Atomically decrement the active analysis count.

        If count reaches 0, records the idle timestamp.
        Ensures count never goes below 0.
        """
        count = self.redis.decr(self.REDIS_ACTIVE_COUNT)
        if count <= 0:
            # Ensure non-negative
            self.redis.set(self.REDIS_ACTIVE_COUNT, 0)
            self.redis.set(self.REDIS_LAST_IDLE_AT, str(time.time()))
            logger.info("Active analysis count reached 0. Recorded idle timestamp.")
        else:
            logger.info(f"Active analysis count decremented to {count}")
        return max(0, count)

    def cleanup_if_idle(self):
        """Delete the pod if it has been idle for longer than the timeout.

        Called by the delayed Celery task `maybe_cleanup_pod`.
        Safe to call multiple times — only deletes once.
        """
        active_count = int(self.redis.get(self.REDIS_ACTIVE_COUNT) or 0)
        if active_count > 0:
            logger.info(f"Cleanup skipped: {active_count} active analyses.")
            return False

        last_idle_raw = self.redis.get(self.REDIS_LAST_IDLE_AT)
        if not last_idle_raw:
            logger.info("Cleanup skipped: no idle timestamp recorded.")
            return False

        last_idle = float(last_idle_raw.decode() if isinstance(last_idle_raw, bytes) else last_idle_raw)
        elapsed = time.time() - last_idle

        if elapsed < self.idle_timeout:
            logger.info(f"Cleanup skipped: only {elapsed:.0f}s idle (need {self.idle_timeout}s).")
            return False

        pod_id_raw = self.redis.get(self.REDIS_POD_ID)
        if not pod_id_raw:
            logger.info("Cleanup skipped: no pod ID in cache.")
            return False

        pod_id = pod_id_raw.decode() if isinstance(pod_id_raw, bytes) else pod_id_raw
        logger.info(f"Pod {pod_id} has been idle for {elapsed:.0f}s. Deleting...")

        try:
            self._delete_pod(pod_id)
            self._clear_pod_cache()
            logger.info(f"Pod {pod_id} deleted successfully after idle timeout.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete idle pod {pod_id}: {e}")
            return False

    def force_cleanup_all(self):
        """Watchdog: find and delete all pods belonging to our template.

        Lists all pods via RunPod API and deletes any that:
        1. Match our template_id
        2. Have no active analyses (based on Redis counter)

        This is the safety net for crashed workers, lost Redis data, etc.
        """
        active_count = int(self.redis.get(self.REDIS_ACTIVE_COUNT) or 0)
        if active_count > 0:
            logger.info(f"Watchdog: {active_count} active analyses. Skipping cleanup.")
            return

        try:
            pods = self._list_pods()
        except Exception as e:
            logger.error(f"Watchdog: failed to list pods: {e}")
            return

        our_pods = [
            p for p in pods
            if p.get("templateId") == self.template_id
        ]

        if not our_pods:
            logger.debug("Watchdog: no pods found with our template.")
            return

        cached_pod_id_raw = self.redis.get(self.REDIS_POD_ID)
        cached_pod_id = None
        if cached_pod_id_raw:
            cached_pod_id = cached_pod_id_raw.decode() if isinstance(cached_pod_id_raw, bytes) else cached_pod_id_raw

        for pod in our_pods:
            pod_id = pod["id"]
            logger.warning(f"Watchdog: found orphaned pod {pod_id}. Deleting...")
            try:
                self._delete_pod(pod_id)
                if pod_id == cached_pod_id:
                    self._clear_pod_cache()
                logger.info(f"Watchdog: deleted orphaned pod {pod_id}.")
            except Exception as e:
                logger.error(f"Watchdog: failed to delete pod {pod_id}: {e}")

    # ── Internal: RunPod API Calls ────────────────────────────────────────

    def _create_pod(self) -> dict:
        """Create a new RunPod pod via REST API.

        Returns the pod creation response dict containing at least 'id'.
        Retries up to 3 times on transient failures.
        """
        url = f"{RUNPOD_API_BASE}/pods"
        payload = {
            "templateId": self.template_id,
            "gpuCount": 1,
            "gpuTypeIds": self.gpu_types,
            "gpuTypePriority": "availability",
        }

        last_error = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    url, json=payload, headers=self._headers, timeout=60
                )
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else 500
                if status_code in [400, 401, 403, 422]:
                    raise PodProvisioningError(
                        f"RunPod API rejected pod creation (HTTP {status_code}): "
                        f"{e.response.text if e.response else str(e)}"
                    )
                last_error = e
                logger.warning(
                    f"Pod creation attempt {attempt + 1}/3 failed (HTTP {status_code}): {e}"
                )
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Pod creation attempt {attempt + 1}/3 failed: {e}")

            if attempt < 2:
                time.sleep(5 * (attempt + 1))

        raise PodProvisioningError(f"Pod creation failed after 3 attempts: {last_error}")

    def _get_pod(self, pod_id: str) -> Optional[dict]:
        """Get pod info by ID. Returns None if pod doesn't exist."""
        url = f"{RUNPOD_API_BASE}/pods/{pod_id}"
        try:
            resp = requests.get(url, headers=self._headers, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get pod {pod_id}: {e}")
            return None

    def _delete_pod(self, pod_id: str) -> bool:
        """Delete a pod by ID. Returns True on success.

        Retries up to 3 times. Treats 404 as success (already deleted).
        """
        url = f"{RUNPOD_API_BASE}/pods/{pod_id}"

        for attempt in range(3):
            try:
                resp = requests.delete(url, headers=self._headers, timeout=30)
                if resp.status_code == 404:
                    logger.info(f"Pod {pod_id} already deleted (404).")
                    return True
                resp.raise_for_status()
                logger.info(f"Pod {pod_id} deletion request accepted.")
                return True
            except requests.RequestException as e:
                logger.warning(f"Pod deletion attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(3)

        raise PodCleanupError(f"Failed to delete pod {pod_id} after 3 attempts")

    def _list_pods(self) -> list:
        """List all pods via RunPod API."""
        url = f"{RUNPOD_API_BASE}/pods"
        resp = requests.get(url, headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Internal: Polling & Health ────────────────────────────────────────

    def _wait_for_running(self, pod_id: str, timeout: int = POD_CREATION_TIMEOUT):
        """Poll until pod status is RUNNING.

        Raises PodProvisioningError on timeout or terminal failure states.
        """
        start = time.time()
        last_status = "UNKNOWN"

        while time.time() - start < timeout:
            pod_info = self._get_pod(pod_id)
            if not pod_info:
                raise PodProvisioningError(f"Pod {pod_id} not found while waiting for RUNNING")

            runtime = pod_info.get("runtime")
            logger.info(f"DEBUG Pod info: {pod_info}")
            
            # Check if pod has a runtime with uptime (means it's running)
            if runtime and runtime.get("uptimeInSeconds") is not None:
                logger.info(f"Pod {pod_id} is RUNNING (uptime: {runtime.get('uptimeInSeconds')}s)")
                return
                
            # FALLBACK: Sometimes RunPod API lags behind the actual pod state.
            # Let's actively ping the proxy URL to see if it's already alive!
            base_url = self._get_proxy_url(pod_id)
            if self._health_check(base_url):
                logger.info(f"Pod {pod_id} health check passed early! It is definitely RUNNING.")
                return

            desired = pod_info.get("desiredStatus", "UNKNOWN")
            last_status = pod_info.get("lastStatusChange", desired)

            # Check for terminal error states
            if desired in ["EXITED", "TERMINATED"]:
                raise PodProvisioningError(
                    f"Pod {pod_id} reached terminal state: {desired}"
                )

            logger.info(
                f"Pod {pod_id} not ready yet (desired: {desired}, "
                f"elapsed: {time.time() - start:.0f}s). Polling..."
            )
            time.sleep(POD_STATUS_POLL_INTERVAL)

        raise PodProvisioningError(
            f"Pod {pod_id} did not reach RUNNING within {timeout}s. "
            f"Last status: {last_status}"
        )

    def _wait_for_health(self, base_url: str, timeout: int = POD_HEALTH_TIMEOUT):
        """Poll until the pod's HTTP service responds on port 8000.

        Tries the /health endpoint first, falls back to root /.
        Raises PodProvisioningError on timeout.
        """
        start = time.time()

        while time.time() - start < timeout:
            if self._health_check(base_url):
                logger.info(f"Pod health check passed at {base_url}")
                return

            elapsed = time.time() - start
            logger.info(
                f"Health check not passing yet (elapsed: {elapsed:.0f}s). Retrying..."
            )
            time.sleep(HEALTH_CHECK_INTERVAL)

        raise PodProvisioningError(
            f"Pod health check timed out after {timeout}s at {base_url}"
        )

    def _health_check(self, base_url: str) -> bool:
        """Check if the pod's HTTP service is responding.

        Tries /health first (standard FastAPI health endpoint),
        then falls back to / (root endpoint).
        """
        for path in ["/health", "/"]:
            try:
                resp = requests.get(f"{base_url}{path}", timeout=10)
                if resp.status_code < 500:
                    return True
            except requests.RequestException:
                pass
        return False

    # ── Internal: Redis Helpers ───────────────────────────────────────────

    def _get_cached_pod_url(self) -> Optional[str]:
        """Get the cached pod URL from Redis."""
        url = self.redis.get(self.REDIS_POD_URL)
        if url:
            return url.decode() if isinstance(url, bytes) else url
        return None

    def _clear_pod_cache(self):
        """Clear all pod-related Redis keys."""
        self.redis.delete(
            self.REDIS_POD_ID,
            self.REDIS_POD_URL,
            self.REDIS_LAST_IDLE_AT,
        )
        logger.info("Pod cache cleared.")

    def _wait_for_cached_pod(self, status_callback: Optional[Callable] = None,
                              timeout: int = POD_CREATION_TIMEOUT + POD_HEALTH_TIMEOUT) -> str:
        """Wait for another worker to finish creating a pod.

        Polls Redis for the pod URL until it appears and the pod is healthy.
        """
        if status_callback:
            status_callback("PROVISIONING_GPU")

        start = time.time()
        while time.time() - start < timeout:
            url = self._get_cached_pod_url()
            if url:
                if status_callback:
                    status_callback("BOOTING_GPU")
                if self._health_check(url):
                    logger.info(f"Pod URL appeared in cache and is healthy: {url}")
                    return url

            time.sleep(HEALTH_CHECK_INTERVAL)

        raise PodProvisioningError(
            f"Timed out waiting for another worker to create pod ({timeout}s)"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_proxy_url(pod_id: str) -> str:
        """Construct the RunPod proxy URL for port 8000."""
        return f"https://{pod_id}-8000.proxy.runpod.net"


# ── Module-level Singleton ───────────────────────────────────────────────────

_manager_instance: Optional[RunPodManager] = None


def get_runpod_manager() -> Optional[RunPodManager]:
    """Get or create the RunPodManager singleton.

    Returns None if RunPod managed mode is not configured
    (i.e., RUNPOD_API_KEY or RUNPOD_TEMPLATE_ID is not set).
    """
    global _manager_instance

    if _manager_instance is not None:
        return _manager_instance

    if not RUNPOD_API_KEY or not RUNPOD_TEMPLATE_ID:
        logger.info(
            "RunPod managed mode not configured "
            "(RUNPOD_API_KEY or RUNPOD_TEMPLATE_ID not set). "
            "Using static TRIBEV2_API_BASE_URL."
        )
        return None

    redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    redis_client = redis.Redis.from_url(redis_url)

    try:
        redis_client.ping()
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis for RunPod state: {e}")
        return None

    _manager_instance = RunPodManager(redis_client)
    logger.info(
        f"RunPod managed mode initialized. Template: {RUNPOD_TEMPLATE_ID}, "
        f"GPU types: {RUNPOD_GPU_TYPES}, Idle timeout: {RUNPOD_IDLE_TIMEOUT_SECONDS}s"
    )
    return _manager_instance
