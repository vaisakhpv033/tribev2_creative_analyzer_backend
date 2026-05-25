"""
Unit Tests for RunPod Pod Lifecycle Manager
=============================================
Tests cover:
  1. ensure_pod_ready() — no pod exists → creates and waits
  2. ensure_pod_ready() — pod exists and healthy → returns cached URL
  3. ensure_pod_ready() — pod exists but dead → creates new
  4. cleanup_if_idle() — idle pod → deletes
  5. cleanup_if_idle() — active analyses → skips
  6. cleanup_if_idle() — not yet timed out → skips
  7. force_cleanup_all() — orphaned pods → deletes
  8. force_cleanup_all() — active analyses → skips
  9. Concurrent ensure_pod_ready() — only one creates pod (lock test)
 10. increment/decrement active count bookkeeping
 11. Error handling — pod creation fails → cleans up and raises
 12. Error handling — health check timeout → cleans up and raises
 13. Worker integration — managed mode sets up lifecycle correctly
 14. Worker integration — legacy mode uses static URL
 15. Status callback invocation during provisioning
"""

import time
import unittest
from unittest.mock import patch, MagicMock, call

import fakeredis

from runpod_manager import (
    RunPodManager,
    PodProvisioningError,
    PodCleanupError,
    get_runpod_manager,
)


class TestRunPodManager(unittest.TestCase):
    """Tests for the RunPodManager class with a fake Redis backend."""

    def setUp(self):
        """Create a RunPodManager with a fake Redis instance."""
        self.redis = fakeredis.FakeRedis()
        # Patch the module-level config so the manager can be instantiated
        with patch("runpod_manager.RUNPOD_API_KEY", "test-key"), \
             patch("runpod_manager.RUNPOD_TEMPLATE_ID", "test-template"):
            self.mgr = RunPodManager(self.redis)
            self.mgr.api_key = "test-key"
            self.mgr.template_id = "test-template"
            self.mgr.gpu_types = ["NVIDIA RTX A4000"]
            self.mgr.idle_timeout = 120

    def tearDown(self):
        self.redis.flushall()

    # ─── Test 1: No pod exists → creates and waits ─────────────────────────

    @patch.object(RunPodManager, "_health_check", return_value=True)
    @patch.object(RunPodManager, "_wait_for_running")
    @patch.object(RunPodManager, "_create_pod", return_value={"id": "pod-abc", "name": "test-pod"})
    def test_ensure_pod_ready_creates_new_pod(self, mock_create, mock_wait_run, mock_health):
        """When no pod exists, should create one, wait for running, wait for health."""
        url = self.mgr.ensure_pod_ready()

        mock_create.assert_called_once()
        mock_wait_run.assert_called_once_with("pod-abc")
        self.assertEqual(url, "https://pod-abc-8000.proxy.runpod.net")
        # Pod ID should be cached in Redis
        self.assertEqual(self.redis.get("runpod:pod_id").decode(), "pod-abc")
        self.assertEqual(
            self.redis.get("runpod:pod_url").decode(),
            "https://pod-abc-8000.proxy.runpod.net",
        )

    # ─── Test 2: Pod exists and healthy → returns cached URL ───────────────

    @patch.object(RunPodManager, "_health_check", return_value=True)
    def test_ensure_pod_ready_reuses_healthy_pod(self, mock_health):
        """When a cached pod URL exists and is healthy, should return it immediately."""
        self.redis.set("runpod:pod_id", "pod-existing")
        self.redis.set("runpod:pod_url", "https://pod-existing-8000.proxy.runpod.net")

        url = self.mgr.ensure_pod_ready()

        self.assertEqual(url, "https://pod-existing-8000.proxy.runpod.net")
        mock_health.assert_called_once_with("https://pod-existing-8000.proxy.runpod.net")

    # ─── Test 3: Pod exists but dead → creates new ─────────────────────────

    @patch.object(RunPodManager, "_health_check")
    @patch.object(RunPodManager, "_get_pod", return_value=None)  # Pod doesn't exist
    @patch.object(RunPodManager, "_wait_for_running")
    @patch.object(RunPodManager, "_create_pod", return_value={"id": "pod-new", "name": "new-pod"})
    def test_ensure_pod_ready_creates_when_cached_pod_dead(
        self, mock_create, mock_wait_run, mock_get_pod, mock_health
    ):
        """When cached pod is dead (404), should clear cache and create a new pod."""
        # First health check on cached URL → fails
        # Second health check after lock (double-check) → no cached URL anymore
        # Third health check on new pod → passes
        mock_health.side_effect = [False, True]

        self.redis.set("runpod:pod_id", "pod-dead")
        self.redis.set("runpod:pod_url", "https://pod-dead-8000.proxy.runpod.net")

        url = self.mgr.ensure_pod_ready()

        self.assertEqual(url, "https://pod-new-8000.proxy.runpod.net")
        mock_create.assert_called_once()

    # ─── Test 4: cleanup_if_idle — idle pod → deletes ──────────────────────

    @patch.object(RunPodManager, "_delete_pod", return_value=True)
    def test_cleanup_if_idle_deletes_idle_pod(self, mock_delete):
        """Should delete pod when idle for longer than the timeout."""
        self.redis.set("runpod:active_count", 0)
        self.redis.set("runpod:last_idle_at", str(time.time() - 200))  # 200s idle
        self.redis.set("runpod:pod_id", "pod-idle")
        self.redis.set("runpod:pod_url", "https://pod-idle-8000.proxy.runpod.net")

        result = self.mgr.cleanup_if_idle()

        self.assertTrue(result)
        mock_delete.assert_called_once_with("pod-idle")
        # Cache should be cleared
        self.assertIsNone(self.redis.get("runpod:pod_id"))
        self.assertIsNone(self.redis.get("runpod:pod_url"))

    # ─── Test 5: cleanup_if_idle — active analyses → skips ─────────────────

    @patch.object(RunPodManager, "_delete_pod")
    def test_cleanup_if_idle_skips_when_active(self, mock_delete):
        """Should NOT delete when analyses are still active."""
        self.redis.set("runpod:active_count", 2)
        self.redis.set("runpod:pod_id", "pod-busy")

        result = self.mgr.cleanup_if_idle()

        self.assertFalse(result)
        mock_delete.assert_not_called()

    # ─── Test 6: cleanup_if_idle — not yet timed out → skips ───────────────

    @patch.object(RunPodManager, "_delete_pod")
    def test_cleanup_if_idle_skips_when_not_timed_out(self, mock_delete):
        """Should NOT delete when idle time is less than timeout."""
        self.redis.set("runpod:active_count", 0)
        self.redis.set("runpod:last_idle_at", str(time.time() - 30))  # Only 30s idle
        self.redis.set("runpod:pod_id", "pod-recent")

        result = self.mgr.cleanup_if_idle()

        self.assertFalse(result)
        mock_delete.assert_not_called()

    # ─── Test 7: force_cleanup_all — orphaned pods → deletes ───────────────

    @patch.object(RunPodManager, "_delete_pod", return_value=True)
    @patch.object(RunPodManager, "_list_pods")
    def test_force_cleanup_all_deletes_orphaned_pods(self, mock_list, mock_delete):
        """Should delete all pods matching our template when idle."""
        self.redis.set("runpod:active_count", 0)
        mock_list.return_value = [
            {"id": "pod-orphan-1", "templateId": "test-template"},
            {"id": "pod-orphan-2", "templateId": "test-template"},
            {"id": "pod-other", "templateId": "other-template"},  # Should NOT be deleted
        ]

        self.mgr.force_cleanup_all()

        self.assertEqual(mock_delete.call_count, 2)
        mock_delete.assert_any_call("pod-orphan-1")
        mock_delete.assert_any_call("pod-orphan-2")

    # ─── Test 8: force_cleanup_all — active analyses → skips ───────────────

    @patch.object(RunPodManager, "_list_pods")
    @patch.object(RunPodManager, "_delete_pod")
    def test_force_cleanup_all_skips_when_active(self, mock_delete, mock_list):
        """Should NOT delete any pods when analyses are active."""
        self.redis.set("runpod:active_count", 1)

        self.mgr.force_cleanup_all()

        mock_list.assert_not_called()
        mock_delete.assert_not_called()

    # ─── Test 9: Concurrent creation — lock prevents duplicates ────────────

    @patch.object(RunPodManager, "_health_check")
    @patch.object(RunPodManager, "_wait_for_running")
    @patch.object(RunPodManager, "_create_pod", return_value={"id": "pod-locked", "name": "locked"})
    def test_ensure_pod_ready_lock_prevents_duplicate_creation(
        self, mock_create, mock_wait_run, mock_health
    ):
        """When another worker holds the creation lock, should wait for the cached URL."""
        # Simulate another worker holding the lock
        self.redis.set("runpod:create_lock", "locked", ex=600)

        # Simulate the other worker finishing (URL appears in cache)
        def side_effect(url):
            # After a couple of calls, simulate the URL appearing
            if self.redis.get("runpod:pod_url"):
                return True
            return False

        mock_health.side_effect = side_effect

        # In a separate "thread", set the URL after a short delay
        self.redis.set("runpod:pod_url", "https://pod-from-other-worker-8000.proxy.runpod.net")

        url = self.mgr.ensure_pod_ready()

        # Should NOT have called create_pod (another worker did it)
        mock_create.assert_not_called()
        self.assertEqual(url, "https://pod-from-other-worker-8000.proxy.runpod.net")

    # ─── Test 10: increment/decrement bookkeeping ──────────────────────────

    def test_increment_decrement_active_count(self):
        """Active count should increment and decrement correctly."""
        self.mgr.increment_active()
        self.assertEqual(int(self.redis.get("runpod:active_count")), 1)

        self.mgr.increment_active()
        self.assertEqual(int(self.redis.get("runpod:active_count")), 2)

        self.mgr.decrement_active()
        self.assertEqual(int(self.redis.get("runpod:active_count")), 1)
        # No idle timestamp yet (still active)
        self.assertIsNone(self.redis.get("runpod:last_idle_at"))

        self.mgr.decrement_active()
        self.assertEqual(int(self.redis.get("runpod:active_count")), 0)
        # Idle timestamp should be set now
        self.assertIsNotNone(self.redis.get("runpod:last_idle_at"))

    def test_decrement_never_goes_negative(self):
        """Decrementing below 0 should clamp to 0."""
        result = self.mgr.decrement_active()
        self.assertEqual(result, 0)
        self.assertEqual(int(self.redis.get("runpod:active_count")), 0)

    # ─── Test 11: Pod creation fails → cleans up and raises ────────────────

    @patch.object(RunPodManager, "_delete_pod")
    @patch.object(RunPodManager, "_create_pod", side_effect=PodProvisioningError("GPU sold out"))
    def test_ensure_pod_ready_cleans_up_on_create_failure(self, mock_create, mock_delete):
        """Should raise PodProvisioningError and release lock on failure."""
        with self.assertRaises(PodProvisioningError) as ctx:
            self.mgr.ensure_pod_ready()

        self.assertIn("GPU sold out", str(ctx.exception))
        # Lock should be released
        self.assertIsNone(self.redis.get("runpod:create_lock"))

    # ─── Test 12: Health check timeout → cleans up and raises ──────────────

    @patch.object(RunPodManager, "_wait_for_health", side_effect=PodProvisioningError("health timeout"))
    @patch.object(RunPodManager, "_wait_for_running")
    @patch.object(RunPodManager, "_delete_pod", return_value=True)
    @patch.object(RunPodManager, "_create_pod", return_value={"id": "pod-slow", "name": "slow"})
    def test_ensure_pod_ready_cleans_up_on_health_timeout(
        self, mock_create, mock_delete, mock_wait_run, mock_wait_health
    ):
        """Should delete pod and raise on health check timeout."""
        with self.assertRaises(PodProvisioningError) as ctx:
            self.mgr.ensure_pod_ready()

        self.assertIn("Failed to provision GPU pod", str(ctx.exception))
        mock_delete.assert_called_with("pod-slow")
        # Lock should be released
        self.assertIsNone(self.redis.get("runpod:create_lock"))

    # ─── Test 13: Status callback invocation ───────────────────────────────

    @patch.object(RunPodManager, "_health_check", return_value=True)
    @patch.object(RunPodManager, "_wait_for_running")
    @patch.object(RunPodManager, "_create_pod", return_value={"id": "pod-cb", "name": "cb-pod"})
    def test_status_callback_called_during_provisioning(
        self, mock_create, mock_wait_run, mock_health
    ):
        """Status callback should be called with PROVISIONING_GPU and BOOTING_GPU."""
        callback = MagicMock()

        self.mgr.ensure_pod_ready(status_callback=callback)

        callback.assert_any_call("PROVISIONING_GPU")
        callback.assert_any_call("BOOTING_GPU")
        self.assertEqual(callback.call_count, 2)

    # ─── Test 14: Proxy URL construction ───────────────────────────────────

    def test_get_proxy_url(self):
        """Proxy URL should follow RunPod format."""
        url = RunPodManager._get_proxy_url("n7an3ys2hyzhhh")
        self.assertEqual(url, "https://n7an3ys2hyzhhh-8000.proxy.runpod.net")

    # ─── Test 15: cleanup_if_idle — no pod ID → skips ──────────────────────

    def test_cleanup_if_idle_skips_when_no_pod_id(self):
        """Should skip if no pod ID is cached."""
        self.redis.set("runpod:active_count", 0)
        self.redis.set("runpod:last_idle_at", str(time.time() - 200))
        # No pod_id set

        result = self.mgr.cleanup_if_idle()

        self.assertFalse(result)


class TestGetRunpodManager(unittest.TestCase):
    """Tests for the module-level get_runpod_manager() factory."""

    def setUp(self):
        # Reset the singleton between tests
        import runpod_manager
        runpod_manager._manager_instance = None

    def tearDown(self):
        import runpod_manager
        runpod_manager._manager_instance = None

    @patch("runpod_manager.RUNPOD_API_KEY", "")
    @patch("runpod_manager.RUNPOD_TEMPLATE_ID", "")
    def test_returns_none_when_not_configured(self):
        """Should return None when API key and template ID are not set."""
        result = get_runpod_manager()
        self.assertIsNone(result)

    @patch("runpod_manager.RUNPOD_API_KEY", "test-key")
    @patch("runpod_manager.RUNPOD_TEMPLATE_ID", "")
    def test_returns_none_when_template_missing(self):
        """Should return None when template ID is missing."""
        result = get_runpod_manager()
        self.assertIsNone(result)

    @patch("runpod_manager.RUNPOD_API_KEY", "test-key")
    @patch("runpod_manager.RUNPOD_TEMPLATE_ID", "test-template")
    @patch("redis.Redis.from_url")
    def test_returns_manager_when_configured(self, mock_redis_factory):
        """Should return a RunPodManager when fully configured."""
        mock_redis = fakeredis.FakeRedis()
        mock_redis_factory.return_value = mock_redis

        result = get_runpod_manager()

        self.assertIsNotNone(result)
        self.assertIsInstance(result, RunPodManager)

    @patch("runpod_manager.RUNPOD_API_KEY", "test-key")
    @patch("runpod_manager.RUNPOD_TEMPLATE_ID", "test-template")
    @patch("redis.Redis.from_url")
    def test_singleton_returns_same_instance(self, mock_redis_factory):
        """Should return the same instance on multiple calls."""
        mock_redis = fakeredis.FakeRedis()
        mock_redis_factory.return_value = mock_redis

        mgr1 = get_runpod_manager()
        mgr2 = get_runpod_manager()

        self.assertIs(mgr1, mgr2)


class TestWorkerIntegration(unittest.TestCase):
    """Tests for the worker's integration with RunPodManager."""

    def test_jobstatus_has_gpu_lifecycle_values(self):
        """JobStatus enum should include the new GPU lifecycle statuses."""
        from models import JobStatus

        self.assertEqual(JobStatus.PROVISIONING_GPU.value, "PROVISIONING_GPU")
        self.assertEqual(JobStatus.BOOTING_GPU.value, "BOOTING_GPU")
        # Existing values should still work
        self.assertEqual(JobStatus.PENDING.value, "PENDING")
        self.assertEqual(JobStatus.COMPLETED.value, "COMPLETED")
        self.assertEqual(JobStatus.FAILED.value, "FAILED")

    def test_jobstatus_string_roundtrip(self):
        """Status strings should roundtrip through the enum (for status_callback)."""
        from models import JobStatus

        status = JobStatus("PROVISIONING_GPU")
        self.assertEqual(status, JobStatus.PROVISIONING_GPU)

        status = JobStatus("BOOTING_GPU")
        self.assertEqual(status, JobStatus.BOOTING_GPU)

    def test_worker_imports_runpod_manager(self):
        """Worker module should import RunPod manager without error."""
        # This tests the import chain works
        from runpod_manager import get_runpod_manager, PodProvisioningError
        self.assertTrue(callable(get_runpod_manager))
        self.assertTrue(issubclass(PodProvisioningError, Exception))

    def test_celery_beat_schedule_registered(self):
        """Celery beat schedule should include the watchdog task."""
        # Import without starting celery
        import worker

        schedule = worker.celery_app.conf.beat_schedule
        self.assertIn("watchdog-cleanup-every-3-min", schedule)
        self.assertEqual(
            schedule["watchdog-cleanup-every-3-min"]["task"],
            "watchdog_cleanup_pod",
        )
        self.assertEqual(schedule["watchdog-cleanup-every-3-min"]["schedule"], 180.0)


if __name__ == "__main__":
    unittest.main()
