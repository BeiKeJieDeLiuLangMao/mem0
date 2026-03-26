"""
Tests for Evolution Layer - Health Monitor and Evolution Tracker.

Following TDD: Test-first approach for health monitoring.
"""

import asyncio
import sys
import pytest
from datetime import datetime, timedelta
from unittest import mock

# Mock mem0 modules that have external dependencies
sys.modules['mem0.client'] = mock.MagicMock()
sys.modules['mem0.client.main'] = mock.MagicMock()
sys.modules['mem0.client.project'] = mock.MagicMock()
sys.modules['mem0.memory'] = mock.MagicMock()
sys.modules['mem0.memory.main'] = mock.MagicMock()
sys.modules['mem0.memory.telemetry'] = mock.MagicMock()

from mem0.evolution.health_monitor import HealthMonitor, HealthStatus, HealthAlert
from mem0.evolution.metrics import HealthMetrics, MetricsCollector


class TestHealthStatusEnum:
    """Test HealthStatus enum values."""

    def test_health_status_values(self):
        """Should have expected status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestHealthMetrics:
    """Test HealthMetrics data model."""

    def test_create_metrics(self):
        """Should create metrics with default values."""
        metrics = HealthMetrics()

        assert metrics.total_adds == 0
        assert metrics.total_updates == 0
        assert metrics.total_deletes == 0
        assert metrics.total_searches == 0
        assert metrics.add_success_rate == 1.0
        assert metrics.update_success_rate == 1.0
        assert metrics.delete_success_rate == 1.0
        assert metrics.search_success_rate == 1.0

    def test_create_metrics_with_values(self):
        """Should create metrics with specified values."""
        metrics = HealthMetrics(
            total_adds=100,
            total_updates=50,
            total_deletes=10,
            total_searches=500,
            add_success_rate=0.95,
            update_success_rate=0.90,
            delete_success_rate=0.98,
            search_success_rate=0.99,
        )

        assert metrics.total_adds == 100
        assert metrics.total_updates == 50
        assert metrics.total_deletes == 10
        assert metrics.total_searches == 500
        assert metrics.add_success_rate == 0.95
        assert metrics.update_success_rate == 0.90
        assert metrics.delete_success_rate == 0.98
        assert metrics.search_success_rate == 0.99

    def test_metrics_to_dict(self):
        """Should convert metrics to dictionary via dataclass."""
        from dataclasses import asdict
        metrics = HealthMetrics(total_adds=50)

        # HealthMetrics is a dataclass; convert using dataclasses.asdict
        result = asdict(metrics)

        assert result["total_adds"] == 50
        assert "add_success_rate" in result


class TestHealthAlert:
    """Test HealthAlert data model."""

    def test_create_alert(self):
        """Should create health alert."""
        alert = HealthAlert(
            severity="warning",
            message="High error rate detected",
            metric_name="add_success_rate",
            current_value=0.5,
            threshold=0.8,
            timestamp=datetime.utcnow(),
        )

        assert alert.severity == "warning"
        assert alert.message == "High error rate detected"
        assert alert.metric_name == "add_success_rate"
        assert alert.current_value == 0.5
        assert alert.threshold == 0.8


class TestMetricsCollector:
    """Test metrics collection."""

    @pytest.mark.asyncio
    async def test_collector_initialization(self):
        """Should initialize collector with zero metrics."""
        collector = MetricsCollector()
        metrics = await collector.get_current_metrics()

        assert metrics.total_adds == 0
        assert metrics.total_updates == 0
        assert metrics.total_deletes == 0
        assert metrics.total_searches == 0

    @pytest.mark.asyncio
    async def test_record_operation_add_success(self):
        """Should record successful add operation."""
        collector = MetricsCollector()
        await collector.record_operation("add", duration=0.1, success=True)

        metrics = await collector.get_current_metrics()
        assert metrics.total_adds == 1
        assert metrics.add_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_record_operation_add_failure(self):
        """Should record failed add operation."""
        collector = MetricsCollector()

        # Record 10 adds, 1 failure
        for _ in range(9):
            await collector.record_operation("add", duration=0.1, success=True)
        await collector.record_operation("add", duration=0.1, success=False)

        metrics = await collector.get_current_metrics()
        assert metrics.total_adds == 10
        assert metrics.add_success_rate == 0.9

    @pytest.mark.asyncio
    async def test_record_operation_update(self):
        """Should record update operations."""
        collector = MetricsCollector()
        await collector.record_operation("update", duration=0.05, success=True)

        metrics = await collector.get_current_metrics()
        assert metrics.total_updates == 1
        assert metrics.update_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_record_operation_delete(self):
        """Should record delete operations."""
        collector = MetricsCollector()
        await collector.record_operation("delete", duration=0.02, success=True)

        metrics = await collector.get_current_metrics()
        assert metrics.total_deletes == 1
        assert metrics.delete_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_record_operation_search(self):
        """Should record search operations."""
        collector = MetricsCollector()
        await collector.record_operation("search", duration=0.3, success=True)

        metrics = await collector.get_current_metrics()
        assert metrics.total_searches == 1
        assert metrics.search_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_record_operation_latency_tracked(self):
        """Should track operation latency."""
        collector = MetricsCollector()
        await collector.record_operation("add", duration=0.15, success=True)

        metrics = await collector.get_current_metrics()
        assert metrics.avg_add_time > 0


class TestHealthMonitor:
    """Test health monitoring."""

    def test_monitor_initialization(self):
        """Should initialize health monitor."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        assert monitor.metrics is not None

    @pytest.mark.asyncio
    async def test_healthy_status(self):
        """Should return HEALTHY when metrics are good."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        # Record successful operations across all types so total_success
        # average (add+update+delete+search)/4 doesn't include 0.0 defaults
        for _ in range(10):
            await collector.record_operation("add", duration=0.1, success=True)
            await collector.record_operation("update", duration=0.1, success=True)
            await collector.record_operation("delete", duration=0.1, success=True)
            await collector.record_operation("search", duration=0.1, success=True)

        status, alerts = await monitor.get_health_status()

        assert status == HealthStatus.HEALTHY
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_unhealthy_status(self):
        """Should return UNHEALTHY when success rate is very low."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        # Record mostly failures (30% success)
        for _ in range(3):
            await collector.record_operation("add", duration=0.1, success=True)
        for _ in range(7):
            await collector.record_operation("add", duration=0.1, success=False)

        status, alerts = await monitor.get_health_status()

        # Should be unhealthy due to low success rate
        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Should generate alerts for low success rates."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        # Create low success rate (below threshold 0.95)
        for _ in range(5):
            await collector.record_operation("add", duration=0.1, success=True)
        for _ in range(5):
            await collector.record_operation("add", duration=0.1, success=False)

        status, alerts = await monitor.get_health_status()

        # Should have alerts for low success rate
        assert len(alerts) > 0

    @pytest.mark.asyncio
    async def test_no_metrics_returns_healthy(self):
        """Should return HEALTHY when no metrics recorded."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        status, alerts = await monitor.get_health_status()

        # No operations = no failures = healthy
        assert status == HealthStatus.HEALTHY
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_multiple_metric_checks(self):
        """Should check multiple metrics for health."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        # Make adds healthy but searches unhealthy
        for _ in range(10):
            await collector.record_operation("add", duration=0.1, success=True)
        for _ in range(10):
            await collector.record_operation("search", duration=0.1, success=False)

        status, alerts = await monitor.get_health_status()

        # Overall status should reflect the unhealthy searches
        assert status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

    @pytest.mark.asyncio
    async def test_trend_calculation(self):
        """Should calculate health trends."""
        collector = MetricsCollector()

        # Record some metrics
        for _ in range(5):
            await collector.record_operation("add", duration=0.1, success=True)

        metrics = await collector.get_current_metrics()

        # Should have trend information
        assert hasattr(metrics, 'overall_trend')
        assert metrics.total_adds == 5

    @pytest.mark.asyncio
    async def test_async_health_check(self):
        """Should support async health checks."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        for _ in range(5):
            await collector.record_operation("add", duration=0.1, success=True)

        status, alerts = await monitor.get_health_status()

        assert status == HealthStatus.HEALTHY
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_get_recent_alerts(self):
        """Should return recent alerts."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        # Generate some alerts
        for _ in range(5):
            await collector.record_operation("add", duration=0.1, success=True)
        for _ in range(10):
            await collector.record_operation("add", duration=0.1, success=False)

        await monitor.get_health_status()

        recent = monitor.get_recent_alerts(limit=5)
        assert isinstance(recent, list)

    @pytest.mark.asyncio
    async def test_clear_alerts(self):
        """Should clear alert history."""
        collector = MetricsCollector()
        monitor = HealthMonitor(collector)

        # Generate some alerts
        for _ in range(5):
            await collector.record_operation("add", duration=0.1, success=True)
        for _ in range(10):
            await collector.record_operation("add", duration=0.1, success=False)

        await monitor.get_health_status()
        monitor.clear_alerts()

        recent = monitor.get_recent_alerts()
        assert len(recent) == 0
