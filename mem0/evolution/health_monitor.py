"""
Health monitoring for memory operations.

Provides real-time health status and alerts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .metrics import HealthMetrics, MetricsCollector


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthAlert:
    """A health alert."""
    severity: str  # info, warning, critical
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: Any  # datetime


class HealthMonitor:
    """
    Monitors system health and generates alerts.

    ECC-style health monitoring:
    - Real-time status tracking
    - Threshold-based alerting
    - Trend-based warnings
    """

    # Alert thresholds
    THRESHOLDS = {
        "min_success_rate": 0.95,
        "max_avg_time": 5.0,  # seconds
        "min_operation_count": 10,  # minimum for valid metrics
    }

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize health monitor.

        Args:
            metrics_collector: MetricsCollector to monitor
        """
        self.metrics = metrics_collector or MetricsCollector()
        self._alerts: List[HealthAlert] = []

    async def get_health_status(self) -> tuple[HealthStatus, List[HealthAlert]]:
        """
        Get current health status with alerts.

        Returns:
            Tuple of (HealthStatus, List[HealthAlert])
        """
        metrics = await self.metrics.get_current_metrics()
        alerts = self._check_thresholds(metrics)

        # Determine overall status (use same metrics instance, don't call again)
        status = self._determine_status(metrics, alerts)

        self._alerts.extend(alerts)
        return status, alerts

    def _check_thresholds(self, metrics: HealthMetrics) -> List[HealthAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []

        # Check success rates
        if metrics.total_adds >= self.THRESHOLDS["min_operation_count"]:
            if metrics.add_success_rate < self.THRESHOLDS["min_success_rate"]:
                alerts.append(HealthAlert(
                    severity="critical",
                    message=f"Add success rate below threshold: {metrics.add_success_rate:.2f}",
                    metric_name="add_success_rate",
                    current_value=metrics.add_success_rate,
                    threshold=self.THRESHOLDS["min_success_rate"],
                    timestamp=metrics.collected_at,
                ))

        # Check operation times
        if metrics.total_adds > 0:
            if metrics.avg_add_time > self.THRESHOLDS["max_avg_time"]:
                alerts.append(HealthAlert(
                    severity="warning",
                    message=f"Add operation time slow: {metrics.avg_add_time:.2f}s",
                    metric_name="avg_add_time",
                    current_value=metrics.avg_add_time,
                    threshold=self.THRESHOLDS["max_avg_time"],
                    timestamp=metrics.collected_at,
                ))

        # Check for declining trends
        if metrics.overall_trend == "declining":
            alerts.append(HealthAlert(
                severity="warning",
                message=f"Overall system performance is declining",
                metric_name="overall_trend",
                current_value=0.0,
                threshold=0.0,
                timestamp=metrics.collected_at,
            ))

        return alerts

    def _determine_status(
        self,
        metrics: HealthMetrics,
        alerts: List[HealthAlert],
    ) -> HealthStatus:
        """Determine overall health status."""
        # Check for critical alerts
        critical = [a for a in alerts if a.severity == "critical"]
        if critical:
            return HealthStatus.UNHEALTHY

        # Check for warning alerts
        warnings = [a for a in alerts if a.severity == "warning"]
        if len(warnings) >= 2:
            return HealthStatus.DEGRADED

        # Check if we have enough data
        total_ops = (
            metrics.total_adds +
            metrics.total_updates +
            metrics.total_deletes +
            metrics.total_searches
        )

        if total_ops < self.THRESHOLDS["min_operation_count"]:
            # Not enough data - assume healthy
            return HealthStatus.HEALTHY

        # Check overall success rate
        total_success = (
            metrics.add_success_rate +
            metrics.update_success_rate +
            metrics.delete_success_rate +
            metrics.search_success_rate
        ) / 4

        if total_success >= self.THRESHOLDS["min_success_rate"]:
            return HealthStatus.HEALTHY
        elif total_success >= 0.8:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    def get_recent_alerts(self, limit: int = 10) -> List[HealthAlert]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self._alerts.clear()
