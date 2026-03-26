"""
Metrics collection for health monitoring.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class MetricSnapshot:
    """A single metric snapshot."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """Collected health metrics."""

    # Operation counts
    total_adds: int = 0
    total_updates: int = 0
    total_deletes: int = 0
    total_searches: int = 0

    # Success rates
    add_success_rate: float = 1.0
    update_success_rate: float = 1.0
    delete_success_rate: float = 1.0
    search_success_rate: float = 1.0

    # Performance
    avg_add_time: float = 0.0
    avg_update_time: float = 0.0
    avg_delete_time: float = 0.0
    avg_search_time: float = 0.0

    # Trends (improving, stable, declining)
    add_trend: str = "stable"
    overall_trend: str = "stable"

    # Timestamps
    collected_at: datetime = field(default_factory=lambda: datetime.utcnow())
    window_start: datetime = field(default_factory=lambda: datetime.utcnow())


class MetricsCollector:
    """
    Collects and aggregates metrics from observations.

    ECC-style metrics with:
    - Rolling time windows
    - Success rate tracking
    - Trend analysis
    """

    def __init__(self, window_size_seconds: int = 3600):
        """
        Initialize collector.

        Args:
            window_size_seconds: Time window for metrics aggregation (default: 1 hour)
        """
        self.window_size = timedelta(seconds=window_size_seconds)

        # Metric storage (per operation type)
        self._add_times: deque = deque()
        self._update_times: deque = deque()
        self._delete_times: deque = deque()
        self._search_times: deque = deque()

        self._add_success: deque = deque()
        self._update_success: deque = deque()
        self._delete_success: deque = deque()
        self._search_success: deque = deque()

        # Historical trends
        self._history: List[HealthMetrics] = []

    async def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record an operation metric.

        Args:
            operation: Operation type (add, update, delete, search)
            duration: Duration in seconds
            success: Whether operation succeeded
            timestamp: Operation timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Record to appropriate deque
        if operation == "add":
            self._add_times.append(MetricSnapshot(timestamp, duration))
            self._add_success.append(MetricSnapshot(timestamp, 1.0 if success else 0.0))
        elif operation == "update":
            self._update_times.append(MetricSnapshot(timestamp, duration))
            self._update_success.append(MetricSnapshot(timestamp, 1.0 if success else 0.0))
        elif operation == "delete":
            self._delete_times.append(MetricSnapshot(timestamp, duration))
            self._delete_success.append(MetricSnapshot(timestamp, 1.0 if success else 0.0))
        elif operation == "search":
            self._search_times.append(MetricSnapshot(timestamp, duration))
            self._search_success.append(MetricSnapshot(timestamp, 1.0 if success else 0.0))

        # Clean old data outside window
        self._clean_old_data(timestamp)

    def _clean_old_data(self, now: datetime) -> None:
        """Remove data outside the time window."""
        cutoff = now - self.window_size

        for dq in [
            self._add_times, self._update_times, self._delete_times, self._search_times,
            self._add_success, self._update_success, self._delete_success, self._search_success,
        ]:
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()

    async def get_current_metrics(self) -> HealthMetrics:
        """
        Get current health metrics.

        Returns:
            HealthMetrics with current values
        """
        now = datetime.utcnow()
        self._clean_old_data(now)

        # Calculate averages
        avg_add = self._avg([s.value for s in self._add_times])
        avg_update = self._avg([s.value for s in self._update_times])
        avg_delete = self._avg([s.value for s in self._delete_times])
        avg_search = self._avg([s.value for s in self._search_times])

        # Calculate success rates
        add_success = self._avg([s.value for s in self._add_success])
        update_success = self._avg([s.value for s in self._update_success])
        delete_success = self._avg([s.value for s in self._delete_success])
        search_success = self._avg([s.value for s in self._search_success])

        # Analyze trends
        add_trend = self._analyze_trend(self._add_success)
        overall_trend = self._analyze_overall_trend()

        # Calculate window start
        window_start = now - self.window_size if self._add_times else now

        metrics = HealthMetrics(
            total_adds=len(self._add_times),
            total_updates=len(self._update_times),
            total_deletes=len(self._delete_times),
            total_searches=len(self._search_times),
            add_success_rate=add_success,
            update_success_rate=update_success,
            delete_success_rate=delete_success,
            search_success_rate=search_success,
            avg_add_time=avg_add,
            avg_update_time=avg_update,
            avg_delete_time=avg_delete,
            avg_search_time=avg_search,
            add_trend=add_trend,
            overall_trend=overall_trend,
            collected_at=now,
            window_start=window_start,
        )

        # Store in history
        self._history.append(metrics)

        # Keep only last 100 snapshots
        if len(self._history) > 100:
            self._history.pop(0)

        return metrics

    def _avg(self, values: List[float]) -> float:
        """Calculate average, handling empty lists."""
        return sum(values) / len(values) if values else 0.0

    def _analyze_trend(self, snapshots: deque[MetricSnapshot]) -> str:
        """
        Analyze trend from recent snapshots.

        Returns:
            "improving", "stable", or "declining"
        """
        if len(snapshots) < 10:
            return "stable"

        # Split into first and second half
        mid = len(snapshots) // 2
        first_half = [s.value for s in list(snapshots)[:mid]]
        second_half = [s.value for s in list(snapshots)[mid:]]

        if not first_half or not second_half:
            return "stable"

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        # Determine trend
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"

    def _analyze_overall_trend(self) -> str:
        """Analyze overall system trend."""
        trends = [
            self._analyze_trend(self._add_success),
            self._analyze_trend(self._update_success),
            self._analyze_trend(self._delete_success),
            self._analyze_trend(self._search_success),
        ]

        improving = trends.count("improving")
        declining = trends.count("declining")

        if improving > declining:
            return "improving"
        elif declining > improving:
            return "declining"
        else:
            return "stable"

    def get_history(self) -> List[HealthMetrics]:
        """Get historical metrics."""
        return list(self._history)
