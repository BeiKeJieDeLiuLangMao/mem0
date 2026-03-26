"""
In-memory buffer for batch observation writes.

Optimizes I/O by accumulating observations before flushing to storage.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

from ..models import Observation


class ObservationBuffer:
    """
    Thread-safe buffer for observations with auto-flush.

    ECC-style buffering:
    - Flush after N observations (default: 1000)
    - Flush after T seconds (default: 20s)
    - Flush on explicit request
    """

    def __init__(
        self,
        flush_size: int = 1000,
        flush_interval_seconds: float = 20.0,
        storage_backend: Optional[Callable] = None,
    ):
        """
        Initialize buffer.

        Args:
            flush_size: Number of observations before auto-flush
            flush_interval_seconds: Seconds before auto-flush
            storage_backend: Async callable to persist observations
        """
        self.flush_size = flush_size
        self.flush_interval_seconds = flush_interval_seconds
        self.storage_backend = storage_backend

        self._buffer: Deque[Observation] = deque()
        self._last_flush = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def add(self, observation: Observation) -> bool:
        """
        Add observation to buffer, flushing if necessary.

        Args:
            observation: Observation to add

        Returns:
            True if flush was triggered, False otherwise
        """
        async with self._lock:
            self._buffer.append(observation)

            # Check if we need to flush
            should_flush = (
                len(self._buffer) >= self.flush_size or
                self._should_flush_by_time()
            )

            if should_flush:
                await self._flush()
                return True

            return False

    async def flush(self) -> None:
        """Explicitly flush buffer to storage."""
        async with self._lock:
            await self._flush()

    async def _flush(self) -> None:
        """Internal flush implementation."""
        if not self._buffer:
            return

        observations = list(self._buffer)
        self._buffer.clear()
        self._last_flush = datetime.utcnow()

        if self.storage_backend:
            await self.storage_backend(observations)

    def _should_flush_by_time(self) -> bool:
        """Check if enough time has passed for auto-flush."""
        elapsed = (datetime.utcnow() - self._last_flush).total_seconds()
        return elapsed >= self.flush_interval_seconds

    def size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    async def get_all(self) -> List[Observation]:
        """
        Get all buffered observations without clearing.

        Returns:
            List of observations in buffer
        """
        async with self._lock:
            return list(self._buffer)
