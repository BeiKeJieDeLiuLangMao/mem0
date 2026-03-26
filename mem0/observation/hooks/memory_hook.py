"""
Memory observation hook for ECC-style 100% reliable capture.

Wraps Memory class methods to capture all operations as observations.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional

from ..models import Observation, ObservationType
from ..filters.privacy_filter import PrivacyFilter
from ..storage.buffer import ObservationBuffer
from ..storage.observation_store import ObservationStore


class MemoryObservationHook:
    """
    Hook wrapper for Memory class operations.

    Captures all memory operations (add, update, delete, search) as observations.
    Provides 100% reliable capture via method wrapping.
    """

    def __init__(
        self,
        project_id: str = "global",
        session_id: str = "default",
        user_id: str = "default",
        storage_backend: Optional[ObservationStore] = None,
        buffer_size: int = 1000,
        buffer_interval: float = 20.0,
    ):
        """
        Initialize observation hook.

        Args:
            project_id: Project identifier (Git remote URL hash)
            session_id: Session identifier
            user_id: User identifier
            storage_backend: Storage backend for persistence
            buffer_size: Observations before auto-flush
            buffer_interval: Seconds before auto-flush
        """
        self.project_id = project_id
        self.session_id = session_id
        self.user_id = user_id

        # Privacy filter
        self.privacy_filter = PrivacyFilter()

        # Buffer setup
        async def storage_func(observations):
            if storage_backend:
                await storage_backend.add_batch(observations)

        self.buffer = ObservationBuffer(
            flush_size=buffer_size,
            flush_interval_seconds=buffer_interval,
            storage_backend=storage_func,
        )

    def wrap_memory(self, memory_instance: Any) -> Any:
        """
        Wrap a Memory instance with observation hooks.

        Args:
            memory_instance: Memory class instance to wrap

        Returns:
            Wrapped Memory with observed methods
        """
        # Wrap key methods
        methods_to_wrap = ["add", "update", "delete", "get", "search", "get_all"]

        for method_name in methods_to_wrap:
            if hasattr(memory_instance, method_name):
                original_method = getattr(memory_instance, method_name)
                wrapped = self._create_wrapper(method_name, original_method)
                setattr(memory_instance, method_name, wrapped)

        return memory_instance

    def _create_wrapper(
        self,
        method_name: str,
        original_method: Callable,
    ) -> Callable:
        """Create an observation wrapper for a method."""

        @functools.wraps(original_method)
        async def wrapped(*args, **kwargs):
            # Create INITIATED observation
            initiated_obs = self._create_observation(
                event_type=self._get_initiated_type(method_name),
                data={
                    "method": method_name,
                    "args": self._sanitize_args(args),
                    "kwargs": self._sanitize_kwargs(kwargs),
                },
            )
            await self.buffer.add(initiated_obs)

            try:
                # Execute original method
                result = await original_method(*args, **kwargs)

                # Create COMPLETED observation
                completed_obs = self._create_observation(
                    event_type=self._get_completed_type(method_name),
                    data={
                        "method": method_name,
                        "result": self._sanitize_result(result),
                        "success": True,
                    },
                )
                await self.buffer.add(completed_obs)

                return result

            except Exception as e:
                # Create FAILED observation
                failed_obs = self._create_observation(
                    event_type=self._get_completed_type(method_name),
                    data={
                        "method": method_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "success": False,
                    },
                )
                await self.buffer.add(failed_obs)
                raise

        return wrapped

    def _create_observation(
        self,
        event_type: ObservationType,
        data: Dict[str, Any],
    ) -> Observation:
        """Create an observation with common metadata."""
        return Observation(
            event_type=event_type,
            project_id=self.project_id,
            session_id=self.session_id,
            user_id=self.user_id,
            data=data,
            metadata={
                "capture_method": "hook_wrapper",
            },
        )

    def _sanitize_args(self, args: tuple) -> list:
        """Sanitize function arguments for storage."""
        sanitized = []
        for arg in args:
            # Skip memory self-reference
            if hasattr(arg, "__class__") and "Memory" in arg.__class__.__name__:
                sanitized.append("<Memory instance>")
            else:
                has_pii, redacted = self.privacy_filter.scan(arg)
                sanitized.append(redacted if has_pii else arg)
        return sanitized

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """Sanitize keyword arguments for storage."""
        sanitized = {}
        for key, value in kwargs.items():
            has_pii, redacted = self.privacy_filter.scan(value)
            sanitized[key] = redacted if has_pii else value
        return sanitized

    def _sanitize_result(self, result: Any) -> Any:
        """Sanitize method result for storage."""
        has_pii, redacted = self.privacy_filter.scan(result)
        return redacted if has_pii else result

    def _get_initiated_type(self, method_name: str) -> ObservationType:
        """Map method name to INITIATED event type."""
        mapping = {
            "add": ObservationType.ADD_INITIATED,
            "update": ObservationType.UPDATE_INITIATED,
            "delete": ObservationType.DELETE_INITIATED,
            "search": ObservationType.SEARCH_INITIATED,
            "get": ObservationType.SEARCH_INITIATED,
            "get_all": ObservationType.SEARCH_INITIATED,
        }
        return mapping.get(method_name, ObservationType.SEARCH_INITIATED)

    def _get_completed_type(self, method_name: str) -> ObservationType:
        """Map method name to COMPLETED event type."""
        mapping = {
            "add": ObservationType.ADD_COMPLETED,
            "update": ObservationType.UPDATE_COMPLETED,
            "delete": ObservationType.DELETE_COMPLETED,
            "search": ObservationType.SEARCH_COMPLETED,
            "get": ObservationType.SEARCH_COMPLETED,
            "get_all": ObservationType.SEARCH_COMPLETED,
        }
        return mapping.get(method_name, ObservationType.SEARCH_COMPLETED)

    async def flush(self) -> None:
        """Manually flush observation buffer."""
        await self.buffer.flush()

    async def record_feedback(
        self,
        memory_id: str,
        feedback: str,
        rating: Optional[float] = None,
    ) -> None:
        """
        Record user feedback on a memory.

        Args:
            memory_id: ID of the memory being rated
            feedback: Feedback text
            rating: Optional numeric rating (0-1)
        """
        has_pii, redacted_feedback = self.privacy_filter.scan(feedback)

        obs = self._create_observation(
            event_type=ObservationType.FEEDBACK,
            data={
                "memory_id": memory_id,
                "feedback": redacted_feedback,
                "rating": rating,
            },
        )
        await self.buffer.add(obs)
