"""
Observation module for mem0 - ECC-style observation capture layer.

This module provides 100% reliable observation capture using hooks,
similar to Everything Claude Code's continuous-learning-v2 system.
"""

from typing import Optional

from .hooks.memory_hook import MemoryObservationHook
from .storage.observation_store import ObservationStore, FileObservationStore
from .storage.buffer import ObservationBuffer
from .filters.privacy_filter import PrivacyFilter
from .collectors.project_detector import ProjectDetector

__all__ = [
    "MemoryObservationHook",
    "ObservationStore",
    "FileObservationStore",
    "ObservationBuffer",
    "PrivacyFilter",
    "ProjectDetector",
    "enable_observation",
]


def enable_observation(
    memory_instance,
    storage_path: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: str = "default",
    user_id: str = "default",
    buffer_size: int = 1000,
    buffer_interval: float = 20.0,
) -> MemoryObservationHook:
    """
    Enable observation capture for a Memory instance.

    This is the main entry point for adding ECC-style observation
    to mem0 Memory instances.

    Args:
        memory_instance: Memory class instance to observe
        storage_path: Path for observation storage (default: ~/.mem0/observations)
        project_id: Project identifier (auto-detected from Git if not provided)
        session_id: Session identifier
        user_id: User identifier
        buffer_size: Observations before auto-flush
        buffer_interval: Seconds before auto-flush

    Returns:
        MemoryObservationHook instance for managing observation

    Example:
        >>> from mem0 import Memory
        >>> from mem0.observation import enable_observation
        >>>
        >>> memory = Memory()
        >>> hook = enable_observation(memory)
        >>>
        >>> # Use memory normally - all operations are captured
        >>> memory.add("User prefers Python", user_id="alice")
        >>>
        >>> # Flush observations on shutdown
        >>> import asyncio
        >>> asyncio.run(hook.flush())
    """
    # Auto-detect project if not provided
    if project_id is None:
        detector = ProjectDetector()
        project_info = detector.detect()
        project_id = project_info.project_id

    # Set up storage backend
    if storage_path is None:
        import os
        storage_path = os.path.expanduser("~/.mem0/observations")

    storage_backend = FileObservationStore(storage_path)

    # Create hook
    hook = MemoryObservationHook(
        project_id=project_id,
        session_id=session_id,
        user_id=user_id,
        storage_backend=storage_backend,
        buffer_size=buffer_size,
        buffer_interval=buffer_interval,
    )

    # Wrap memory instance
    hook.wrap_memory(memory_instance)

    return hook
