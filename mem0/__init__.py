import importlib.metadata

__version__ = importlib.metadata.version("mem0ai")

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa

# AI Learning Architecture
from mem0.observation import (
    enable_observation,
    MemoryObservationHook,
    ObservationStore,
    FileObservationStore,
    ObservationBuffer,
    PrivacyFilter,
    ProjectDetector,
)
from mem0.ailearn import enable_ailearn, Mem0AILearn

__all__ = [
    # Core mem0
    "AsyncMemoryClient",
    "MemoryClient",
    "AsyncMemory",
    "Memory",
    # AI Learning Observation Layer
    "enable_observation",
    "MemoryObservationHook",
    "ObservationStore",
    "FileObservationStore",
    "ObservationBuffer",
    "PrivacyFilter",
    "ProjectDetector",
    # AI Learning Integration
    "enable_ailearn",
    "Mem0AILearn",
]
