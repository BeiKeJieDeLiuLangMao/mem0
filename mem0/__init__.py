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
from mem0.ailearn.enhanced import enable_enhanced_ailearn, EnhancedAILearn

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
    # AI Learning Integration (Basic - Deprecated)
    "enable_ailearn",
    "Mem0AILearn",
    # AI Learning Integration (Enhanced - Recommended)
    "enable_enhanced_ailearn",
    "EnhancedAILearn",
]
