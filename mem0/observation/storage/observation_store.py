"""
Storage backend for observations.
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models import Observation


class ObservationStore(ABC):
    """
    Abstract storage backend for observations.

    ECC-style storage with:
    - JSONL format for append-only logs
    - Project-based isolation
    - Time-based partitioning
    """

    @abstractmethod
    async def add(self, observation: Observation) -> None:
        """Store a single observation."""
        pass

    @abstractmethod
    async def add_batch(self, observations: List[Observation]) -> None:
        """Store multiple observations atomically."""
        pass

    @abstractmethod
    async def get_by_project(
        self,
        project_id: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Retrieve observations by project ID."""
        pass

    @abstractmethod
    async def get_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Retrieve observations by session ID."""
        pass

    @abstractmethod
    async def query(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Query observations with filters."""
        pass

    @abstractmethod
    async def iterate(
        self,
        project_id: Optional[str] = None,
    ) -> AsyncIterator[Observation]:
        """Stream observations efficiently."""
        pass


class FileObservationStore(ObservationStore):
    """
    File-based JSONL storage backend.

    Stores observations in project-specific JSONL files:
    <base_path>/projects/<project_id>/observations.jsonl
    """

    def __init__(self, base_path: str | Path):
        """
        Initialize file store.

        Args:
            base_path: Base directory for observation storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_project_path(self, project_id: str) -> Path:
        """Get storage path for a project."""
        project_dir = self.base_path / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir / "observations.jsonl"

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get storage path for an agent (replaces project-based path)."""
        agent_dir = self.base_path / "agents" / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir / "observations.jsonl"

    async def add(self, observation: Observation) -> None:
        """Append observation to project's JSONL file."""
        project_path = self._get_project_path(observation.project_id)

        # Append to file
        with open(project_path, "a", encoding="utf-8") as f:
            f.write(observation.to_jsonl() + "\n")

    async def add_batch(self, observations: List[Observation]) -> None:
        """Atomically append multiple observations."""
        # Group by project
        by_project: Dict[str, List[Observation]] = {}
        for obs in observations:
            if obs.project_id not in by_project:
                by_project[obs.project_id] = []
            by_project[obs.project_id].append(obs)

        # Write each project's file
        for project_id, obs_list in by_project.items():
            project_path = self._get_project_path(project_id)
            with open(project_path, "a", encoding="utf-8") as f:
                for obs in obs_list:
                    f.write(obs.to_jsonl() + "\n")

    async def get_by_project(
        self,
        project_id: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Read observations from project's JSONL file."""
        project_path = self._get_project_path(project_id)

        if not project_path.exists():
            return []

        observations = []
        with open(project_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    import json
                    data = json.loads(line)
                    obs = Observation.from_dict(data)
                    observations.append(obs)

                    if limit and len(observations) >= limit:
                        break

        return observations

    async def get_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Scan all projects for session observations."""
        # This is inefficient - in production, use an index
        observations = []

        projects_dir = self.base_path / "projects"
        if not projects_dir.exists():
            return []

        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                project_obs = await self.get_by_project(project_dir.name)
                for obs in project_obs:
                    if obs.session_id == session_id:
                        observations.append(obs)
                        if limit and len(observations) >= limit:
                            return observations

        return observations

    async def query(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Observation]:
        """Query observations with filters."""
        # Simple implementation - scan all projects
        results = []

        projects_dir = self.base_path / "projects"
        if not projects_dir.exists():
            return []

        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                project_obs = await self.get_by_project(project_dir.name)

                for obs in project_obs:
                    match = True
                    for key, value in filters.items():
                        if getattr(obs, key, None) != value:
                            match = False
                            break

                    if match:
                        results.append(obs)
                        if limit and len(results) >= limit:
                            return results

        return results

    async def iterate(
        self,
        project_id: Optional[str] = None,
    ) -> AsyncIterator[Observation]:
        """Stream observations from storage."""
        if project_id:
            projects = [project_id]
        else:
            projects_dir = self.base_path / "projects"
            if not projects_dir.exists():
                return
            projects = [d.name for d in projects_dir.iterdir() if d.is_dir()]

        for pid in projects:
            project_path = self._get_project_path(pid)
            if not project_path.exists():
                continue

            with open(project_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        import json
                        data = json.loads(line)
                        yield Observation.from_dict(data)
