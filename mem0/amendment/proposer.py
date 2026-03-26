"""
Amendment proposal generator.

Analyzes observations and patterns to propose memory improvements.
"""

import asyncio
from typing import Any, Dict, List, Optional

from .models import AmendmentProposal, AmendmentType, AmendmentStatus
from ..learning.pattern import Pattern
from ..observation.models import Observation, ObservationType


class AmendmentProposer:
    """
    Proposes amendments to memories based on analysis.

    ECC-style proposal generation:
    - Analyze patterns for improvement opportunities
    - Generate evidence-backed proposals
    - Confidence scoring
    """

    def __init__(self, min_confidence: float = 0.7):
        """
        initialize proposer.

        Args:
            min_confidence: Minimum confidence for proposals
        """
        self.min_confidence = min_confidence
        self._generators = {
            AmendmentType.CONTENT_UPDATE: self._propose_content_update,
            AmendmentType.MERGE: self._propose_merge,
            AmendmentType.DEPRECATE: self._propose_deprecate,
            AmendmentType.REPRIORITIZE: self._propose_reprioritize,
        }

    async def propose_amendments(
        self,
        observations: List[Observation],
        patterns: List[Pattern],
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AmendmentProposal]:
        """
        Generate amendment proposals from analysis.

        Args:
            observations: Recent observations to analyze
            patterns: Detected patterns
            memories: Current memory states (if available)

        Returns:
            List of amendment proposals
        """
        proposals = []

        # Analyze patterns for amendments
        for pattern in patterns:
            if pattern.confidence < self.min_confidence:
                continue

            pattern_proposals = await self._analyze_pattern(pattern, observations)
            proposals.extend(pattern_proposals)

        # Analyze observations for issues
        observation_proposals = await self._analyze_observations(observations, memories)
        proposals.extend(observation_proposals)

        # Filter by confidence
        proposals = [p for p in proposals if p.confidence >= self.min_confidence]

        # Sort by confidence and impact
        proposals.sort(
            key=lambda p: (p.confidence, abs(p.expected_impact)),
            reverse=True,
        )

        return proposals

    async def _analyze_pattern(
        self,
        pattern: Pattern,
        observations: List[Observation],
    ) -> List[AmendmentProposal]:
        """Analyze a pattern for amendment opportunities."""
        proposals = []

        if pattern.pattern_type.value == "user_preference":
            # Check if preference conflicts with current memory
            proposal = await self._propose_preference_alignment(pattern)
            if proposal:
                proposals.append(proposal)

        elif pattern.pattern_type.value == "error_recovery":
            # Propose updates to prevent errors
            proposal = await self._propose_error_prevention(pattern)
            if proposal:
                proposals.append(proposal)

        return proposals

    async def _analyze_observations(
        self,
        observations: List[Observation],
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AmendmentProposal]:
        """Analyze observations for amendment opportunities."""
        proposals = []

        # Look for negative feedback
        for obs in observations:
            if obs.event_type == ObservationType.FEEDBACK:
                rating = obs.data.get("rating")
                memory_id = obs.data.get("memory_id")

                if rating is not None and rating < 0.5 and memory_id:
                    # Propose deprecation or update
                    proposal = AmendmentProposal(
                        amendment_type=AmendmentType.DEPRECATE,
                        memory_id=memory_id,
                        reasoning=f"Low user rating ({rating}) indicates this memory may be outdated or incorrect",
                        evidence_observation_ids=[obs.id],
                        confidence=0.8,
                        expected_impact=-0.3,
                        impact_description="Removing low-quality memory",
                        project_id=obs.project_id,
                    )
                    proposals.append(proposal)

        return proposals

    async def _propose_content_update(
        self,
        memory_id: str,
        current_content: Dict[str, Any],
        new_content: Dict[str, Any],
        reasoning: str,
        evidence: List[str],
        confidence: float,
        project_id: str = "global",
    ) -> AmendmentProposal:
        """Propose a content update."""
        return AmendmentProposal(
            amendment_type=AmendmentType.CONTENT_UPDATE,
            memory_id=memory_id,
            current_content=current_content,
            proposed_change=new_content,
            reasoning=reasoning,
            evidence_observation_ids=evidence,
            confidence=confidence,
            expected_impact=0.5,
            impact_description="Update memory with new information",
            project_id=project_id,
        )

    async def _propose_merge(
        self,
        memory_ids: List[str],
        reasoning: str,
        confidence: float,
        project_id: str = "global",
    ) -> AmendmentProposal:
        """Propose merging similar memories."""
        return AmendmentProposal(
            amendment_type=AmendmentType.MERGE,
            memory_id=memory_ids[0],
            proposed_change={"merge_with": memory_ids[1:]},
            reasoning=reasoning,
            confidence=confidence,
            expected_impact=0.3,
            impact_description="Consolidate duplicate/similar memories",
            project_id=project_id,
        )

    async def _propose_deprecate(
        self,
        memory_id: str,
        reasoning: str,
        evidence: List[str],
        confidence: float,
        project_id: str = "global",
    ) -> AmendmentProposal:
        """Propose deprecating a memory."""
        return AmendmentProposal(
            amendment_type=AmendmentType.DEPRECATE,
            memory_id=memory_id,
            proposed_change={"deprecated": True},
            reasoning=reasoning,
            evidence_observation_ids=evidence,
            confidence=confidence,
            expected_impact=-0.2,
            impact_description="Remove outdated memory",
            project_id=project_id,
        )

    async def _propose_reprioritize(
        self,
        memory_id: str,
        new_score: float,
        reasoning: str,
        confidence: float,
        project_id: str = "global",
    ) -> AmendmentProposal:
        """Propose changing memory priority."""
        return AmendmentProposal(
            amendment_type=AmendmentType.REPRIORITIZE,
            memory_id=memory_id,
            proposed_change={"score": new_score},
            reasoning=reasoning,
            confidence=confidence,
            expected_impact=0.2,
            impact_description=f"Change memory score to {new_score}",
            project_id=project_id,
        )

    async def _propose_preference_alignment(
        self,
        pattern,
    ) -> Optional[AmendmentProposal]:
        """Propose updating memory to align with detected preference."""
        # This would analyze the preference pattern and propose changes
        return None

    async def _propose_error_prevention(
        self,
        pattern,
    ) -> Optional[AmendmentProposal]:
        """Propose updates to prevent recurring errors."""
        content = pattern.pattern_content
        error_type = content.get("error_type")
        method = content.get("failed_method")

        if error_type and method:
            return AmendmentProposal(
                amendment_type=AmendmentType.CONTENT_UPDATE,
                memory_id=f"error_pattern_{method}",
                current_content={},
                proposed_change={
                    "error_type": error_type,
                    "prevention": "Apply retry logic",
                },
                reasoning=f"Recurring {error_type} in {method} - preventive action needed",
                evidence_observation_ids=pattern.evidence.observation_ids[:3],
                confidence=pattern.confidence,
                expected_impact=0.4,
                impact_description=f"Prevent {error_type} errors",
                project_id=pattern.project_id,
            )

        return None
