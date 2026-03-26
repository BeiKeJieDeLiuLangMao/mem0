"""
Skill extraction from patterns.

⚠️ DEPRECATED: This module is deprecated in favor of mem0.learning.enhanced_skill_extractor.
The enhanced version includes:
- LLM-powered skill extraction
- Natural language skill descriptions
- Context-aware trigger phrases

Transforms detected patterns into reusable skill candidates.
"""

from typing import Any, Dict, List

from .pattern import Pattern, SkillCandidate, PatternType


class SkillExtractor:
    """
    Extracts skill candidates from patterns.

    ECC-style skill extraction:
    - Synthesize patterns into actionable skills
    - Generate trigger phrases
    - Create instructions from patterns
    """

    def __init__(self):
        """Initialize skill extractor."""
        self._skill_generators = {
            PatternType.WORKFLOW_SEQUENCE: self._generate_workflow_skill,
            PatternType.USER_PREFERENCE: self._generate_preference_skill,
            PatternType.ERROR_RECOVERY: self._generate_recovery_skill,
        }

    async def extract_skills(
        self,
        patterns: List[Pattern],
    ) -> List[SkillCandidate]:
        """
        Extract skill candidates from patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            List of skill candidates
        """
        skills = []

        for pattern in patterns:
            if pattern.pattern_type in self._skill_generators:
                generator = self._skill_generators[pattern.pattern_type]
                skill = await generator(pattern)
                if skill:
                    skills.append(skill)

        return skills

    async def _generate_workflow_skill(
        self,
        pattern: Pattern,
    ) -> SkillCandidate:
        """Generate skill from workflow pattern."""
        sequence = pattern.pattern_content.get("sequence", [])

        # Extract method names from sequence
        methods = [obs.data.get("method", "") for obs in sequence if obs.data.get("method")]

        # Generate trigger phrases
        trigger_phrases = [
            f"repeat the workflow",
            f"follow the sequence",
            f"execute the pattern",
        ]

        # Generate instructions
        instructions = self._generate_workflow_instructions(methods)

        skill = SkillCandidate(
            name=f"Workflow: {' -> '.join(methods[:3])}",
            description=f"Automated workflow pattern (confidence: {pattern.confidence:.2f})",
            trigger_phrases=trigger_phrases,
            source_pattern_ids=[pattern.id],
            confidence=pattern.confidence,
            instructions=instructions,
            examples=[
                {
                    "context": "User requests workflow execution",
                    "workflow": methods,
                    "frequency": pattern.frequency,
                }
            ],
            project_id=pattern.project_id,
            tags=["workflow", "automation", "pattern"],
        )

        return skill

    async def _generate_preference_skill(
        self,
        pattern: Pattern,
    ) -> SkillCandidate:
        """Generate skill from user preference pattern."""
        content = pattern.pattern_content

        trigger_phrases = [
            f"remember my preference",
            f"what do I prefer",
            f"my choice",
        ]

        instructions = f"""
User preference detected with {pattern.confidence:.2f} confidence:
- Memory ID: {content.get('memory_id', 'unknown')[:8]}
- Average rating: {content.get('average_rating', 0):.2f}/1.0
- Based on {content.get('rating_count', 0)} ratings

When user asks about preferences related to this memory,
prioritize this information.
        """.strip()

        skill = SkillCandidate(
            name=f"Preference: {content.get('memory_id', 'unknown')[:8]}",
            description=f"User preference pattern (rating: {content.get('average_rating', 0):.2f})",
            trigger_phrases=trigger_phrases,
            source_pattern_ids=[pattern.id],
            confidence=pattern.confidence,
            instructions=instructions,
            project_id=pattern.project_id,
            tags=["preference", "user-behavior"],
        )

        return skill

    async def _generate_recovery_skill(
        self,
        pattern: Pattern,
    ) -> SkillCandidate:
        """Generate skill from error recovery pattern."""
        content = pattern.pattern_content

        trigger_phrases = [
            f"handle {content.get('error_type', '')} error",
            f"fix {content.get('failed_method', '')} failure",
            f"recover from error",
        ]

        instructions = f"""
Error recovery pattern for {content.get('failed_method', 'unknown')}:

When encountering {content.get('error_type', 'error')}:
1. Recovery typically takes {content.get('recovery_time_seconds', 0):.1f} seconds
2. Pattern observed with {pattern.confidence:.2f} confidence
3. Retry the operation with adjusted parameters

This is based on observed user behavior, not tested code.
        """.strip()

        skill = SkillCandidate(
            name=f"Recovery: {content.get('error_type', 'error')}",
            description=f"Error recovery pattern for {content.get('failed_method', 'unknown')}",
            trigger_phrases=trigger_phrases,
            source_pattern_ids=[pattern.id],
            confidence=pattern.confidence,
            instructions=instructions,
            project_id=pattern.project_id,
            tags=["error-handling", "recovery", "pattern"],
        )

        return skill

    def _generate_workflow_instructions(self, methods: List[str]) -> str:
        """Generate step-by-step instructions from workflow methods."""
        if not methods:
            return "No workflow steps identified."

        steps = []
        for i, method in enumerate(methods, 1):
            steps.append(f"{i}. Execute {method}")

        return """
Workflow pattern detected:
When user requests this workflow, execute the following steps:

{steps}

Note: This pattern is based on observed behavior with {count} occurrences.
        """.format(steps="\n".join(steps), count=len(methods)).strip()
