"""
Enhanced Skill Extraction with LLM

使用 LLM 分析对话上下文，提取高质量的技能描述。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .pattern import Pattern, SkillCandidate, PatternType

logger = logging.getLogger(__name__)


class LLMSkillExtractor:
    """
    LLM-powered skill extractor.

    Features:
    - Analyzes conversation context to extract skills
    - Generates natural language skill descriptions
    - Identifies trigger phrases from user queries
    - Provides examples from actual conversations
    """

    def __init__(
        self,
        llm_client: Any,  # OpenAI client
        model: str = "gpt-4.1-nano-2025-04-14",
    ):
        """
        Initialize LLM skill extractor.

        Args:
            llm_client: OpenAI-compatible client
            model: Model to use for extraction
        """
        self.llm_client = llm_client
        self.model = model

    async def extract_skills(
        self,
        patterns: List[Pattern],
        turn_contexts: Dict[str, List[Dict[str, Any]]],  # session_id -> turns
    ) -> List[SkillCandidate]:
        """
        Extract skills from patterns with LLM analysis.

        Args:
            patterns: Detected patterns
            turn_contexts: Conversation turns for context

        Returns:
            List of skill candidates
        """
        skills = []

        for pattern in patterns:
            if pattern.pattern_type == PatternType.WORKFLOW_SEQUENCE:
                skill = await self._extract_workflow_skill(pattern, turn_contexts)
                if skill:
                    skills.append(skill)
            elif pattern.pattern_type == PatternType.USER_PREFERENCE:
                skill = await self._extract_preference_skill(pattern, turn_contexts)
                if skill:
                    skills.append(skill)
            elif pattern.pattern_type == PatternType.DECISION_CRITERIA:
                skill = await self._extract_decision_skill(pattern, turn_contexts)
                if skill:
                    skills.append(skill)

        return skills

    async def _extract_workflow_skill(
        self,
        pattern: Pattern,
        turn_contexts: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[SkillCandidate]:
        """
        Extract workflow skill using LLM.

        Analyzes:
        1. What the user was trying to accomplish
        2. The steps they took
        3. The outcome
        4. When this workflow should be applied
        """
        # Get relevant turns
        session_id = pattern.metadata.get("session_id")
        turns = turn_contexts.get(session_id, []) if session_id else []

        if not turns:
            # Fallback to rule-based extraction
            return None

        # Build analysis prompt
        prompt = self._build_workflow_analysis_prompt(pattern, turns)

        try:
            # Call LLM
            response = await self._call_llm(prompt)
            skill_data = json.loads(response)

            # Validate and build skill
            return SkillCandidate(
                name=skill_data.get("name", "Unnamed Workflow"),
                description=skill_data.get("description", ""),
                trigger_phrases=skill_data.get("trigger_phrases", []),
                source_pattern_ids=[pattern.id],
                confidence=pattern.confidence,
                instructions=skill_data.get("instructions", ""),
                examples=[{
                    "context": skill_data.get("context", ""),
                    "workflow": skill_data.get("workflow", ""),
                    "outcome": skill_data.get("outcome", ""),
                }],
                project_id=pattern.project_id,
                agent_id=pattern.agent_id,
                tags=["workflow", "llm-extracted"],
            )

        except Exception as e:
            logger.error(f"Failed to extract workflow skill: {e}")
            return None

    def _build_workflow_analysis_prompt(
        self,
        pattern: Pattern,
        turns: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for workflow analysis."""

        # Extract conversation
        conversation = self._format_conversation(turns)

        return f"""Analyze the following conversation and extract a reusable skill.

**Conversation:**
{conversation}

**Detected Pattern:**
- Type: {pattern.pattern_type.value}
- Frequency: {pattern.frequency} occurrences
- Confidence: {pattern.confidence:.2f}

**Instructions:**
1. Identify what the user was trying to accomplish
2. Extract the key steps in their approach
3. Identify trigger phrases that would indicate this skill is needed
4. Write clear instructions for reproducing this workflow
5. Provide context about when this workflow is useful

**Response Format (JSON):**
{{
    "name": "Brief skill name (5 words or less)",
    "description": "One-sentence description of what this skill does",
    "trigger_phrases": ["phrase1", "phrase2", "phrase3"],
    "instructions": "Step-by-step instructions for applying this skill",
    "context": "When and why this workflow is useful",
    "workflow": "Summary of the workflow steps",
    "outcome": "Expected result of applying this workflow"
}}

Respond ONLY with valid JSON, no additional text."""

    async def _extract_preference_skill(
        self,
        pattern: Pattern,
        turn_contexts: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[SkillCandidate]:
        """Extract user preference skill."""
        # Get relevant turns
        session_id = pattern.metadata.get("session_id")
        turns = turn_contexts.get(session_id, []) if session_id else []

        if not turns:
            return None

        prompt = self._build_preference_analysis_prompt(pattern, turns)

        try:
            response = await self._call_llm(prompt)
            skill_data = json.loads(response)

            return SkillCandidate(
                name=f"Preference: {skill_data.get('preference_topic', 'Unknown')}",
                description=skill_data.get("description", ""),
                trigger_phrases=skill_data.get("trigger_phrases", [
                    "remember my preference",
                    "what do I prefer",
                    "my choice",
                ]),
                source_pattern_ids=[pattern.id],
                confidence=pattern.confidence,
                instructions=skill_data.get("instructions", ""),
                examples=[{
                    "preference": skill_data.get("preference", ""),
                    "rationale": skill_data.get("rationale", ""),
                }],
                project_id=pattern.project_id,
                agent_id=pattern.agent_id,
                tags=["preference", "llm-extracted"],
            )

        except Exception as e:
            logger.error(f"Failed to extract preference skill: {e}")
            return None

    def _build_preference_analysis_prompt(
        self,
        pattern: Pattern,
        turns: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for preference analysis."""

        conversation = self._format_conversation(turns)

        return f"""Analyze the conversation to identify user preferences.

**Conversation:**
{conversation}

**Pattern Data:**
- Frequency: {pattern.frequency}
- Confidence: {pattern.confidence:.2f}
- Content: {pattern.pattern_content}

**Instructions:**
1. Identify what the user consistently prefers
2. Understand the rationale behind their preference
3. Determine when this preference should be applied

**Response Format (JSON):**
{{
    "preference_topic": "What the preference is about",
    "description": "Clear description of the preference",
    "instructions": "How to apply this preference in future interactions",
    "trigger_phrases": ["phrases that indicate this preference is relevant"],
    "rationale": "Why the user prefers this"
}}

Respond ONLY with valid JSON."""

    async def _extract_decision_skill(
        self,
        pattern: Pattern,
        turn_contexts: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[SkillCandidate]:
        """Extract decision-making skill."""
        # TODO: Implement decision skill extraction
        return None

    def _format_conversation(self, turns: List[Dict[str, Any]]) -> str:
        """Format turns for LLM prompt."""
        formatted = []

        for i, turn in enumerate(turns, 1):
            turn_id = turn.get("id", "")[:8]
            messages = turn.get("messages", [])

            formatted.append(f"\n--- Turn {i} ({turn_id}) ---")

            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "user":
                    formatted.append(f"User: {content}")
                elif role == "assistant":
                    # Truncate long assistant messages
                    if len(content) > 500:
                        content = content[:500] + "..."
                    formatted.append(f"Assistant: {content}")

                # Include tool calls
                if msg.get("tool_calls"):
                    for call in msg["tool_calls"]:
                        func_name = call.get("function", {}).get("name", "unknown")
                        formatted.append(f"[Tool: {func_name}]")

        return "\n".join(formatted)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing user behavior and extracting reusable skills and patterns. Always respond with valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


class EnhancedSkillExtractor:
    """
    Enhanced skill extractor combining rule-based and LLM-based extraction.

    Fallback strategy:
    1. Try LLM extraction first
    2. Fall back to rule-based if LLM unavailable
    3. Combine results from both methods
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4.1-nano-2025-04-14",
    ):
        """
        Initialize enhanced skill extractor.

        Args:
            llm_client: LLM client (optional)
            model: Model name for LLM
        """
        self.llm_client = llm_client
        self.model = model

        # Import rule-based extractor as fallback
        from .skill_extractor import SkillExtractor as RuleBasedSkillExtractor
        self.rule_based = RuleBasedSkillExtractor()

        # Initialize LLM-based extractor if client available
        self.llm_based = LLMSkillExtractor(llm_client, model) if llm_client else None

    async def extract_skills(
        self,
        patterns: List[Pattern],
        turn_contexts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[SkillCandidate]:
        """
        Extract skills using best available method.

        Args:
            patterns: Detected patterns
            turn_contexts: Conversation turns (for LLM extraction)

        Returns:
            List of skill candidates
        """
        skills = []

        # Try LLM extraction first
        if self.llm_based and turn_contexts:
            try:
                llm_skills = await self.llm_based.extract_skills(patterns, turn_contexts)
                skills.extend(llm_skills)
            except Exception as e:
                logger.warning(f"LLM skill extraction failed, falling back to rule-based: {e}")

        # Fallback to rule-based for remaining patterns
        rule_skills = await self.rule_based.extract_skills(patterns)
        skills.extend(rule_skills)

        # Deduplicate by name
        seen_names = set()
        unique_skills = []
        for skill in skills:
            if skill.name not in seen_names:
                seen_names.add(skill.name)
                unique_skills.append(skill)

        return unique_skills
