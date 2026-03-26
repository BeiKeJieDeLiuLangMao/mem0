"""
Skill Health Dashboard — generates health reports for skills/instincts.

ECC-style dashboard providing:
- Per-skill health scoring (0–100)
- Aggregate health metrics
- Visualization data (bar chart format for dashboards)
"""

from typing import Any, Dict, List, Optional

from ..instincts.instincts import Instinct


class SkillHealthDashboard:
    """
    Generate skill health reports from instinct data.

    Health score formula:
        disabled → 0
        never applied → 50 (neutral baseline)
        otherwise → weighted sum:
            40% confidence
            35% success_rate * 100
            15% min(1.0, times_applied / 20)
            10% decay_factor (how recently confirmed)
    """

    HEALTHY_THRESHOLD = 70.0
    NEUTRAL_SCORE = 50.0

    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize dashboard.

        Args:
            agent_id: Optional agent identifier for scoping.
        """
        self.agent_id = agent_id

    def calculate_skill_health_score(self, instinct: Instinct) -> float:
        """
        Calculate a health score (0–100) for a single instinct.

        Args:
            instinct: Instinct to score.

        Returns:
            Health score in range [0, 100].
        """
        if not instinct.enabled:
            return 0.0

        if instinct.times_applied == 0:
            return self.NEUTRAL_SCORE

        # Confidence component (0–40)
        confidence_score = instinct.confidence * 40.0

        # Success rate component (0–35)
        success_score = instinct.success_rate * 35.0

        # Volume component (0–15): rewards 20+ applications
        volume_score = min(1.0, instinct.times_applied / 20.0) * 15.0

        # Decay factor component (0–10)
        decay_score = self._calculate_decay_factor(instinct) * 10.0

        total = confidence_score + success_score + volume_score + decay_score
        return min(100.0, max(0.0, total))

    def _calculate_decay_factor(self, instinct: Instinct) -> float:
        """
        Compute how healthy the instinct's confirmation decay looks.

        Returns 0.0 (not confirmed recently) to 1.0 (recently confirmed).
        """
        if instinct.last_confirmed_at is None:
            return 0.5  # never confirmed — neutral

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        age = (now - instinct.last_confirmed_at).total_seconds()
        age_days = age / 86400

        if age_days <= 7:
            return 1.0
        elif age_days <= 14:
            return 0.5
        else:
            return 0.0

    def get_skill_health_report(
        self,
        instincts: List[Instinct],
    ) -> Dict[str, Any]:
        """
        Generate a full health report for a list of instincts.

        Args:
            instincts: List of instincts to include.

        Returns:
            Dict with overall score, per-skill breakdown, and summary counts.
        """
        if not instincts:
            return {
                "overall_health_score": 0.0,
                "skills": [],
                "total_skills": 0,
                "healthy_count": 0,
                "unhealthy_count": 0,
                "agent_id": self.agent_id,
            }

        skill_entries = []
        for instinct in instincts:
            score = self.calculate_skill_health_score(instinct)
            skill_entries.append({
                "id": instinct.id,
                "name": instinct.name,
                "health_score": score,
                "confidence": instinct.confidence,
                "success_rate": instinct.success_rate,
                "times_applied": instinct.times_applied,
                "enabled": instinct.enabled,
            })

        scores = [e["health_score"] for e in skill_entries]
        overall = sum(scores) / len(scores) if scores else 0.0

        healthy_count = sum(1 for s in scores if s >= self.HEALTHY_THRESHOLD)
        unhealthy_count = sum(1 for s in scores if s < self.HEALTHY_THRESHOLD)

        return {
            "overall_health_score": round(overall, 2),
            "skills": skill_entries,
            "total_skills": len(skill_entries),
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "agent_id": self.agent_id,
        }

    def get_visualization_data(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert a health report into chart-ready data structures.

        Args:
            report: Output of get_skill_health_report().

        Returns:
            Dict with "bar_chart" (list of {label, score} dicts).
        """
        bar_chart = [
            {
                "label": entry.get("name", entry.get("id", "unknown")),
                "score": entry["health_score"],
            }
            for entry in report.get("skills", [])
        ]

        return {
            "bar_chart": bar_chart,
        }
