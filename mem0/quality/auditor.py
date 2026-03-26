"""
Skill Quality Auditor — detects and reports skill quality issues.

ECC-style auditor:
- Detects stale, low-confidence, and disabled skills
- Assigns severity levels (info, warning, critical)
- Generates actionable recommendations per skill
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..instincts.instincts import Instinct


# Thresholds
LOW_CONFIDENCE_THRESHOLD = 0.3
STALE_THRESHOLD_DAYS = 30


class SkillQualityAuditor:
    """
    Audit instincts/skills and produce findings with recommendations.

    Issue types:
    - disabled: skill is disabled (critical)
    - low_confidence: confidence below threshold (warning)
    - low_success_rate: success_rate < 0.5 (warning)
    - stale: no confirmation in STALE_THRESHOLD_DAYS (info)
    - never_applied: times_applied == 0 (info)
    """

    def audit_skills(self, instincts: List[Instinct]) -> List[Dict[str, Any]]:
        """
        Audit a list of instincts and return findings.

        Args:
            instincts: List of instincts to audit.

        Returns:
            List of finding dicts, each with:
            - skill_id: str
            - issue_type: str
            - severity: str  ("info" | "warning" | "critical")
            - message: str
        """
        findings: List[Dict[str, Any]] = []

        for instinct in instincts:
            findings.extend(self._audit_single(instinct))

        return findings

    def _audit_single(self, instinct: Instinct) -> List[Dict[str, Any]]:
        """Audit one instinct, return zero or more findings."""
        findings: List[Dict[str, Any]] = []

        # Disabled check (critical)
        if not instinct.enabled:
            findings.append(self._make_finding(
                instinct,
                "disabled",
                "critical",
                "Skill is disabled and will not be applied.",
            ))
            return findings  # skip further checks

        # Low confidence (warning)
        if instinct.confidence < LOW_CONFIDENCE_THRESHOLD:
            findings.append(self._make_finding(
                instinct,
                "low_confidence",
                "warning",
                f"Confidence is very low ({instinct.confidence:.2f}). "
                f"Consider improving the skill's trigger conditions.",
            ))

        # Low success rate (warning)
        if instinct.success_rate < 0.5 and instinct.times_applied >= 5:
            findings.append(self._make_finding(
                instinct,
                "low_success_rate",
                "warning",
                f"Success rate is low ({instinct.success_rate:.2f}) after "
                f"{instinct.times_applied} applications.",
            ))

        # Never applied (info)
        if instinct.times_applied == 0:
            findings.append(self._make_finding(
                instinct,
                "never_applied",
                "info",
                "Skill has never been applied.",
            ))

        # Stale check (info)
        if instinct.last_confirmed_at is None:
            findings.append(self._make_finding(
                instinct,
                "stale",
                "info",
                "Skill has never been confirmed.",
            ))
        else:
            now = datetime.now(timezone.utc)
            age_days = (now - instinct.last_confirmed_at).total_seconds() / 86400
            if age_days > STALE_THRESHOLD_DAYS:
                findings.append(self._make_finding(
                    instinct,
                    "stale",
                    "info",
                    f"Skill has not been confirmed in {int(age_days)} days.",
                ))

        return findings

    def _make_finding(
        self,
        instinct: Instinct,
        issue_type: str,
        severity: str,
        message: str,
    ) -> Dict[str, Any]:
        """Construct a finding dict."""
        return {
            "skill_id": instinct.id,
            "issue_type": issue_type,
            "severity": severity,
            "message": message,
        }

    def generate_recommendations(
        self,
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Turn audit findings into actionable recommendations.

        Args:
            findings: Output of audit_skills().

        Returns:
            Dict with:
            - skills: {skill_id: {actions: [...], summary: str}}
            - total_issues: int
        """
        skill_recs: Dict[str, Dict[str, Any]] = {}

        for finding in findings:
            skill_id = finding["skill_id"]
            if skill_id not in skill_recs:
                skill_recs[skill_id] = {"actions": [], "summary": ""}

            action = self._action_for_finding(finding)
            if action not in skill_recs[skill_id]["actions"]:
                skill_recs[skill_id]["actions"].append(action)

        # Build summary
        for skill_id, rec in skill_recs.items():
            issue_types = [
                f["issue_type"]
                for f in findings
                if f["skill_id"] == skill_id
            ]
            rec["summary"] = f"Issues: {', '.join(set(issue_types))}."

        return {
            "skills": skill_recs,
            "total_issues": len(findings),
        }

    def _action_for_finding(self, finding: Dict[str, Any]) -> str:
        """Map a finding to a recommended action."""
        mapping = {
            "disabled": "Review and re-enable if valid, or delete.",
            "low_confidence": "Improve trigger conditions or increase training data.",
            "low_success_rate": "Analyze failed cases and refine skill logic.",
            "never_applied": "Monitor for triggers or adjust trigger phrases.",
            "stale": "Trigger a confirmation or deprecate if no longer relevant.",
        }
        return mapping.get(
            finding["issue_type"],
            "Investigate and resolve.",
        )
