"""
TDD RED phase: SkillQualityAuditor tests.
"""

from datetime import datetime, timedelta, timezone

import pytest


class TestAuditorInit:
    """Auditor initialization."""

    def test_auditor_instantiates(self):
        """SkillQualityAuditor can be instantiated."""
        from mem0.quality.auditor import SkillQualityAuditor
        auditor = SkillQualityAuditor()
        assert auditor is not None


class TestAuditFindings:
    """Audit finding structure tests."""

    def test_audit_returns_list(self):
        """audit_skills() returns a list."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(confidence=0.9, success_rate=0.9, times_applied=10)
        findings = auditor.audit_skills([instinct])
        assert isinstance(findings, list)

    def test_audit_findings_have_required_fields(self):
        """Each finding has severity, skill_id, issue_type, message."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="skill-1", confidence=0.05, success_rate=0.1, times_applied=2)
        findings = auditor.audit_skills([instinct])

        # Should have at least one finding for low quality
        if findings:
            f = findings[0]
            assert "severity" in f
            assert "skill_id" in f
            assert "issue_type" in f
            assert "message" in f

    def test_high_quality_skill_produces_no_findings(self):
        """Healthy skill generates zero findings."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(
            id="good-skill",
            confidence=0.95,
            success_rate=0.95,
            times_applied=50,
            enabled=True,
        )
        instinct.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=1)

        findings = auditor.audit_skills([instinct])
        assert findings == []


class TestStaleSkillDetection:
    """Detection of stale (never-confirmed) skills."""

    def test_detects_stale_skill(self):
        """Skill with last_confirmed_at=None is flagged as stale."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="stale-1", confidence=0.8, success_rate=0.8, times_applied=10)
        instinct.last_confirmed_at = None

        findings = auditor.audit_skills([instinct])
        stale_findings = [f for f in findings if f.get("issue_type") == "stale"]
        assert len(stale_findings) == 1


class TestLowConfidenceDetection:
    """Detection of low-confidence skills."""

    def test_detects_low_confidence_skill(self):
        """Skill with confidence < 0.3 is flagged."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="low-conf", confidence=0.1, success_rate=0.5, times_applied=5)

        findings = auditor.audit_skills([instinct])
        low_conf_findings = [f for f in findings if f.get("issue_type") == "low_confidence"]
        assert len(low_conf_findings) == 1


class TestDisabledSkillDetection:
    """Detection of disabled skills."""

    def test_detects_disabled_skill(self):
        """Disabled skill is flagged."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="disabled-1", confidence=0.9, success_rate=0.9, times_applied=20, enabled=False)

        findings = auditor.audit_skills([instinct])
        disabled_findings = [f for f in findings if f.get("issue_type") == "disabled"]
        assert len(disabled_findings) == 1


class TestSeverityLevels:
    """Severity classification tests."""

    def test_disabled_skill_is_critical(self):
        """Disabled skill gets critical severity."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="disabled", confidence=0.9, success_rate=0.9, times_applied=20, enabled=False)

        findings = auditor.audit_skills([instinct])
        assert any(f.get("severity") == "critical" for f in findings)

    def test_low_confidence_is_warning(self):
        """Low confidence gets warning severity."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="low-conf", confidence=0.1, success_rate=0.5, times_applied=5)

        findings = auditor.audit_skills([instinct])
        assert any(f.get("severity") == "warning" for f in findings)


class TestRecommendations:
    """Recommendation generation tests."""

    def test_generate_recommendations_returns_dict(self):
        """generate_recommendations() returns a dict."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="skill-1", confidence=0.1, success_rate=0.1, times_applied=2)
        findings = auditor.audit_skills([instinct])

        recs = auditor.generate_recommendations(findings)
        assert isinstance(recs, dict)

    def test_recommendations_has_skills_key(self):
        """Recommendations has 'skills' key with per-skill advice."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="skill-1", confidence=0.1, success_rate=0.1, times_applied=2)
        findings = auditor.audit_skills([instinct])
        recs = auditor.generate_recommendations(findings)

        assert "skills" in recs
        assert isinstance(recs["skills"], dict)

    def test_recommendations_includes_actions(self):
        """Per-skill recommendations include suggested actions."""
        from mem0.quality.auditor import SkillQualityAuditor
        from mem0.instincts.instincts import Instinct

        auditor = SkillQualityAuditor()
        instinct = Instinct(id="skill-1", confidence=0.1, success_rate=0.1, times_applied=2)
        findings = auditor.audit_skills([instinct])
        recs = auditor.generate_recommendations(findings)

        skill_recs = recs.get("skills", {}).get("skill-1", {})
        assert "actions" in skill_recs
        assert isinstance(skill_recs["actions"], list)
