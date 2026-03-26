"""
TDD RED phase: SkillHealthDashboard tests.
"""

from datetime import datetime, timezone

import pytest


class TestSkillHealthDashboardInit:
    """Basic dashboard initialization."""

    def test_dashboard_instantiates(self):
        """SkillHealthDashboard can be instantiated."""
        from mem0.quality.dashboard import SkillHealthDashboard
        dashboard = SkillHealthDashboard()
        assert dashboard is not None

    def test_dashboard_accepts_agent_id(self):
        """Dashboard accepts optional agent_id."""
        from mem0.quality.dashboard import SkillHealthDashboard
        dashboard = SkillHealthDashboard(agent_id="agent-42")
        assert dashboard.agent_id == "agent-42"


class TestCalculateSkillHealthScore:
    """Score calculation tests."""

    def test_healthy_skill_returns_high_score(self):
        """Skill with high success_rate returns near-100 score."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.95, success_rate=0.95, times_applied=20)

        score = dashboard.calculate_skill_health_score(instinct)
        assert score >= 85.0

    def test_never_applied_skill_returns_neutral_score(self):
        """Skill with times_applied=0 returns score around 50."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.8, success_rate=0.0, times_applied=0)

        score = dashboard.calculate_skill_health_score(instinct)
        assert 40.0 <= score <= 60.0

    def test_low_confidence_returns_low_score(self):
        """Skill with low confidence returns low score."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.1, success_rate=0.5, times_applied=5)

        score = dashboard.calculate_skill_health_score(instinct)
        assert score < 40.0

    def test_disabled_skill_returns_zero_score(self):
        """Disabled skill returns 0 score."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.95, success_rate=0.95, times_applied=20, enabled=False)

        score = dashboard.calculate_skill_health_score(instinct)
        assert score == 0.0


class TestGetSkillHealthReport:
    """Full health report generation tests."""

    def test_report_returns_dict(self):
        """get_skill_health_report() returns a dict."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.9, success_rate=0.9, times_applied=10)
        instincts = [instinct]

        report = dashboard.get_skill_health_report(instincts)
        assert isinstance(report, dict)

    def test_report_contains_required_fields(self):
        """Report contains expected top-level keys."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.9, success_rate=0.9, times_applied=10)
        report = dashboard.get_skill_health_report([instinct])

        assert "overall_health_score" in report
        assert "skills" in report
        assert "total_skills" in report
        assert "healthy_count" in report
        assert "unhealthy_count" in report

    def test_report_includes_per_skill_scores(self):
        """Each skill in report has a health_score field."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.9, success_rate=0.9, times_applied=10)
        report = dashboard.get_skill_health_report([instinct])

        assert len(report["skills"]) == 1
        assert "health_score" in report["skills"][0]

    def test_report_healthy_count(self):
        """healthy_count reflects skills scoring >= 70."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        healthy = Instinct(confidence=0.95, success_rate=0.95, times_applied=20)
        unhealthy = Instinct(confidence=0.05, success_rate=0.1, times_applied=2)

        report = dashboard.get_skill_health_report([healthy, unhealthy])
        assert report["healthy_count"] == 1
        assert report["unhealthy_count"] == 1

    def test_empty_instincts_returns_zero_score(self):
        """Empty list returns overall_health_score of 0."""
        from mem0.quality.dashboard import SkillHealthDashboard

        dashboard = SkillHealthDashboard()
        report = dashboard.get_skill_health_report([])
        assert report["overall_health_score"] == 0.0
        assert report["total_skills"] == 0

    def test_report_agent_id_reflected(self):
        """Report reflects the agent_id when provided."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard(agent_id="agent-42")
        instinct = Instinct(confidence=0.9, success_rate=0.9, times_applied=10)
        report = dashboard.get_skill_health_report([instinct])

        assert report.get("agent_id") == "agent-42"


class TestVisualizationData:
    """Chart/visualization data generation tests."""

    def test_get_chart_data_returns_dict(self):
        """get_visualization_data() returns a dict."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(confidence=0.9, success_rate=0.9, times_applied=10)
        report = dashboard.get_skill_health_report([instinct])

        viz = dashboard.get_visualization_data(report)
        assert isinstance(viz, dict)

    def test_viz_contains_bar_chart_format(self):
        """Visualization data contains bar chart format for skills."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instincts = [
            Instinct(confidence=0.9, success_rate=0.9, times_applied=10),
            Instinct(confidence=0.5, success_rate=0.5, times_applied=5),
        ]
        report = dashboard.get_skill_health_report(instincts)
        viz = dashboard.get_visualization_data(report)

        assert "bar_chart" in viz
        bars = viz["bar_chart"]
        assert len(bars) == 2  # one per skill

    def test_viz_bar_chart_contains_score_and_label(self):
        """Each bar has score and label fields."""
        from mem0.quality.dashboard import SkillHealthDashboard
        from mem0.instincts.instincts import Instinct

        dashboard = SkillHealthDashboard()
        instinct = Instinct(name="TestSkill", confidence=0.9, success_rate=0.9, times_applied=10)
        report = dashboard.get_skill_health_report([instinct])
        viz = dashboard.get_visualization_data(report)

        bar = viz["bar_chart"][0]
        assert "score" in bar
        assert "label" in bar
        assert bar["score"] >= 0.0
