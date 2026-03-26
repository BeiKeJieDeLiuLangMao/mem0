"""
TDD RED phase: InstinctDecayEngine tests.
All tests should FAIL until the engine is implemented.
"""

from datetime import datetime, timedelta, timezone

import pytest


class TestInstinctDecayEngineBasic:
    """Basic engine functionality."""

    def test_engine_instantiates(self):
        """InstinctDecayEngine can be instantiated."""
        from mem0.instincts.decay import InstinctDecayEngine
        engine = InstinctDecayEngine()
        assert engine is not None

    def test_engine_accepts_custom_decay_rate(self):
        """Engine accepts custom default decay rate."""
        from mem0.instincts.decay import InstinctDecayEngine
        engine = InstinctDecayEngine(default_decay_rate=0.05)
        assert engine.default_decay_rate == 0.05


class TestCalculateDecay:
    """calculate_decay() method tests."""

    def test_no_decay_when_recently_confirmed(self):
        """No decay applied if confirmed within decay_window."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        instinct.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=2)

        decayed = engine.calculate_decay(instinct)
        assert decayed == 0.0

    def test_decay_applied_after_decay_window(self):
        """Decay is applied when no recent confirmation."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        instinct.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        decayed = engine.calculate_decay(instinct)
        assert 0.0 < decayed <= 0.8

    def test_decay_never_confirmed_instinct(self):
        """Instinct with no last_confirmed_at gets max decay (age = 0 → 0 decay)."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        instinct.last_confirmed_at = None

        decayed = engine.calculate_decay(instinct)
        # No confirmed date means no decay tick — age cannot be computed
        assert decayed == 0.0

    def test_decay_disabled_instinct_returns_zero(self):
        """Disabled instinct returns zero decay (already handled separately)."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct = Instinct(confidence=0.8, decay_rate=0.02, enabled=False)
        instinct.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        decayed = engine.calculate_decay(instinct)
        assert decayed == 0.0

    def test_decay_rate_used_in_calculation(self):
        """Higher decay_rate → higher decayed confidence."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct_slow = Instinct(confidence=0.8, decay_rate=0.01)
        instinct_slow.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        instinct_fast = Instinct(confidence=0.8, decay_rate=0.05)
        instinct_fast.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        decayed_slow = engine.calculate_decay(instinct_slow)
        decayed_fast = engine.calculate_decay(instinct_fast)

        assert decayed_fast >= decayed_slow


class TestConfirmInstinct:
    """confirm_instinct() method tests."""

    def test_confirm_sets_last_confirmed_at(self):
        """confirm_instinct() sets last_confirmed_at to now."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine()
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        before = datetime.now(timezone.utc)

        engine.confirm_instinct(instinct)

        assert instinct.last_confirmed_at is not None
        assert instinct.last_confirmed_at >= before

    def test_confirm_clears_last_contradicted_at(self):
        """confirm_instinct() clears any previous contradiction."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine()
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        instinct.last_contradicted_at = datetime.now(timezone.utc) - timedelta(days=1)

        engine.confirm_instinct(instinct)

        assert instinct.last_contradicted_at is None

    def test_confirm_does_not_modify_confidence(self):
        """confirm_instinct() only updates timestamps, not confidence."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine()
        instinct = Instinct(confidence=0.8, decay_rate=0.02)

        engine.confirm_instinct(instinct)

        assert instinct.confidence == 0.8


class TestContradictInstinct:
    """contradict_instinct() method tests."""

    def test_contradict_sets_last_contradicted_at(self):
        """contradict_instinct() sets last_contradicted_at to now."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine()
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        before = datetime.now(timezone.utc)

        engine.contradict_instinct(instinct)

        assert instinct.last_contradicted_at is not None
        assert instinct.last_contradicted_at >= before

    def test_contradict_reduces_confidence(self):
        """contradict_instinct() reduces instinct confidence."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(contradiction_penalty=0.1)
        instinct = Instinct(confidence=0.8, decay_rate=0.02)

        engine.contradict_instinct(instinct)

        assert instinct.confidence < 0.8
        assert instinct.confidence >= 0.0

    def test_contradict_confidence_never_negative(self):
        """contradict_instinct() never reduces confidence below 0."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(contradiction_penalty=0.5)
        instinct = Instinct(confidence=0.1, decay_rate=0.02)

        engine.contradict_instinct(instinct)

        assert instinct.confidence >= 0.0


class TestTick:
    """tick() batch processing tests."""

    def test_tick_applies_decay_to_overdue_instincts(self):
        """tick() reduces confidence for instincts past decay_window."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instincts = []

        for i in range(3):
            instinct = Instinct(
                confidence=0.8,
                decay_rate=0.02,
                last_confirmed_at=datetime.now(timezone.utc) - timedelta(days=14),
            )
            instincts.append(instinct)

        engine.tick(instincts)

        for instinct in instincts:
            # Each instinct has been decayed
            assert instinct.confidence < 0.8

    def test_tick_respects_enabled_flag(self):
        """tick() skips disabled instincts."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        disabled = Instinct(confidence=0.8, decay_rate=0.02, enabled=False)
        disabled.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        engine.tick([disabled])

        assert disabled.confidence == 0.8  # not modified

    def test_tick_returns_list_of_decayed_instincts(self):
        """tick() returns list of instincts whose confidence changed."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)

        overdue = Instinct(confidence=0.8, decay_rate=0.02)
        overdue.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        recent = Instinct(confidence=0.8, decay_rate=0.02)
        recent.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=2)

        decayed = engine.tick([overdue, recent])

        assert overdue in decayed
        assert recent not in decayed

    def test_tick_empty_list(self):
        """tick() handles empty list without error."""
        from mem0.instincts.decay import InstinctDecayEngine

        engine = InstinctDecayEngine()
        result = engine.tick([])
        assert result == []


class TestDecayEdgeCases:
    """Edge case tests."""

    def test_decay_with_future_last_confirmed_at(self):
        """Instinct with future last_confirmed_at gets no decay."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        instinct.last_confirmed_at = datetime.now(timezone.utc) + timedelta(days=1)

        decayed = engine.calculate_decay(instinct)
        assert decayed == 0.0

    def test_decay_confidence_exactly_zero(self):
        """Instinct with confidence=0 returns 0 decay."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)
        instinct = Instinct(confidence=0.0, decay_rate=0.02)
        instinct.last_confirmed_at = datetime.now(timezone.utc) - timedelta(days=14)

        decayed = engine.calculate_decay(instinct)
        assert decayed == 0.0

    def test_decay_age_calculation_with_tz_aware(self):
        """Age is computed correctly with timezone-aware datetimes."""
        from mem0.instincts.decay import InstinctDecayEngine
        from mem0.instincts.instincts import Instinct

        engine = InstinctDecayEngine(decay_window_days=7)

        # UTC+0 confirmation 14 days before 2026-03-24
        confirmed = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        instinct = Instinct(confidence=0.8, decay_rate=0.02)
        instinct.last_confirmed_at = confirmed

        decayed = engine.calculate_decay(instinct)
        # Confirmed 14 days ago — beyond decay window → should decay
        assert decayed > 0.0
