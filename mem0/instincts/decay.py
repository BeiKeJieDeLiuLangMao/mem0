"""
Instinct decay engine — confidence decay over time.

ECC-style decay:
- No confirmation for decay_window → weekly decay_rate applied
- Confirmation → no decay, clock resets
- Contradiction → penalty applied immediately
"""

from datetime import datetime, timezone, timedelta
from typing import List

from .instincts import Instinct


class InstinctDecayEngine:
    """
    Computes confidence decay for instincts based on time since last confirmation.

    Decay formula:
        age = now - last_confirmed_at
        if age > decay_window:
            decay_amount = instinct.confidence * instinct.decay_rate * (age / 7 days)
            new_confidence = instinct.confidence - decay_amount
    """

    def __init__(
        self,
        decay_window_days: int = 7,
        default_decay_rate: float = 0.02,
        contradiction_penalty: float = 0.15,
    ):
        """
        Initialize decay engine.

        Args:
            decay_window_days: Days without confirmation before decay starts.
            default_decay_rate: Weekly decay rate (fraction of confidence lost per week).
            contradiction_penalty: Fraction to subtract on contradiction.
        """
        self.decay_window_days = decay_window_days
        self.default_decay_rate = default_decay_rate
        self.contradiction_penalty = contradiction_penalty

    def calculate_decay(self, instinct: Instinct) -> float:
        """
        Calculate the decay amount for an instinct.

        Args:
            instinct: Instinct to evaluate.

        Returns:
            Decay amount (0.0 if no decay applies).
        """
        # Disabled instincts are handled separately (skip decay)
        if not instinct.enabled:
            return 0.0

        # No confirmed date → cannot compute age → no decay
        if instinct.last_confirmed_at is None:
            return 0.0

        # Future confirmation date → no decay
        now = datetime.now(timezone.utc)
        if instinct.last_confirmed_at > now:
            return 0.0

        # Zero confidence → nothing to decay
        if instinct.confidence <= 0.0:
            return 0.0

        age = now - instinct.last_confirmed_at
        age_days = age.total_seconds() / 86400

        if age_days <= self.decay_window_days:
            return 0.0

        # Number of weeks beyond the decay window
        weeks = (age_days - self.decay_window_days) / 7.0
        decay = instinct.confidence * instinct.decay_rate * weeks

        return min(decay, instinct.confidence)

    def confirm_instinct(self, instinct: Instinct) -> None:
        """
        Record a confirmation for an instinct (resets the decay clock).

        Args:
            instinct: Instinct that was confirmed.
        """
        instinct.last_confirmed_at = datetime.now(timezone.utc)
        instinct.last_contradicted_at = None

    def contradict_instinct(self, instinct: Instinct) -> None:
        """
        Record a contradiction and reduce instinct confidence.

        Args:
            instinct: Instinct that was contradicted.
        """
        instinct.last_contradicted_at = datetime.now(timezone.utc)
        instinct.confidence = max(0.0, instinct.confidence - self.contradiction_penalty)

    def tick(self, instincts: List[Instinct]) -> List[Instinct]:
        """
        Apply decay to a batch of instincts.

        Args:
            instincts: List of instincts to evaluate.

        Returns:
            List of instincts whose confidence changed.
        """
        decayed: List[Instinct] = []

        for instinct in instincts:
            if not instinct.enabled:
                continue

            decay_amount = self.calculate_decay(instinct)
            if decay_amount > 0.0:
                instinct.confidence = max(0.0, instinct.confidence - decay_amount)
                decayed.append(instinct)

        return decayed
