"""Budget split across the recommended publishers.

Share is driven by fit (score squared) with reach as the tie-breaker (log-scaled, half weight),
then water-filled into a floor/cap band so no single publisher swallows the pilot and none gets
a token amount, and finally rounded to whole steps so the numbers read like a media plan."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationCandidate:
    publisher_id: str
    score: float
    monthly_impressions: int


@dataclass(frozen=True)
class AllocationShare:
    publisher_id: str
    share: float  # 0..1, already rounded to the allocator's step


class BudgetAllocator:
    def __init__(self, floor: float = 0.10, cap: float = 0.40, max_publishers: int = 5,
                 step: float = 0.05) -> None:
        if not 0 < floor <= cap <= 1:
            raise ValueError("need 0 < floor <= cap <= 1")
        self._floor = floor
        self._cap = cap
        self._max_publishers = max_publishers
        self._step = step

    def allocate(self, candidates: list[AllocationCandidate]) -> list[AllocationShare]:
        """`candidates` must already be ranked best-first; only the top N get budget."""
        top = candidates[: self._max_publishers]
        if not top:
            return []
        weights = self._weights(top)
        shares = self._water_fill([w / sum(weights) for w in weights], len(top))
        rounded = self._round(shares)
        return [AllocationShare(c.publisher_id, s) for c, s in zip(top, rounded, strict=True)]

    @staticmethod
    def _weights(top: list[AllocationCandidate]) -> list[float]:
        max_log = max(math.log10(max(c.monthly_impressions, 1)) for c in top)
        return [
            (c.score / 100) ** 2 * (0.5 + 0.5 * math.log10(max(c.monthly_impressions, 1)) / max_log)
            for c in top
        ]

    def _water_fill(self, shares: list[float], n: int) -> list[float]:
        # with few publishers the cap cannot hold (2 publishers need >= 50% each), so relax it
        floor = min(self._floor, 1 / n)
        cap = max(self._cap, 1 / n)
        current = list(shares)
        for _ in range(20):
            clamped = [min(cap, max(floor, s)) for s in current]
            free = [floor < s < cap for s in clamped]
            excess = 1 - sum(clamped)
            if abs(excess) < 1e-9 or not any(free):
                return clamped
            free_sum = sum(s for s, f in zip(clamped, free, strict=True) if f)
            current = [s + excess * (s / free_sum) if f else s
                       for s, f in zip(clamped, free, strict=True)]
        return clamped

    def _round(self, shares: list[float]) -> list[float]:
        rounded = [round(round(s / self._step) * self._step, 4) for s in shares]
        drift = round(1 - sum(rounded), 4)
        largest = rounded.index(max(rounded))
        rounded[largest] = round(rounded[largest] + drift, 4)
        return rounded
