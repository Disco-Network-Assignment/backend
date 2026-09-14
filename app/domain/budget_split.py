"""Budget split across the recommended publishers.

How a pilot budget is divided:

    1. Weight each publisher by fit (score squared) with reach as a tie-breaker.
    2. Turn the weights into shares that add up to 1.
    3. Push every share into a floor/cap band (no token amounts, no publisher swallowing
       the pilot), handing the difference to the publishers that still have room.
    4. Round to whole steps (5%) so the numbers read like a media plan, and fix the
       rounding drift on the largest share so the total is exactly 100%.
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_FLOOR = 0.10          # no publisher gets less than 10% of the pilot
DEFAULT_CAP = 0.40            # no publisher gets more than 40% of the pilot
DEFAULT_MAX_PUBLISHERS = 5    # a pilot spread over more than 5 publishers learns nothing
DEFAULT_STEP = 0.05           # shares are reported in 5% steps
WATER_FILL_MAX_ROUNDS = 20    # the redistribution converges in a few rounds; this is a safety stop


@dataclass(frozen=True)
class AllocationCandidate:
    publisher_id: str
    score: float               # the matcher's 0-100 score after guardrails
    monthly_impressions: int   # reach, from the catalog


@dataclass(frozen=True)
class AllocationShare:
    publisher_id: str
    share: float               # 0..1, already rounded to the allocator's step


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

class BudgetAllocator:
    def __init__(self, floor=DEFAULT_FLOOR, cap=DEFAULT_CAP, max_publishers=DEFAULT_MAX_PUBLISHERS,
                 step=DEFAULT_STEP):
        if not 0 < floor <= cap <= 1:
            raise ValueError("need 0 < floor <= cap <= 1")
        self.floor = floor
        self.cap = cap
        self.max_publishers = max_publishers
        self.step = step

    def allocate(self, candidates: list[AllocationCandidate]) -> list[AllocationShare]:
        """
        Split the budget over the top candidates. `candidates` must already be ranked
        best-first; only the first `max_publishers` get any budget.
        """
        top = candidates[:self.max_publishers]
        if not top:
            return []

        # --- Step 1: weight by fit, with reach as the tie-breaker ---
        weights = self._weights(top)

        # --- Step 2: normalise the weights into shares that add up to 1 ---
        total_weight = sum(weights)
        shares = []
        for weight in weights:
            shares.append(weight / total_weight)

        # --- Step 3: keep every share inside the floor/cap band ---
        shares = self._water_fill(shares)

        # --- Step 4: round to whole steps and make the total exactly 1 ---
        shares = self._round_to_steps(shares)

        result = []
        for candidate, share in zip(top, shares):
            result.append(AllocationShare(candidate.publisher_id, share))
        return result

    @staticmethod
    def _weights(top: list[AllocationCandidate]) -> list[float]:
        """
        Fit dominates: a score of 90 weighs (0.9)^2 = 0.81, a score of 60 weighs 0.36.
        Reach only breaks ties: it scales the weight between 0.5 (smallest publisher in the
        set) and 1.0 (largest), on a log scale so a 60M publisher does not dwarf a 5M one.
        """
        log_reach = []
        for candidate in top:
            log_reach.append(math.log10(max(candidate.monthly_impressions, 1)))
        largest_log_reach = max(log_reach)

        weights = []
        for candidate, this_log_reach in zip(top, log_reach):
            fit = (candidate.score / 100) ** 2
            reach_factor = 0.5 + 0.5 * (this_log_reach / largest_log_reach)
            weights.append(fit * reach_factor)
        return weights

    def _water_fill(self, shares: list[float]) -> list[float]:
        """
        Clamp every share into [floor, cap], then hand whatever the clamping added or removed
        to the shares that are still strictly inside the band, in proportion to their size.
        Repeat until the total is 1 again or nothing can move.
        """
        count = len(shares)
        # With few publishers the band cannot hold: two publishers need 50% each, which is
        # above a 40% cap. Relax the band to whatever an even split needs.
        floor = min(self.floor, 1 / count)
        cap = max(self.cap, 1 / count)

        current = list(shares)
        for _ in range(WATER_FILL_MAX_ROUNDS):
            clamped = []
            for share in current:
                clamped.append(min(cap, max(floor, share)))

            # how far the clamped shares are from adding up to 1
            excess = 1 - sum(clamped)

            # the shares that can still absorb the excess: strictly inside the band
            free_indexes = []
            for index, share in enumerate(clamped):
                if floor < share < cap:
                    free_indexes.append(index)

            if abs(excess) < 1e-9 or not free_indexes:
                return clamped

            free_total = 0.0
            for index in free_indexes:
                free_total += clamped[index]

            # give each free share its proportional part of the excess
            current = list(clamped)
            for index in free_indexes:
                current[index] = clamped[index] + excess * (clamped[index] / free_total)

        return clamped

    def _round_to_steps(self, shares: list[float]) -> list[float]:
        """Round each share to the nearest step, then put the rounding drift on the largest."""
        rounded = []
        for share in shares:
            steps = round(share / self.step)
            rounded.append(round(steps * self.step, 4))

        drift = round(1 - sum(rounded), 4)
        largest_index = rounded.index(max(rounded))
        rounded[largest_index] = round(rounded[largest_index] + drift, 4)
        return rounded
