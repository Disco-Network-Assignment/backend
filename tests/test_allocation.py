import pytest

from app.domain.allocation import AllocationCandidate, BudgetAllocator


def candidates(*rows):
    return [AllocationCandidate(f"pub_{i}", score, imps) for i, (score, imps) in enumerate(rows)]


class TestBudgetAllocator:
    def test_shares_sum_to_one(self):
        shares = BudgetAllocator().allocate(candidates((94, 4_800_000), (89, 62_000_000), (81, 8_400_000)))
        assert sum(s.share for s in shares) == pytest.approx(1.0)
        assert all(round(s.share / 0.05) * 0.05 == pytest.approx(s.share) for s in shares)

    def test_single_publisher_gets_everything(self):
        assert BudgetAllocator().allocate(candidates((80, 1_000_000)))[0].share == 1.0

    def test_two_publishers_relax_the_cap(self):
        shares = BudgetAllocator().allocate(candidates((95, 1_000_000), (70, 1_000_000)))
        assert sum(s.share for s in shares) == pytest.approx(1.0)
        assert max(s.share for s in shares) >= 0.5

    def test_floor_and_cap_hold_with_five(self):
        shares = BudgetAllocator().allocate(
            candidates((99, 80_000_000), (60, 3_000_000), (60, 3_000_000), (60, 3_000_000), (60, 3_000_000)))
        assert max(s.share for s in shares) <= 0.40 + 1e-9
        assert min(s.share for s in shares) >= 0.10 - 1e-9

    def test_only_top_n_get_budget(self):
        shares = BudgetAllocator(max_publishers=3).allocate(candidates(*[(80, 1_000_000)] * 6))
        assert len(shares) == 3

    def test_fit_beats_reach(self):
        shares = BudgetAllocator().allocate(
            candidates((95, 3_000_000), (75, 80_000_000), (75, 80_000_000)))
        assert shares[0].share > shares[1].share == shares[2].share

    def test_empty(self):
        assert BudgetAllocator().allocate([]) == []

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            BudgetAllocator(floor=0.5, cap=0.4)
