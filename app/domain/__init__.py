"""Pure domain logic: no I/O, no LLM calls, fully unit-testable.

catalog     the mock data pack as typed objects
taxonomy    controlled category vocabulary <-> publisher subcategories <-> persona affinities
signals     deterministic per-publisher fit evidence
guards      rules a model verdict must never break, plus ranking
economics   every money constant in one place
allocation  budget split across recommended publishers
lint        creative QA rules
router      what to do with clear / vague / off-catalog / junk input
planner     assembles the campaign config from everything upstream
"""
