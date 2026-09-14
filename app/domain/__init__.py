"""Pure domain logic: no I/O, no LLM calls, fully unit-testable.

catalog          the mock data pack as typed objects
categories       controlled category vocabulary <-> publisher subcategories <-> persona affinities
fit_signals      deterministic per-publisher fit evidence
guardrails       rules a model verdict must never break, plus ranking
economics        every money constant in one place
budget_split     budget split across recommended publishers
creative_checks  the ad unit's length limits
input_policy     what to do with clear / vague / off-catalog / junk input
config_builder   assembles the campaign config from everything upstream
"""
