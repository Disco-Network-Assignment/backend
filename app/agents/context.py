"""Local context for one pipeline run - the SDK's context-management primitive.

One `RunContext` is created per request and passed to every `Runner.run(..., context=ctx)`.
Function tools and handoff callbacks receive it as `RunContextWrapper[RunContext]` and read or
write it; nothing in it reaches the model unless a tool returns it. Every agent in the run
shares this one type, as the SDK requires."""

from dataclasses import dataclass, field

from app.domain.catalog import CatalogRepository
from app.domain.fit_signals import SignalCalculator
from app.schemas import AdvertiserBrief, FitSignals, PublisherAssessment


@dataclass
class RunContext:
    # dependencies the tools need
    catalog: CatalogRepository
    signals: SignalCalculator
    # working state, filled in as the stages advance
    description: str
    brief: AdvertiserBrief | None = None
    fit_signals: dict[str, FitSignals] = field(default_factory=dict)  # cached per publisher
    recommended: list[PublisherAssessment] = field(default_factory=list)
    handoff_reason: str | None = None  # why triage handed off, from the handoff callback
    creative_checks: int = 0           # how often the copywriter asked for a review
