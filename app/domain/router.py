"""What to do with an advertiser description once the intake agent has classified it.

The classification needs language understanding, so the model makes it; the consequences are
policy, so code decides them and tests pin them down: junk stops the run, vague or ambiguous
input continues with visible assumptions and a smaller pilot, off-catalog input continues but
is expected to end with nothing recommended."""

import re
from dataclasses import dataclass

from app.enums import InputQuality, RouteFlag
from app.schemas import AdvertiserBrief


@dataclass(frozen=True)
class RouteDecision:
    stop: bool
    reason: str | None = None
    flags: tuple[RouteFlag, ...] = ()
    persona_cap: int = 5
    confidence_multiplier: float = 1.0


class InputRouter:
    _JUNK = re.compile(r"^(test|testing|asdf|idk|hello|hi|hey|\.+|\?+|-+)$", re.I)
    MIN_WORDS = 3
    STOP_REASON = ("The description does not say what is sold or to whom, so there is nothing "
                   "to match publishers or personas against.")

    def is_trivially_insufficient(self, description: str) -> bool:
        """Cheap pre-check that saves a model call on empty or junk input."""
        text = description.strip()
        return len(text.split()) < self.MIN_WORDS or bool(self._JUNK.match(text))

    def route(self, brief: AdvertiserBrief) -> RouteDecision:
        quality = brief.input_quality
        if quality is InputQuality.INSUFFICIENT:
            return RouteDecision(stop=True, reason=self.STOP_REASON)
        if quality is InputQuality.VAGUE:
            return RouteDecision(stop=False, flags=(RouteFlag.ASSUMPTIONS,), persona_cap=3,
                                 confidence_multiplier=0.7)
        if quality is InputQuality.AMBIGUOUS:
            return RouteDecision(stop=False,
                                 flags=(RouteFlag.ASSUMPTIONS, RouteFlag.INTERPRETATIONS),
                                 persona_cap=3, confidence_multiplier=0.7)
        if quality is InputQuality.OFF_CATALOG or not brief.is_consumer_commerce:
            return RouteDecision(stop=False, flags=(RouteFlag.OFF_CATALOG, RouteFlag.ASSUMPTIONS),
                                 persona_cap=3, confidence_multiplier=0.5)
        return RouteDecision(stop=False)
