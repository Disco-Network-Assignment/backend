"""The mock data pack as typed objects. Loaded once per process; every lookup after that is a
dict hit. A class because it owns state (the parsed rows) and every stage depends on it."""

import math
from pathlib import Path

from pydantic import TypeAdapter

from app.schemas import ExampleAdvertiser, Publisher, ShopperPersona


class CatalogRepository:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._publishers = TypeAdapter(list[Publisher]).validate_json(
            (data_dir / "publishers.json").read_text(encoding="utf-8")
        )
        self._personas = TypeAdapter(list[ShopperPersona]).validate_json(
            (data_dir / "shopper_personas.json").read_text(encoding="utf-8")
        )
        self._examples = self._parse_examples(
            (data_dir / "example_advertisers.txt").read_text(encoding="utf-8")
        )
        self._publishers_by_id = {p.id: p for p in self._publishers}
        self._personas_by_id = {p.id: p for p in self._personas}
        logs = [math.log10(p.monthly_impressions) for p in self._publishers]
        self._log_reach_bounds = (min(logs), max(logs))

    # ---- publishers ----
    @property
    def publishers(self) -> list[Publisher]:
        return list(self._publishers)

    def publisher(self, publisher_id: str) -> Publisher:
        return self._publishers_by_id[publisher_id]

    def has_publisher(self, publisher_id: str) -> bool:
        return publisher_id in self._publishers_by_id

    @property
    def log_reach_bounds(self) -> tuple[float, float]:
        """(min, max) of log10(monthly_impressions) across the catalog, for the reach index."""
        return self._log_reach_bounds

    # ---- personas ----
    @property
    def personas(self) -> list[ShopperPersona]:
        return list(self._personas)

    def persona(self, persona_id: str) -> ShopperPersona:
        return self._personas_by_id[persona_id]

    def has_persona(self, persona_id: str) -> bool:
        return persona_id in self._personas_by_id

    # ---- examples ----
    @property
    def examples(self) -> list[ExampleAdvertiser]:
        return list(self._examples)

    def example_descriptions(self) -> list[str]:
        return [e.description for e in self._examples]

    def publishers_as_dicts(self) -> list[dict]:
        """The catalog as the agents see it: every field, nothing renamed."""
        return [p.model_dump() for p in self._publishers]

    def personas_as_dicts(self) -> list[dict]:
        return [p.model_dump() for p in self._personas]

    @staticmethod
    def _parse_examples(text: str) -> list[ExampleAdvertiser]:
        """Lines look like '3. We sell ...'; anything else (blank lines, headings) is skipped."""
        examples: list[ExampleAdvertiser] = []
        for line in text.splitlines():
            number, dot, description = line.strip().partition(".")
            if dot and number.isdigit() and description.strip():
                examples.append(ExampleAdvertiser(id=f"example-{int(number):02d}", number=int(number),
                                                  description=description.strip()))
        return examples

    def __repr__(self) -> str:  # helps in logs and debugger sessions
        return (f"CatalogRepository(publishers={len(self._publishers)}, "
                f"personas={len(self._personas)}, examples={len(self._examples)})")


def load_catalog(data_dir: Path) -> CatalogRepository:
    if not (data_dir / "publishers.json").exists():
        raise FileNotFoundError(f"data pack not found in {data_dir}")
    return CatalogRepository(data_dir)
