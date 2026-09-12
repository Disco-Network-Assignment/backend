"""In-process cache for LLM stage outputs. The key covers everything that changes an answer -
stage, prompt version, model, reasoning effort and the exact input - so a prompt edit or a
model swap never serves a stale result. Bounded LRU; a class because it owns state."""

import hashlib
import json
from collections import OrderedDict
from typing import TypeVar

from pydantic import BaseModel

from app.agents.runner import StageRun
from app.enums import Stage
from app.schemas import StageMeta

T = TypeVar("T", bound=BaseModel)


class StageCache:
    def __init__(self, enabled: bool = True, max_entries: int = 256) -> None:
        self._enabled = enabled
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[dict, StageMeta]] = OrderedDict()

    @staticmethod
    def key(stage: Stage, prompt_version: str, model: str, effort: str, payload) -> str:
        material = json.dumps(
            {"stage": str(stage), "prompt": prompt_version, "model": model, "effort": effort,
             "payload": payload},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str, output_type: type[T]) -> StageRun[T] | None:
        if not self._enabled or key not in self._entries:
            return None
        data, meta = self._entries[key]
        self._entries.move_to_end(key)
        return StageRun(output=output_type.model_validate(data),
                        meta=meta.model_copy(update={"cached": True, "ms": 0}))

    def put(self, key: str, run: StageRun) -> None:
        if not self._enabled:
            return
        self._entries[key] = (run.output.model_dump(mode="json"), run.meta)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
