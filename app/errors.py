"""The one exception the pipeline raises on purpose. Everything a stage can fail with is
classified into a FailureKind so the API can stream a typed `failed` event and the UI can
say something more useful than "error"."""

from app.enums import FailureKind, Stage


class StageError(Exception):
    def __init__(self, stage: Stage, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.kind = kind
        self.message = message

    def __str__(self) -> str:
        return f"{self.stage}: {self.message}"
