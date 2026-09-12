from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.enums import ExecutionMode

# repo root (the directory holding prompts/, data/, fixtures/), independent of the cwd
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    openai_api_key: str = ""

    # Per-stage models: the judgement-heavy matcher gets the strongest tier, the rest the
    # balanced tier, the narrative summary the cheapest. Effort follows the same idea.
    openai_intake_model: str = "gpt-5.6-terra"
    openai_match_model: str = "gpt-5.6-sol"
    openai_persona_model: str = "gpt-5.6-terra"
    openai_creative_model: str = "gpt-5.6-terra"
    openai_summary_model: str = "gpt-5.6-luna"
    intake_reasoning_effort: str = "low"
    match_reasoning_effort: str = "medium"
    persona_reasoning_effort: str = "low"
    creative_reasoning_effort: str = "low"
    summary_reasoning_effort: str = "none"
    llm_timeout_s: float = 90.0     # per model call; the matcher scores 20 publishers in one go
    llm_max_output_tokens: int = 8000

    # "llm" needs OPENAI_API_KEY; without a key the app falls back to "heuristic" and says so
    execution_mode: ExecutionMode = ExecutionMode.LLM
    summary_stage_enabled: bool = True
    stage_cache_enabled: bool = True
    tracing_enabled: bool = True    # OpenAI Agents SDK traces (visible in the OpenAI dashboard)

    frontend_origin: str = "http://localhost:5173"  # comma-separated list of allowed origins

    data_dir: Path = ROOT_DIR / "data"
    prompts_dir: Path = ROOT_DIR / "prompts"
    fixtures_dir: Path = ROOT_DIR / "fixtures"

    @property
    def frontend_origins(self) -> list[str]:
        return [o.strip() for o in self.frontend_origin.split(",") if o.strip()]

    @property
    def effective_mode(self) -> ExecutionMode:
        """The mode the app actually runs in: LLM only when a key is configured."""
        if self.execution_mode is ExecutionMode.LLM and not self.openai_api_key:
            return ExecutionMode.HEURISTIC
        return self.execution_mode


@lru_cache
def settings() -> Settings:
    return Settings()
