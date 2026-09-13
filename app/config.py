from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.enums import ExecutionMode

# repo root (holds prompts/ and data/), independent of the cwd
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-terra"        # intake, personas, creatives, summary
    openai_match_model: str = "gpt-5.6-sol"    # the judgement-heavy matcher gets the strongest tier
    reasoning_effort: str = "low"
    match_reasoning_effort: str = "medium"
    llm_timeout_s: float = 90.0
    llm_max_output_tokens: int = 8000

    # "llm" needs OPENAI_API_KEY; without a key the app runs the deterministic "heuristic" mode
    execution_mode: ExecutionMode = ExecutionMode.LLM
    summary_stage_enabled: bool = True
    tracing_enabled: bool = True  # OpenAI Agents SDK traces, visible in the OpenAI dashboard

    frontend_origin: str = "http://localhost:5173"  # comma-separated allowlist (CORS)
    data_dir: Path = ROOT_DIR / "data"
    prompts_dir: Path = ROOT_DIR / "prompts"

    @property
    def frontend_origins(self) -> list[str]:
        return [o.strip() for o in self.frontend_origin.split(",") if o.strip()]

    @property
    def effective_mode(self) -> ExecutionMode:
        if self.execution_mode is ExecutionMode.LLM and not self.openai_api_key:
            return ExecutionMode.HEURISTIC
        return self.execution_mode


@lru_cache
def settings() -> Settings:
    return Settings()
