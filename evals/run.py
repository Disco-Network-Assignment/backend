"""Run the 15 sample advertisers through the pipeline and grade them against evals/cases.py.

    python -m evals.run                  # mode from settings (llm when OPENAI_API_KEY is set)
    python -m evals.run --mode heuristic # the keyless mode, ~1 s for all cases
    python -m evals.run --only 1,7,15    # a subset

Prints one row per case with an ok/XX per check, writes evals/results/<timestamp>.json (and
latest.json), and exits non-zero when any check fails. Token counts come from the trace, so a
run in LLM mode also reports what it cost in tokens."""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.dependencies import create_pipeline, get_catalog, settings
from app.enums import ExecutionMode
from app.schemas import CampaignPlan, PlanRequest, StopResult
from app.settings import ROOT_DIR
from evals.cases import CASES, EvalCase

RESULTS_DIR = ROOT_DIR / "evals" / "results"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    number: int
    slug: str
    passed: bool
    checks: list[CheckResult]
    ms: int
    input_tokens: int
    output_tokens: int
    summary: str


def grade(case: EvalCase, plan: CampaignPlan | None, stop: StopResult | None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if plan is not None:
        checks.append(CheckResult("quality", plan.brief.input_quality in case.quality,
                                  str(plan.brief.input_quality)))
    checks.append(CheckResult("stopped", (stop is not None) == case.stopped,
                              "stopped" if stop else "ran"))
    if plan is None:
        return checks
    recommended = [p.publisher_id for p in plan.recommended]
    if case.top3_any:
        hit = set(recommended[:3]) & set(case.top3_any)
        checks.append(CheckResult("top3_any", bool(hit), ", ".join(recommended[:3]) or "none"))
    if case.must_not_recommend:
        leaked = sorted(set(recommended) & set(case.must_not_recommend))
        checks.append(CheckResult("must_not_recommend", not leaked, ", ".join(leaked) or "clean"))
    selected = [p.persona_id for p in plan.personas.selected] if plan.personas else []
    if case.personas_any:
        checks.append(CheckResult("personas_any", bool(set(selected) & set(case.personas_any)),
                                  ", ".join(selected) or "none"))
    if case.personas_none:
        leaked = sorted(set(selected) & set(case.personas_none))
        checks.append(CheckResult("personas_none", not leaked, ", ".join(leaked) or "clean"))
    if case.max_personas is not None:
        checks.append(CheckResult("max_personas", len(selected) <= case.max_personas, str(len(selected))))
    if case.status is not None:
        checks.append(CheckResult("status", plan.config.status is case.status, str(plan.config.status)))
    if case.max_budget_usd is not None:
        checks.append(CheckResult("max_budget", plan.config.budget.total_usd <= case.max_budget_usd,
                                  f"${plan.config.budget.total_usd:,.0f}"))
    if case.needs_interpretations:
        checks.append(CheckResult("interpretations", bool(plan.brief.interpretations),
                                  str(len(plan.brief.interpretations))))
    if plan.config.status.value == "draft":
        checks.append(CheckResult("creatives", 1 <= len(plan.creatives) <= 5
                                  and all(c.lint.passed for c in plan.creatives),
                                  f"{len(plan.creatives)} written, "
                                  f"{sum(not c.lint.passed for c in plan.creatives)} failed lint"))
    return checks


async def run_case(pipeline, case: EvalCase, description: str) -> CaseResult:
    started = time.perf_counter()
    response = await pipeline.run(PlanRequest(description=description))
    ms = round((time.perf_counter() - started) * 1000)
    plan, stop = response.plan, response.stopped
    checks = grade(case, plan, stop)
    trace = plan.trace if plan else stop.trace if stop else []
    summary = (f"recommend={[p.publisher_name for p in plan.recommended]} "
               f"personas={[p.persona_name for p in plan.personas.selected] if plan.personas else []}"
               if plan else f"stopped: {stop.reason[:60] if stop else ''}")
    return CaseResult(
        number=case.number, slug=case.slug, passed=all(c.passed for c in checks), checks=checks,
        ms=ms, input_tokens=sum(m.input_tokens or 0 for m in trace),
        output_tokens=sum(m.output_tokens or 0 for m in trace), summary=summary,
    )


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=[m.value for m in ExecutionMode], default=None)
    parser.add_argument("--only", help="comma-separated case numbers")
    args = parser.parse_args(argv)

    mode = ExecutionMode(args.mode) if args.mode else settings().effective_mode
    if mode is ExecutionMode.LLM and not settings().openai_api_key:
        print("OPENAI_API_KEY is not set; running in heuristic mode", file=sys.stderr)
        mode = ExecutionMode.HEURISTIC
    only = {int(n) for n in args.only.split(",")} if args.only else None
    pipeline = create_pipeline(mode)
    examples = {e.number: e.description for e in get_catalog().examples}

    results: list[CaseResult] = []
    print(f"mode={mode}  cases={len([c for c in CASES if not only or c.number in only])}\n")
    for case in CASES:
        if only and case.number not in only:
            continue
        result = await run_case(pipeline, case, examples[case.number])
        results.append(result)
        marks = "  ".join(f"{'ok' if c.passed else 'XX'}:{c.name}" for c in result.checks)
        failed = [f"{c.name}={c.detail}" for c in result.checks if not c.passed]
        print(f"{'PASS' if result.passed else 'FAIL'} {case.number:>2} {case.slug:<22} {result.ms:>6} ms  "
              f"{result.input_tokens:>6}/{result.output_tokens:<5} tok  {marks}")
        if failed:
            print(f"        -> {'; '.join(failed)}")
        print(f"        {result.summary[:150]}")

    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} cases passed")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "ran_at": datetime.now(UTC).isoformat(), "passed": passed,
               "total": len(results), "results": [asdict(r) for r in results]}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for name in (f"{stamp}.json", "latest.json"):
        Path(RESULTS_DIR / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
