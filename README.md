# disco-backend

FastAPI backend for the Disco take-home: an advertiser describes their business in a sentence
or two and the service returns **ranked publishers with reasons and exclusions**, **3-5
persona-tuned creatives** with the persona reasoning attached, and a **structured campaign
config**, streamed stage by stage. Orchestration is a typed pipeline over the OpenAI Agents SDK.

## Run it

```bash
cp .env.example .env            # set OPENAI_API_KEY for LLM mode; leave empty for heuristic mode
uv venv && uv pip install -e ".[dev]"    # or: pip install -e ".[dev]"
uvicorn app.main:app --reload   # http://localhost:8000/docs
```

```bash
curl -N localhost:8000/api/plan -H 'content-type: application/json' \
  -d '{"description":"We sell premium dog food for senior dogs. Grain-free, vet-formulated, subscription-based."}'
```

`pytest` runs 85 hermetic tests in four files (domain rules, pipeline end to end, API, SDK runner
against a fake model; no network). `python -m evals.run` grades the 15 sample advertisers
against `evals/cases.py` in whichever mode is configured.

## How it works

```
POST /api/plan ─▶ intake ─▶ router ─▶ signals ─▶ match ─▶ guards ─▶ personas ─▶ creatives ─▶ config ─▶ summary
   (NDJSON)       LLM       code      code       LLM      code       LLM       LLM ×N + lint  code      LLM
```

- **Intake** (agent) turns free text into an `AdvertiserBrief`: controlled category, price tier,
  purchase model, attributes, target customer, and an `input_quality` flag with assumptions,
  clarifying questions and interpretations. Advertiser text is wrapped as data, never as an
  instruction.
- **Router** (code) decides the consequence: junk stops the run, vague/ambiguous input continues
  with a 3-persona cap and a smaller pilot, off-catalog input continues but is expected to end
  with nothing recommended.
- **Signals** (code) compute per-publisher evidence the model must not "vibe": taxonomy category
  overlap, age/gender/income alignment, the post-purchase AOV ratio, reach, note keyword hits.
- **Match** (agent) scores all 20 publishers against a rubric with the signals as evidence;
  **guards** (code) enforce completeness, caps for off-catalog and price mismatches, verdict/score
  consistency and a bounded recommended set, tagging every rule that fired.
- **Personas** (agent) pick 3-5 with a messaging angle each and reject the rest with a reason;
  **creatives** run one agent call per persona in parallel, then a lint pass (lengths,
  unsubstantiated claims, persona disinterests) with one regeneration.
- **Config** (code) assembles targeting, fit-weighted allocation with floor/cap, CPM bands, CPA
  target, confidence-scaled pilot budget, KPIs and a forecast; every constant used is echoed into
  `assumptions`. The optional **summary** agent writes the reviewer narrative.

Without an API key the same pipeline runs a deterministic `heuristic` executor built from the
signals, so the app is clickable and the whole flow is testable end to end; the trace says which
mode ran.

## Layout

```
app/
  main.py           FastAPI app (lifespan warms catalog + prompts), CORS, /health
  config.py         pydantic-settings: models (matcher vs the rest), effort, mode, timeouts
  enums.py          domain vocabularies (StrEnum)
  schemas.py        contracts: catalog rows, stage hand-offs (*Draft = agent output), API shapes
  dependencies.py   composition root (create_pipeline, PipelineProvider)
  routes/           plan (stream + run), examples (the sample advertisers)
  pipeline/         orchestrator: the workflow, event protocol, creative fan-out
  agents/           factory (SDK Agent per stage) · runner (structured output, validation,
                    one retry, usage) · executor (StageExecutor contract + LLM impl) · heuristic
  domain/           taxonomy · signals · guards · economics · allocation · lint · router · planner
  prompts/registry  loads prompts/*.md ({{var}} templating, versioned)
prompts/            every prompt the system uses (see prompts/README.md)
evals/              cases + runner; tests/ pytest (unit, pipeline, API, fake-model runner)
```

## Config shape, and why

Allocation is fit² × log(reach) with a 10% floor and 40% cap over at most 5 publishers: fit
dominates, reach breaks ties, no single publisher swallows a pilot. Bids start from CPM bands by
publisher income tier with a fit multiplier; the target CPA is 30% of the price for one-time
purchases and 60% for subscriptions. The pilot budget scales with how much of the brief was
stated rather than assumed ($5k/14d → $1.5k/7d). `confidence`, `assumptions` and
`open_questions` are fields of the config because a draft that hides its uncertainty is not
reviewable.

## With another week

Learn from outcomes: log every recommendation and its CTR/CVR per advertiser × publisher ×
persona, re-weight the signal prior nightly, and grow a human-labelled eval set into a CI gate.
A retrieval layer (embeddings over category + notes, hard filters on demographics/AOV) so a
10k-publisher catalog is a top-50 re-rank, not a 20-row prompt. Durable orchestration (queue,
idempotent stages, retries, a fallback model), a claims-policy engine per category, an editable
config with an approval state, and per-publisher creative variants.

## Intentionally cut

A database and accounts (a JSON download is the hand-off), image creative, real auction
modelling (heuristic bands with stated assumptions are more honest than fake precision),
multi-turn refinement (one interpretation-chip re-run covers most of it), agent frameworks (the
orchestration is ~150 lines and the seams stay visible), and config editing in the UI.

## Hard vs easy, and where the engineering lives

Easy: the UI plumbing, the JSON contracts, streaming, the allocation arithmetic. Hard:
calibrated matching without ground truth (a model will happily rank by reach; the signals,
guardrails and rubric anchors exist to stop that), copy that is persona-specific rather than
merely plausible (angle-per-persona plus lint plus one retry), deciding when to ask versus
assume (the router policy, tested case by case), and keeping the demo deterministic (the
heuristic executor). The interesting work is the contract layer between model and code, the
eval harness that measures it, and the outcome feedback loop that does not exist yet.
