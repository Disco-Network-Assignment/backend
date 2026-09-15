# Disco campaign brain

An advertiser types one or two sentences about their business. The system returns **ranked
publishers with a reason each and why the rest were excluded**, **3-5 ad creatives, one per
shopper persona, with the persona reasoning shown**, and a **structured campaign config**,
streamed stage by stage so every decision is visible as it is made.

**Demo:** https://main.d3afgogco54xng.amplifyapp.com · API: https://3-109-228-123.sslip.io/health
· Frontend repo: https://github.com/Disco-Network-Assignment/frontend

## Run it locally

```bash
cp .env.example .env                       # put your OPENAI_API_KEY in it
docker compose up -d                       # Postgres (session memory + run history)
uv venv && uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"
uvicorn app.main:app --reload              # http://localhost:8000/docs
# in ../frontend:  npm install && npm run dev   -> http://localhost:5173
```

`pytest` runs 66 hermetic tests (a scripted executor and a fake SDK model stand in for the
agents). `python -m evals.run` runs the real agents over the 15 sample advertisers and grades
each against `evals/cases.py` (add `--only 1,7,15` for a subset; a full pass is ~500k tokens).
Deploy to one small EC2 box with `deploy/deploy.sh <ip>` (Postgres + API + Caddy TLS).

## What it is

A code-orchestrated pipeline over the OpenAI Agents SDK. Agents answer the four judgement
questions; code does the numbers and the rules; every hand-off is a typed contract.

- **Intake**: a triage agent hands off to a brief writer (a real business) or a clarifier
  (nothing to plan with). Junk stops with questions; vague or ambiguous input continues with
  stated assumptions, interpretation chips and a smaller pilot; off-catalog input continues but
  ends "not recommended". A Postgres session lets a refined description build on the last turn.
- **Match**: code computes fit signals per publisher (category overlap, age/gender/income
  alignment, price vs the publisher's AOV, reach). The matcher agent scores all 20 with a rubric,
  calling the signals as a tool, and reads the publisher notes itself. Guardrails then enforce
  what a model must never break: every publisher accounted for, off-catalog and price-mismatch
  caps, verdict/score consistency, a recommended set of 3-6. Every rule that fires is tagged.
- **Personas and creatives**: the strategist picks 3-5 personas with an angle and watch-outs
  each and rejects the rest with a reason. One copywriter per persona runs in parallel and checks
  its own draft's lengths with a tool; claims and persona fit are its judgement, explained in
  `persona_reasoning` next to the copy.
- **Config** (code): targeting from brief + personas + publishers; budget split by fit² × log
  reach inside a 10-40% band; CPM bands by publisher income tier with a fit multiplier; CPA target
  at 30% of price (60% for subscriptions); pilot size scaled by how much of the brief was stated
  rather than assumed; KPIs and a forecast. `confidence`, `assumptions` and `open_questions` are
  fields because a draft that hides its uncertainty is not reviewable.

Every prompt is in `prompts/`: one file per agent, the retry fragment, and the tool and handoff
descriptions the model sees. Runs are stored and reloadable from the dashboard.

## With another week

Learn from outcomes: log CTR/CVR per advertiser × publisher × persona, re-weight the signal prior
nightly, grow a human-labelled eval set into a CI gate. A retrieval layer so a 10k-publisher
catalog is a top-50 re-rank, not a 20-row prompt. Durable orchestration (queue, idempotent
stages, fallback model), a claims policy per category, an editable config with approval state.

## Intentionally cut

Accounts and multi-tenancy (one workspace, a JSON export is the hand-off), image creative, real
auction modelling (bands with stated assumptions beat fake precision), config editing in the UI,
and any keyword or regex fallback: reading the advertiser is the agents' job, so an input no
pattern anticipated is handled like any other.

## Hard vs easy

Easy: the UI, the contracts, streaming, the allocation arithmetic. Hard: calibrated matching
without ground truth (a model happily ranks by reach; signals, guardrails and rubric anchors
stop it), copy that is persona-specific rather than plausible, deciding when to ask versus
assume, and testing an agent pipeline without paying for every run. The interesting engineering
is the contract layer between model and code, the eval harness that measures it, and the outcome
feedback loop that does not exist yet.
