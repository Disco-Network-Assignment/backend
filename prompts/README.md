# Prompts

Every prompt the system uses lives here and is loaded at runtime by `app/prompts/loader.py`,
so this directory is the source of truth, not a copy.

## File format

```
---
name: match_publishers      # loader key
version: 2                  # bump on ANY wording change; it is recorded in the run trace
---
# System
...the agent's instructions...

# User
...the user message template...
```

`{{variable}}` placeholders are substituted at render time; objects are inserted as pretty JSON.
Rendering with a missing variable raises, and `tests/test_domain.py` renders every prompt with
its declared variables so a typo never reaches production.

| Prompt | Agent | Variables | SDK features |
|---|---|---|---|
| `triage.md` | triage | `description` | handoffs to the brief writer or the clarifier (with a reason), session memory |
| `intake.md` | brief writer | `categories`, `attributes` | structured output `AdvertiserBrief` |
| `clarify.md` | clarifier | none | structured output `ClarificationRequest` |
| `match_publishers.md` | publisher matcher | `catalog`, `brief`, `publisher_count` | tool `fit_signals(publisher_id)` |
| `select_personas.md` | persona strategist | `personas`, `persona_cap`, `brief`, `recommended` | tool `audience_overlap(persona_id)` |
| `write_creative.md` | copywriter (one run per persona) | `brief`, `persona`, `angle`, `watchouts`, `target_publishers` | tool `check_creative(...)` (lint as a tool) |
| `campaign_summary.md` | strategy summariser | `plan` | optional hosted `CodeInterpreterTool` |
| `_fragments/validation_retry.md` | appended when structured output fails validation | `errors` | |

The advertiser's text is always wrapped in `<advertiser_description>` tags and declared as data;
it is never interpolated into an instruction sentence. The triage prompt is prefixed at runtime
with the SDK's recommended handoff instructions (`prompt_with_handoff_instructions`).
