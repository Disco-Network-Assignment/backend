# Prompts

Every prompt the system uses lives here and is loaded at runtime by `app/prompts/registry.py`,
so this directory is the source of truth, not a copy.

## File format

```
---
name: match_publishers      # registry key
version: 3                  # bump on ANY wording change; it is part of the stage cache key
---
# System
...instructions (the agent's system prompt)...

# User
...the user message template...
```

`{{variable}}` placeholders are substituted at render time; objects are inserted as pretty JSON.
Rendering with a missing variable raises, and `tests/test_prompts.py` renders every prompt with
its declared variables so a typo never reaches production.

| Prompt | Stage | Variables |
|---|---|---|
| `intake.md` | 1 · Intake Analyst | `categories`, `attributes`, `description` |
| `match_publishers.md` | 2 · Publisher Matcher | `catalog`, `brief`, `signals`, `publisher_count` |
| `select_personas.md` | 3 · Persona Strategist | `personas`, `persona_cap`, `brief`, `recommended` |
| `write_creative.md` | 4 · Copywriter (one call per persona) | `brief`, `persona`, `angle`, `watchouts`, `target_publishers`, `feedback` |
| `campaign_summary.md` | 5b · Strategy summary (optional) | `plan` |
| `_fragments/validation_retry.md` | appended when structured output fails validation | `errors` |
| `_fragments/lint_retry.md` | appended when a creative fails lint | `issues` |

The advertiser's text is always wrapped in `<advertiser_description>` tags and declared as data;
it is never interpolated into an instruction sentence.
