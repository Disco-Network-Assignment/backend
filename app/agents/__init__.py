"""The OpenAI Agents SDK layer.

factory    builds the SDK Agent for a stage (model, reasoning effort, output schema)
runner     runs one agent call: structured output, validation, one retry, timings, usage
executor   the StageExecutor contract the pipeline programs against, and its LLM implementation
heuristic  the deterministic implementation used without an API key and in tests
"""
