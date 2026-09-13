"""The OpenAI Agents SDK layer.

openai_agent      build one SDK agent for a stage and run it once (structured output, one retry)
llm_stages        the StageExecutor contract the pipeline programs against, and its LLM implementation
heuristic_stages  the deterministic implementation used without an API key and in tests
"""
