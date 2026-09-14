"""The OpenAI Agents SDK layer.

context        RunContext: the local state every agent and tool in one run shares
tools          function tools the agents call (fit signals, audience overlap, creative check)
memory         conversation memory for intake, stored in Postgres through the SDK's SQLAlchemySession
openai_agent   build one SDK agent for a stage and run it (structured output, one retry)
llm_stages     the StageExecutor contract the pipeline programs against, and its SDK implementation
"""
