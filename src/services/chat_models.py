from functools import cache

from langchain_openrouter import ChatOpenRouter

from src.config.settings import LLM_GUARDRAIL_MODEL, LLM_MODEL, LLM_ROUTER_MODEL

"""
Here I'll be using Open Router as my LLM provider.
You'll need to change the functions for your own provider (any LangChain chat
model works: the nodes only use invoke, bind_tools and with_structured_output).
Check this at LangGraph official documentation.

Each model is created once and reused; model_copy only swaps the session_id,
which groups the requests of one conversation in the OpenRouter dashboard.
"""


@cache
def _chat_model(model: str) -> ChatOpenRouter:
    return ChatOpenRouter(model=model, temperature=0.2)


# This should be your main model, which will be used on most agents
def get_llm_model(session_id: str) -> ChatOpenRouter:
    return _chat_model(LLM_MODEL).model_copy(update={"session_id": session_id})


# You can have a different smaller model for your guardrails, in order to save tokens
def get_llm_guardrail_model(session_id: str) -> ChatOpenRouter:
    return _chat_model(LLM_GUARDRAIL_MODEL).model_copy(update={"session_id": session_id})


# You can have another model for routing only
def get_llm_router_model(session_id: str) -> ChatOpenRouter:
    return _chat_model(LLM_ROUTER_MODEL).model_copy(update={"session_id": session_id})
