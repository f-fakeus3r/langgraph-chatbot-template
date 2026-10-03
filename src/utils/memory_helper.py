from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

MAX_HISTORY_MESSAGES = 20
MAX_SHARED_HISTORY_TURNS = 6


def system_prompt(agent_name: str, content: str) -> SystemMessage:
    """
    System prompt with a fixed id: add_messages replaces the previous prompt
    of that agent instead of appending a new copy every turn.
    """
    return SystemMessage(content=content, id=f"{agent_name}-system-prompt")


def append_shared_turn(agent_name: str, input_message: str, assistant_reply) -> list[BaseMessage]:
    """
    Build the messages of one turn for the shared history, tagged with the
    name of the agent that answered.
    """
    messages: list[BaseMessage] = []
    if input_message:
        messages.append(HumanMessage(content=input_message, name=agent_name))
    if assistant_reply:
        messages.append(AIMessage(content=str(
            assistant_reply), name=agent_name))
    return messages


def trim_messages_for_llm(
    messages: list[BaseMessage], keep: int = MAX_HISTORY_MESSAGES
) -> list[BaseMessage]:
    """
    Keep the latest system prompt plus the last `keep` messages, without
    starting on a ToolMessage whose tool call was cut off.
    """
    if not messages:
        return messages

    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]

    start = max(len(rest) - keep, 0)
    while start > 0 and isinstance(rest[start], ToolMessage):
        start -= 1

    return system_messages[-1:] + rest[start:]


def render_shared_history_context(shared_history: list | None, current_agent: str) -> str:
    if not shared_history:
        return ""
    other_turns = [
        m for m in shared_history
        if getattr(m, "name", None) and m.name != current_agent
    ]
    if not other_turns:
        return ""

    recent = other_turns[-(MAX_SHARED_HISTORY_TURNS * 2):]
    lines = [
        f"  - {'User' if isinstance(m, HumanMessage) else f'Assistant ({m.name})'}: {
            m.content}"
        for m in recent
    ]

    return f"""
    ## Recent conversations with other agents (shared memory in this session)

    {chr(10).join(lines)}

    Use this only as continuity context (for example, if the user
    switches topics between conversations and subjects.
    **Do not repeat or summarize this history to the user** — use it only
    to understand the conversation's context.
    """
