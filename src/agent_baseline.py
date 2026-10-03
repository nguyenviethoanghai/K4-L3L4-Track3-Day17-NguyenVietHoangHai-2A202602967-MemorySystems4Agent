from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import FactEntry, apply_candidates, estimate_tokens, extract_profile_candidates
from model_provider import build_chat_model
from offline_responder import compose_reply

BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý kỹ thuật nói tiếng Việt. Trả lời ngắn gọn, chính xác. "
    "Chỉ dùng thông tin người dùng đã nói; nếu không biết thì nói rõ là chưa có thông tin."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-thread memory only.

    - every turn re-sends the full thread history (no compaction)
    - no persistent `User.md`; a new thread id starts from zero
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting for one turn.

        `user_id` is accepted for API parity with the advanced agent but is
        deliberately unused: the baseline has no per-user memory.
        """

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _prompt_tokens(self, session: SessionState) -> int:
        """System prompt + the whole thread so far (what a naive agent re-sends)."""

        return estimate_tokens(BASE_SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

    def _thread_facts(self, session: SessionState) -> dict[str, str]:
        """What a model reading the full thread could know: facts from this thread only."""

        entries: dict[str, FactEntry] = {}
        for msg in session.messages:
            if msg["role"] == "user":
                apply_candidates(entries, extract_profile_candidates(msg["content"]), self.config.profile_confidence_threshold)
        return {key: entry.value for key, entry in entries.items()}

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._prompt_tokens(session)

        response = compose_reply(message, self._thread_facts(session))

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "offline",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response, prompt_tokens = _last_ai_text_and_prompt_tokens(result["messages"], fallback=self._prompt_tokens(session))

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "live",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Wire `create_agent` + `InMemorySaver` when a provider is configured.

        Returns None (offline mode) when forced offline, when no credentials are
        set, or when LangChain / the provider SDK is not installed.
        """

        if self.force_offline or not self.config.model.is_configured():
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
        except Exception:
            return None
        return create_agent(model, tools=[], system_prompt=BASE_SYSTEM_PROMPT, checkpointer=InMemorySaver())


def _last_ai_text_and_prompt_tokens(messages: list, fallback: int) -> tuple[str, int]:
    """Final AI text + input tokens the provider reported for the current turn
    (every model call after the latest human message, tool loops included)."""

    since = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=0)
    text = ""
    prompt_tokens = 0
    for msg in messages[since:]:
        if getattr(msg, "type", "") != "ai":
            continue
        usage = getattr(msg, "usage_metadata", None) or {}
        prompt_tokens += int(usage.get("input_tokens", 0))
        text = str(getattr(msg, "text", "") or msg.content)
    return text, prompt_tokens or fallback
