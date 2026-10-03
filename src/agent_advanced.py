from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_baseline import BASE_SYSTEM_PROMPT, _last_ai_text_and_prompt_tokens
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_candidates
from model_provider import build_chat_model
from offline_responder import compose_reply

try:  # module-level so tool annotations resolve; offline mode works without LangChain
    from langchain.tools import ToolRuntime
except ImportError:  # pragma: no cover
    ToolRuntime = None

MEMORY_GUIDANCE = (
    "Thông tin trong User.md là fact ổn định đã được xác nhận; ưu tiên chúng khi trả lời. "
    "Khi người dùng đính chính, fact mới thay fact cũ. Đừng lưu câu hỏi, câu đùa hay ví dụ thành fact."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. short-term: recent messages of the current thread (CompactMemoryManager)
    2. persistent: `User.md` per user, survives new threads and new processes
    3. compact: older thread content folded into a bounded summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._live_compactions: dict[str, int] = {}
        self._live_message_count: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        if self.langchain_agent is not None:
            return self._live_compactions.get(thread_id, 0)
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, message: str) -> list[str]:
        """Persist confident, non-question facts from the message into User.md."""

        candidates = extract_profile_candidates(message)
        return self.profile_store.apply_candidates(user_id, candidates, self.config.profile_confidence_threshold)

    def _record(self, thread_id: str, message: str, response: str, prompt_tokens: int) -> int:
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return agent_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message, updates)

        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = self._record(thread_id, message, response, prompt_tokens)
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "offline",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
            "profile_updates": updates,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """System prompt + User.md profile + compact summary + kept recent messages."""

        return (
            estimate_tokens(BASE_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.profile_prompt(user_id))
            + self.compact_memory.context_tokens(thread_id)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str, updates: list[str] | None = None) -> str:
        """Answer from persisted memory (User.md already includes this thread's facts)."""

        return compose_reply(message, self.profile_store.facts(user_id), updates)

    # -- live mode ---------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # The deterministic extractor is the write guardrail even in live mode;
        # the model can still call `save_user_fact` for anything it misses.
        updates = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        before = self._live_message_count.get(thread_id, 0)

        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id))),
        )
        messages = result["messages"]
        if len(messages) < before + 2:
            # SummarizationMiddleware replaced older messages with a summary.
            self._live_compactions[thread_id] = self._live_compactions.get(thread_id, 0) + 1
        self._live_message_count[thread_id] = len(messages)

        response, prompt_tokens = _last_ai_text_and_prompt_tokens(
            messages, fallback=self._estimate_prompt_context_tokens(user_id, thread_id)
        )
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = self._record(thread_id, message, response, prompt_tokens)
        return {
            "response": response,
            "thread_id": thread_id,
            "mode": "live",
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
            "profile_updates": updates,
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + dynamic
        prompt that injects the profile + summarization middleware.

        Returns None (offline mode) when forced offline, unconfigured, or when
        LangChain / the provider SDK is unavailable.
        """

        if self.force_offline or not self.config.model.is_configured() or ToolRuntime is None:
            return None
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
        except Exception:
            return None

        store = self.profile_store

        @tool
        def read_user_memory(runtime: ToolRuntime[AgentContext]) -> str:
            """Read the current user's User.md profile."""

            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Save one stable fact (name, location, profession, response_style, interests,
            favorite_drink, favorite_food, pet) about the user into User.md. Never save
            questions, jokes, hypotheticals or temporary context."""

            changed = store.upsert_fact(runtime.context.user_id, key.strip().lower(), value.strip(), confidence=0.9)
            return "saved" if changed else "unchanged"

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            profile = store.profile_prompt(request.runtime.context.user_id)
            return "\n\n".join(part for part in (BASE_SYSTEM_PROMPT, MEMORY_GUIDANCE, profile) if part)

        return create_agent(
            model,
            tools=[read_user_memory, save_user_fact],
            middleware=[
                inject_profile,
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
