from __future__ import annotations

import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, recall_points
from config import LabConfig, load_config
from memory_store import LIST_FACTS, CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig, normalize_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_config(tmp_path: Path, threshold: int = 800, keep: int = 4) -> LabConfig:
    """Isolated config: state under tmp_path, no provider credentials (offline)."""

    offline = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=offline,
        judge_model=offline,
    )


def load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def feed(agent, user_id: str, thread_id: str, turns: list[str]) -> None:
    for turn in turns:
        agent.reply(user_id, thread_id, turn)


# --- required tests ---------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user = "dungct"

    assert store.file_size(user) == 0
    assert "## Profile" in store.read_text(user)  # default template before the file exists

    path = store.write_text(user, "# User.md — dungct\n\n## Profile\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path == store.path_for(user) and path.name == "User.md" and path.exists()
    assert store.facts(user) == {"name": "DũngCT", "location": "Đà Nẵng"}

    assert store.edit_text(user, "Đà Nẵng", "Huế") is True
    assert store.facts(user)["location"] == "Huế"
    assert store.edit_text(user, "Sài Gòn", "Hà Nội") is False  # nothing to replace

    assert store.upsert_fact(user, "profession", "MLOps engineer") is True
    assert store.facts(user)["profession"] == "MLOps engineer"
    assert store.file_size(user) == path.stat().st_size > 0

    # a hand-written section survives structured updates
    store.write_text(user, store.read_text(user) + "\n## Notes\n- viết tay\n")
    store.upsert_fact(user, "pet", "corgi tên Bơ")
    assert "## Notes\n- viết tay" in store.read_text(user)


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path, threshold=200, keep=4), force_offline=True)
    stress = load("advanced_long_context.json")[0]
    feed(agent, stress["user_id"], "long", stress["turns"])

    state = agent.compact_memory.context("long")
    assert agent.compaction_count("long") >= 3
    assert len(state["messages"]) <= 4 + 1  # kept window (+ the turn that just arrived)
    assert state["summary"].strip()
    assert state["compacted_messages"] > 0

    # a short thread never compacts
    short = AdvancedAgent(make_config(tmp_path / "short"), force_offline=True)
    feed(short, "u", "short", ["Mình tên là An.", "Mình ở Huế."])
    assert short.compaction_count("short") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    conv = load("conversations.json")[0]
    question = conv["recall_questions"][0]

    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    feed(advanced, conv["user_id"], "session-1", conv["turns"])
    feed(baseline, conv["user_id"], "session-1", conv["turns"])

    adv_answer = advanced.reply(conv["user_id"], "session-2", question["question"])["response"]
    base_answer = baseline.reply(conv["user_id"], "session-2", question["question"])["response"]
    assert recall_points(adv_answer, question["expected_contains"]) == 1.0
    assert recall_points(base_answer, question["expected_contains"]) == 0.0

    # survives a brand-new process: a fresh agent reading the same state dir
    restarted = AdvancedAgent(config, force_offline=True)
    answer = restarted.reply(conv["user_id"], "session-3", question["question"])["response"]
    assert "DũngCT" in answer and "cà phê sữa đá" in answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    stress = load("advanced_long_context.json")[0]
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    feed(baseline, stress["user_id"], "stress", stress["turns"])
    feed(advanced, stress["user_id"], "stress", stress["turns"])

    assert advanced.compaction_count("stress") > 0
    assert advanced.prompt_token_usage("stress") < 0.7 * baseline.prompt_token_usage("stress")
    # per-turn context stays bounded instead of growing with the thread
    assert advanced.compact_memory.context_tokens("stress") <= config.compact_threshold_tokens


# --- behaviour beyond the happy path ------------------------------------------


def test_short_conversation_advanced_costs_more_prompt(tmp_path: Path) -> None:
    """Honest trade-off: without compaction, carrying User.md is pure overhead."""

    config = make_config(tmp_path)
    conv = load("conversations.json")[0]
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    feed(baseline, conv["user_id"], "t", conv["turns"])
    feed(advanced, conv["user_id"], "t", conv["turns"])

    assert advanced.compaction_count("t") == 0
    assert advanced.prompt_token_usage("t") > baseline.prompt_token_usage("t")


def test_baseline_remembers_within_thread_only(tmp_path: Path) -> None:
    baseline = BaselineAgent(make_config(tmp_path), force_offline=True)
    baseline.reply("u", "t1", "Mình tên là DũngCT.")
    assert "DũngCT" in baseline.reply("u", "t1", "Mình tên gì?")["response"]
    assert "DũngCT" not in baseline.reply("u", "t2", "Mình tên gì?")["response"]
    assert baseline.memory_file_size("u") == 0
    assert not (tmp_path / "state" / "profiles").exists()


def test_correction_replaces_old_fact_and_ignores_noise(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    stress = load("advanced_long_context.json")[0]
    feed(agent, stress["user_id"], "stress", stress["turns"])

    facts = agent.profile_store.facts(stress["user_id"])
    assert facts["location"] == "Đà Nẵng"  # corrected from Huế; Hà Nội was only a trip
    assert facts["profession"] == "MLOps engineer"  # "product manager" was a joke
    assert facts["name"] == "DũngCT Stress"
    assert "location: Huế → Đà Nẵng" in agent.profile_store.superseded(stress["user_id"])

    text = agent.profile_store.read_text(stress["user_id"])
    profile_section = text.split("## Superseded")[0]
    assert "Hà Nội" not in text and "product manager" not in text
    assert "Huế" not in profile_section  # old value only lives in the audit trail


def test_questions_hedges_and_conditionals_are_not_stored() -> None:
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Hiện tại mình đang ở đâu?") == {}
    # hedged statement falls below the confidence threshold
    assert "profession" not in extract_profile_updates("Có lẽ mình sẽ làm product manager.")
    # conditional / negated clauses
    assert "location" not in extract_profile_updates(
        "Nếu sau này mình có nhắc lại Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại."
    )
    assert extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.") == {
        "profession": "MLOps engineer"
    }
    # bare mention without a "mình ở" frame stays below threshold
    assert "location" not in extract_profile_updates("Tuần sau có hội thảo ở Hà Nội.")


def test_list_facts_decay_oldest_items(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    cap = LIST_FACTS["interests"]
    for i in range(cap + 3):
        store.upsert_fact("u", "interests", [f"topic-{i}"])
    store.upsert_fact("u", "interests", ["topic-3"])  # re-mention refreshes recency

    items = store.facts("u")["interests"].split("; ")
    assert len(items) == cap
    assert "topic-0" not in items and items[-1] == "topic-3"


def test_compact_manager_keeps_recent_messages_verbatim() -> None:
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    for i in range(6):
        manager.append("t", "user", f"Tin số {i}: " + "nội dung dài " * 10)
    state = manager.context("t")
    assert [m["content"][:8] for m in state["messages"]] == ["Tin số 4", "Tin số 5"]
    assert manager.compaction_count("t") >= 1
    assert "Tin số 0" in state["summary"]


def test_benchmark_scoring_helpers() -> None:
    assert recall_points("DũngCT thích cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("DũngCT", ["DũngCT", "cà phê sữa đá"]) == 0.5
    assert recall_points("không biết", ["DũngCT"]) == 0.0
    assert heuristic_quality("- Tên: DũngCT", ["DũngCT"]) == 1.0
    assert heuristic_quality("chưa có thông tin", ["DũngCT"]) < 0.5


def test_provider_aliases_and_config(tmp_path: Path, monkeypatch) -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Open-Router") == "openrouter"
    for provider in ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter"):
        assert normalize_provider(provider) == provider

    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "321")
    config = load_config(tmp_path)
    assert config.model.provider == "anthropic" and config.model.is_configured()
    assert config.judge_model.provider == "anthropic"
    assert config.compact_threshold_tokens == 321
    assert config.state_dir.exists()
