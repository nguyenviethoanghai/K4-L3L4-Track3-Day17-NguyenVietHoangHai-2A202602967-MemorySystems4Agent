from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

COLUMNS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    details: list[dict[str, Any]] = field(default_factory=list, repr=False)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of conversations")
    return data


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def _hits(answer: str, expected: list[str]) -> int:
    normalized = _norm(answer)
    return sum(1 for item in expected if _norm(item) in normalized)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1].

    - 60% coverage: share of expected facts present
    - 20% concision: full marks up to 60 words, decays after that
    - 20% structure: answer uses bullets (the user's stated preference)
    """

    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    words = len(answer.split())
    concision = 1.0 if words <= 60 else max(0.0, 1 - (words - 60) / 120)
    structure = 1.0 if any(line.lstrip().startswith(("- ", "* ", "• ")) for line in answer.splitlines()) else 0.0
    return round(0.6 * coverage + 0.2 * concision + 0.2 * structure, 3)


def llm_judge_quality(judge, question: str, answer: str, expected: list[str]) -> float | None:
    """Ask the judge model for a 0-10 score; None when the call or parsing fails."""

    prompt = (
        "Chấm điểm câu trả lời của trợ lý từ 0 đến 10 theo: đúng fact mong đợi, ngắn gọn, có cấu trúc.\n"
        f"Câu hỏi: {question}\nFact mong đợi: {', '.join(expected)}\nCâu trả lời: {answer}\n"
        "Chỉ trả về một số nguyên."
    )
    try:
        reply = judge.invoke(prompt)
        match = re.search(r"\d+(?:\.\d+)?", str(getattr(reply, "content", reply)))
        return round(min(float(match.group(0)), 10.0) / 10, 3) if match else None
    except Exception:
        return None


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None) -> BenchmarkRow:
    """Feed every conversation to one agent, then ask recall questions in fresh threads."""

    users = sorted({conv["user_id"] for conv in conversations})
    size_before = {user: agent.memory_file_size(user) for user in users}
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = f"{agent_name.lower()}-{conv['id']}"
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        threads = [thread_id]

        for index, item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            threads.append(recall_thread)
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall = recall_points(answer, expected)
            quality = llm_judge_quality(judge, item["question"], answer, expected) if judge else None
            quality = heuristic_quality(answer, expected) if quality is None else quality
            recall_scores.append(recall)
            quality_scores.append(quality)
            details.append({"conversation": conv["id"], "question": item["question"], "answer": answer, "recall": recall})

        for tid in threads:
            agent_tokens += agent.token_usage(tid)
            prompt_tokens += agent.prompt_token_usage(tid)
            compactions += agent.compaction_count(tid)

    growth = sum(agent.memory_file_size(user) - size_before[user] for user in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=compactions,
        details=details,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ]
        for r in rows
    ]
    if len(rows) == 2 and rows[0].prompt_tokens_processed:
        base, adv = rows
        delta = (adv.prompt_tokens_processed - base.prompt_tokens_processed) / base.prompt_tokens_processed
        table.append(["Advanced vs Baseline", "", f"{delta:+.1%}", f"{adv.recall_score - base.recall_score:+.2f}", "", "", ""])
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github", colalign=("left",) + ("right",) * 6)
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(title: str, dataset: Path, config, live: bool, judge=None, show_details: bool = False) -> str:
    conversations = load_conversations(dataset)
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=not live), conversations, config, judge),
        run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=not live), conversations, config, judge),
    ]
    parts = [f"## {title}", "", f"Dataset: `{dataset.relative_to(config.base_dir).as_posix()}` "
             f"({len(conversations)} conversation(s), {sum(len(c['turns']) for c in conversations)} turns)", "",
             format_rows(rows)]
    if show_details:
        for row in rows:
            parts += ["", f"### {row.agent_name} — recall answers", ""]
            for d in row.details:
                answer = d["answer"].replace("\n", " ")
                parts.append(f"- [{d['conversation']}] recall={d['recall']} — Q: {d['question']} → A: {answer}")
    return "\n".join(parts)


def main() -> None:
    """Run the Standard and Long-Context Stress benchmarks for Baseline vs Advanced."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark")
    parser.add_argument("--live", action="store_true", help="use the configured LLM provider instead of offline mode")
    parser.add_argument("--judge", action="store_true", help="score response quality with the judge model (needs credentials)")
    parser.add_argument("--details", action="store_true", help="print every recall answer")
    parser.add_argument("--output", type=Path, help="also write the report to this markdown file")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    # Isolated, fresh state so every run starts with empty User.md files.
    bench_state = config.state_dir / "benchmark"
    shutil.rmtree(bench_state, ignore_errors=True)
    bench_state.mkdir(parents=True)
    config = replace(config, state_dir=bench_state)

    judge = None
    if args.judge and config.judge_model.is_configured():
        from model_provider import build_chat_model

        judge = build_chat_model(config.judge_model)

    mode = "live" if args.live else "offline (deterministic)"
    report = "\n\n".join([
        f"# Day 17 Memory Benchmark — mode: {mode}, quality: {'LLM judge' if judge else 'heuristic'}, "
        f"compact threshold: {config.compact_threshold_tokens} tokens, keep: {config.compact_keep_messages} messages",
        run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, args.live, judge, args.details),
        run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config, args.live, judge, args.details),
    ])
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
