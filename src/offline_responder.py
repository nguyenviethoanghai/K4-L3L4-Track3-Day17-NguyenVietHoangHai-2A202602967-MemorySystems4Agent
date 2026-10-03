"""Deterministic responder shared by both agents in offline mode.

Both agents answer with the exact same template; the only difference is where
the facts come from (Baseline: the current thread, Advanced: User.md + thread).
That keeps the benchmark a fair comparison of memory, not of phrasing.
"""

from __future__ import annotations

import re

from memory_store import FACT_ORDER, split_sentences

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "interests": "Mối quan tâm kỹ thuật",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}

# Which facts a question is asking about.
TOPIC_PATTERNS = (
    ("name", r"\btên\b|là ai"),
    ("location", r"ở đâu|nơi ở|còn ở"),
    ("profession", r"nghề|làm gì|công việc"),
    ("favorite_drink", r"đồ uống|uống gì"),
    ("favorite_food", r"món ăn|ăn gì"),
    ("pet", r"nuôi|con gì|thú cưng"),
    ("response_style", r"style|kiểu trả lời|cách trả lời|trả lời mình thích|trả lời như thế nào"),
    ("interests", r"quan tâm|là ai|sở thích"),
)
RECALL_CUES = re.compile(r"\?|nhắc lại|nhớ lại|tóm tắt|mô tả|là ai|bạn có biết", re.IGNORECASE)


def requested_facts(message: str) -> list[str]:
    """Fact keys the message asks about (empty when it is not a recall request)."""

    lowered = message.lower()
    if not RECALL_CUES.search(lowered):
        return []
    keys = {key for key, pattern in TOPIC_PATTERNS if re.search(pattern, lowered)}
    if "tóm tắt" in lowered and "về mình" in lowered:
        keys |= {"name", "profession", "interests"}
    return [key for key in FACT_ORDER if key in keys]


def compose_reply(message: str, facts: dict[str, str], updates: list[str] | None = None) -> str:
    """Short, bullet-style reply built only from the facts the agent can see."""

    keys = requested_facts(message)
    if keys:
        lines = ["Theo những gì mình đang nhớ:"]
        for key in keys:
            value = facts.get(key)
            lines.append(f"- {FACT_LABELS[key]}: {value if value else 'chưa có thông tin trong memory'}")
        return "\n".join(lines)

    first = split_sentences(message)[:1]
    topic = " ".join(first[0].split()[:8]) if first else ""
    reply = f"Đã ghi nhận: {topic}…" if topic else "Đã ghi nhận."
    if updates:
        reply += "\n- Đã cập nhật memory: " + "; ".join(updates)
    return reply
