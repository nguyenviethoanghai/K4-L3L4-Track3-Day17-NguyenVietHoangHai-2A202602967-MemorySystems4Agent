from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Fact schema
# ---------------------------------------------------------------------------

# Scalar facts: a newer, confident value replaces the old one (conflict handling).
SCALAR_FACTS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")
# List facts: items are merged; the least recently mentioned items decay out once the cap is hit.
LIST_FACTS = {"response_style": 6, "interests": 6}
FACT_ORDER = ("name", "location", "profession", "response_style", "interests", "favorite_drink", "favorite_food", "pet")
LIST_SEPARATOR = "; "
MAX_SUPERSEDED = 5


@dataclass
class FactCandidate:
    """One fact pulled from a user message, with how sure the extractor is."""

    key: str
    value: str | list[str]
    confidence: float
    evidence: str = ""


@dataclass
class FactEntry:
    value: str
    confidence: float = 1.0
    seen: int = 1

    def items(self) -> list[str]:
        return [item.strip() for item in self.value.split(LIST_SEPARATOR) if item.strip()]


def apply_candidates(
    entries: dict[str, FactEntry],
    candidates: list[FactCandidate],
    threshold: float,
    superseded: list[str] | None = None,
) -> list[str]:
    """Merge extracted candidates into a fact dict in place.

    - candidates below `threshold` are ignored (confidence threshold)
    - a scalar fact with a different value is replaced and the old value is
      recorded in `superseded` (conflict handling: newest correction wins)
    - list facts move re-mentioned items to the end and drop the oldest ones
      past the cap (recency-based memory decay)

    Returns human-readable change descriptions.
    """

    changes: list[str] = []
    for cand in candidates:
        if cand.confidence < threshold:
            continue
        current = entries.get(cand.key)

        if cand.key in LIST_FACTS:
            new_items = list(cand.value) if isinstance(cand.value, list) else [cand.value]
            items = current.items() if current else []
            for item in new_items:
                if item.endswith(" bullet"):
                    # "3 bullet" is a sharper version of a generic "dạng bullet".
                    items = [i for i in items if i != "dạng bullet"]
                elif item == "dạng bullet" and any(i.endswith(" bullet") for i in items):
                    continue
                if item in items:
                    items.remove(item)
                items.append(item)
            items = items[-LIST_FACTS[cand.key]:]
            value = LIST_SEPARATOR.join(items)
            if current and current.value == value:
                current.seen += 1
                continue
            entries[cand.key] = FactEntry(value, max(cand.confidence, current.confidence if current else 0), (current.seen + 1) if current else 1)
            changes.append(f"{cand.key}: {value}")
            continue

        value = str(cand.value)
        if current and current.value.casefold() == value.casefold():
            current.seen += 1
            current.confidence = max(current.confidence, cand.confidence)
            continue
        if current and superseded is not None:
            superseded.append(f"{cand.key}: {current.value} → {value}")
            del superseded[:-MAX_SUPERSEDED]
        entries[cand.key] = FactEntry(value, cand.confidence, 1)
        changes.append(f"{cand.key}: {value}")
    return changes


# ---------------------------------------------------------------------------
# User.md persistent store
# ---------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^- (?P<key>[a-z_]+): (?P<value>.*?)(?:\s*<!--\s*conf=(?P<conf>[\d.]+)\s+seen=(?P<seen>\d+)\s*-->)?\s*$")
_PROFILE_HEADER = "## Profile"
_SUPERSEDED_HEADER = "## Superseded"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` — one markdown file per user.

    Layout::

        # User.md — <user_id>
        ## Profile
        - name: DũngCT <!-- conf=0.95 seen=2 -->
        ...
        ## Superseded
        - location: Đà Nẵng → Huế

    Only `## Profile` is injected into prompts / used for answers. `## Superseded`
    keeps a short audit trail of corrections. Any other section a human adds is
    preserved verbatim.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip()).strip("._") or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def default_text(self, user_id: str) -> str:
        return f"# User.md — {user_id}\n\n{_PROFILE_HEADER}\n"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return self.default_text(user_id)
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def exists(self, user_id: str) -> bool:
        return self.path_for(user_id).exists()

    # -- structured view ---------------------------------------------------

    def _parse(self, user_id: str) -> tuple[dict[str, FactEntry], list[str], list[str]]:
        entries: dict[str, FactEntry] = {}
        superseded: list[str] = []
        extra: list[str] = []
        section = None
        for line in self.read_text(user_id).splitlines():
            if line.startswith("## "):
                section = line.strip()
                if section not in (_PROFILE_HEADER, _SUPERSEDED_HEADER):
                    extra.append(line)
                continue
            if section == _PROFILE_HEADER:
                match = _FACT_LINE.match(line)
                if match:
                    entries[match["key"]] = FactEntry(
                        value=match["value"].strip(),
                        confidence=float(match["conf"] or 1.0),
                        seen=int(match["seen"] or 1),
                    )
            elif section == _SUPERSEDED_HEADER:
                if line.startswith("- "):
                    superseded.append(line[2:].strip())
            elif section is not None:
                extra.append(line)
        return entries, superseded, extra

    def _render(self, user_id: str, entries: dict[str, FactEntry], superseded: list[str], extra: list[str]) -> str:
        lines = [f"# User.md — {user_id}", "", _PROFILE_HEADER]
        ordered = [k for k in FACT_ORDER if k in entries] + sorted(k for k in entries if k not in FACT_ORDER)
        for key in ordered:
            entry = entries[key]
            lines.append(f"- {key}: {entry.value} <!-- conf={entry.confidence:.2f} seen={entry.seen} -->")
        if superseded:
            lines += ["", _SUPERSEDED_HEADER] + [f"- {item}" for item in superseded]
        if extra:
            lines += [""] + extra
        return "\n".join(lines).rstrip() + "\n"

    def fact_entries(self, user_id: str) -> dict[str, FactEntry]:
        return self._parse(user_id)[0]

    def facts(self, user_id: str) -> dict[str, str]:
        return {key: entry.value for key, entry in self.fact_entries(user_id).items()}

    def superseded(self, user_id: str) -> list[str]:
        return self._parse(user_id)[1]

    def upsert_fact(self, user_id: str, key: str, value: str | list[str], confidence: float = 1.0) -> bool:
        """Insert/update one fact; returns True when User.md changed."""

        return bool(self.apply_candidates(user_id, [FactCandidate(key, value, confidence)], threshold=0.0))

    def apply_candidates(self, user_id: str, candidates: list[FactCandidate], threshold: float) -> list[str]:
        entries, superseded, extra = self._parse(user_id)
        before = {k: (v.value, v.seen) for k, v in entries.items()}
        changes = apply_candidates(entries, candidates, threshold, superseded)
        after = {k: (v.value, v.seen) for k, v in entries.items()}
        if after != before or not self.exists(user_id):
            self.write_text(user_id, self._render(user_id, entries, superseded, extra))
        return changes

    def profile_prompt(self, user_id: str) -> str:
        """The part of User.md that is injected into the prompt (no audit trail)."""

        entries = self.fact_entries(user_id)
        if not entries:
            return ""
        ordered = [k for k in FACT_ORDER if k in entries] + sorted(k for k in entries if k not in FACT_ORDER)
        return "User profile (User.md):\n" + "\n".join(f"- {k}: {entries[k].value}" for k in ordered)


# ---------------------------------------------------------------------------
# Rule-based extraction of stable facts from Vietnamese messages
# ---------------------------------------------------------------------------

CITIES = (
    "Hồ Chí Minh", "TP.HCM", "Sài Gòn", "Hà Nội", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ",
    "Nha Trang", "Đà Lạt", "Quy Nhơn", "Vũng Tàu", "Hội An", "Hạ Long", "Vinh", "Biên Hòa",
    "Buôn Ma Thuột", "Bắc Ninh", "Quảng Ngãi",
)
DRINKS = ("cà phê sữa đá", "cà phê muối", "cà phê đen", "bạc xỉu", "trà sữa", "trà đá", "trà đào", "nước cam", "cà phê", "trà")
TECH_TOPICS = (
    "async Python", "Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "evaluation",
    "memory architecture", "benchmark memory", "LangChain", "LangGraph", "LLM", "TypeScript", "Rust",
)
STYLE_RULES: tuple[tuple[str, str | None], ...] = (
    (r"(\d+)\s*bullet", None),  # "3 bullet"
    (r"bullet", "dạng bullet"),
    (r"ngắn|gọn", "ngắn gọn"),
    (r"rõ ý", "rõ ý"),
    (r"có cấu trúc", "có cấu trúc"),
    (r"ví dụ (?:thực tế|thực chiến)", "có ví dụ thực tế"),
    (r"trade-off", "nhấn trade-off"),
    (r"số liệu|định lượng", "có số liệu minh họa"),
)


def _alternation(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_CITY = _alternation(CITIES)
_DRINK = _alternation(DRINKS)
_TECH_RE = re.compile(rf"(?<!\w)(?:{_alternation(TECH_TOPICS)})(?!\w)", re.IGNORECASE)
_ROLE = r"(?:[A-Za-z][A-Za-z0-9+.-]*\s)?(?:engineer|developer|scientist|manager|designer|analyst|researcher|architect)"

_LOCATION_PATTERNS = (
    (re.compile(rf"nơi ở(?: hiện tại)?(?: của mình)?\s+(?:vẫn\s+)?(?:là|:)\s+({_CITY})", re.I), 0.95),
    (re.compile(rf"nơi ở.*?từ\s+(?:{_CITY})\s+sang\s+({_CITY})", re.I), 0.9),
    (re.compile(rf"\b(?:mình|tôi|em)(?:\s+\S+){{0,4}}?\s+(?:ở|tại|sống ở)\s+({_CITY})", re.I), 0.85),
    (re.compile(rf"\bhiện\s+(?:đang\s+)?(?:ở|sống ở)\s+({_CITY})", re.I), 0.85),
    # A bare "ở <city>" could be a trip or an example: keep it below the default threshold.
    (re.compile(rf"\bở\s+({_CITY})", re.I), 0.5),
)
_PROFESSION_PATTERNS = (
    (re.compile(rf"\bnghề(?: nghiệp)?(?: hiện tại)?(?:\s+thì)?(?:\s+vẫn)?(?:\s+là|:)?\s+({_ROLE})", re.I), 0.95),
    (re.compile(rf"\b(?:làm|chuyển sang|là)\s+(?:một\s+)?({_ROLE})", re.I), 0.85),
)
_NAME_PATTERNS = (
    (re.compile(r"\btên\s+(?:mình|tôi|em)?\s*(?:là|:)\s+", re.I), 0.95),
    (re.compile(r"^(?:và\s+)?tên\s+(?=[A-ZĐ])"), 0.8),  # "tên DũngCT Stress" at clause start
)
_DRINK_EXPLICIT = re.compile(r"đồ uống\s+(?:yêu thích|ưa thích|ruột)(?: của mình)?\s+(?:vẫn\s+)?(?:là|:)\s+([^.,;!?]+)", re.I)
_DRINK_IMPLICIT = re.compile(rf"\b(?:mình|tôi|em)\s+(?:vẫn\s+|hay\s+|thường\s+)?uống\s+({_DRINK})", re.I)
_FOOD_EXPLICIT = re.compile(r"món\s+(?:ăn\s+)?(?:yêu thích|ưa thích|ruột)(?: của mình)?\s+(?:vẫn\s+)?(?:là|:)\s+([^.,;!?]+)", re.I)
_PET = re.compile(r"\bnuôi\s+(?:một\s+)?(?:bé|con|chú|em)?\s*([^\s.,;!?]+)(?:\s+tên\s+([^\s.,;!?]+))?", re.I)

_CLAUSE_SPLIT = re.compile(r"[,;:]|\s(?:chứ|nhưng|dù|tuy nhiên|song)\s", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Clause-level cues that the fact mentioned is NOT the current truth.
NEGATION_CUES = (
    "không còn", "không phải", "chỉ là", "đùa", "trước đó", "lúc đầu", "đừng", "thông tin cũ", "ví dụ cũ", "không lấy",
)
# Cues that lower confidence (the user is unsure / speculating).
HEDGE_CUES = ("có lẽ", "hình như", "hay là", "chắc là", "đang cân nhắc", "dự định", "định chuyển", "biết đâu")
QUESTION_PREFIXES = ("bạn có biết", "bạn có thể nhắc lại", "bạn thử nhớ lại", "nhắc lại giúp", "cho mình hỏi", "mình tên gì")
QUESTION_CUES = ("là gì", "ở đâu", "con gì", "như thế nào", "ra sao")
PREFERENCE_CUES = ("muốn", "thích", "hãy", "giữ", "ưu tiên", "nên", "nhớ", "mong")
INTEREST_CUES = ("thích", "quan tâm", "đang học", "học thêm", "ôn lại", "đang đọc về", "tìm hiểu")


def split_sentences(message: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(_nfc(message).strip()) if s.strip()]


def is_question(sentence: str) -> bool:
    """A sentence that asks for information should never be stored as a fact."""

    lowered = sentence.strip().lower()
    if "?" in lowered:
        return True
    if lowered.startswith(QUESTION_PREFIXES):
        return True
    return any(cue in lowered for cue in QUESTION_CUES)


def _clause_is_negated(clause: str) -> bool:
    lowered = clause.strip().lower()
    return lowered.startswith("nếu") or any(cue in lowered for cue in NEGATION_CUES)


def _hedge_penalty(sentence: str) -> float:
    lowered = sentence.lower()
    return 0.35 if any(cue in lowered for cue in HEDGE_CUES) else 0.0


def _canonical(value: str, vocabulary: tuple[str, ...]) -> str:
    for word in vocabulary:
        if word.casefold() == value.strip().casefold():
            return word
    return value.strip()


def _best(candidates: list[FactCandidate]) -> FactCandidate | None:
    """Highest confidence wins; on ties the later mention (the correction) wins."""

    best = None
    for cand in candidates:
        if best is None or cand.confidence >= best.confidence:
            best = cand
    return best


def _capitalized_run(text: str, max_words: int = 4) -> str:
    words: list[str] = []
    for raw in text.split()[:max_words]:
        word = raw.strip(".,;:!?\"'()")
        if not word or not word[0].isupper():
            break
        words.append(word)
        if raw != word and raw[len(word):][:1] in ".,;:!?":
            break
    return " ".join(words)


def _style_tags(sentence: str) -> list[str]:
    lowered = sentence.lower()
    tags: list[str] = []
    for pattern, tag in STYLE_RULES:
        match = re.search(pattern, lowered)
        if not match:
            continue
        label = f"{match.group(1)} bullet" if tag is None else tag
        if label == "dạng bullet" and any(t.endswith(" bullet") for t in tags):
            continue
        if label not in tags:
            tags.append(label)
    return tags


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Structured entity extraction with a confidence score per fact.

    Guards:
    - question sentences are skipped (asking ≠ telling)
    - clauses with negation / "only a joke" / conditional cues are skipped
    - hedged statements get a confidence penalty
    - bare mentions without a clear "I live / I work as" frame get low confidence
    """

    per_key: dict[str, list[FactCandidate]] = {}
    styles: list[str] = []
    interests: list[str] = []

    def add(key: str, value: str | list[str], confidence: float, evidence: str) -> None:
        per_key.setdefault(key, []).append(FactCandidate(key, value, round(min(confidence, 1.0), 2), evidence))

    for sentence in split_sentences(message):
        if is_question(sentence):
            continue
        penalty = _hedge_penalty(sentence)
        lowered = sentence.lower()

        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip()
            if not clause or _clause_is_negated(clause):
                continue

            for pattern, conf in _NAME_PATTERNS:
                match = pattern.search(clause)
                if match:
                    name = _capitalized_run(clause[match.end():])
                    if name:
                        add("name", name, conf - penalty, clause)
                        break

            for pattern, conf in _LOCATION_PATTERNS:
                match = pattern.search(clause)
                if match:
                    add("location", _canonical(match.group(1), CITIES), conf - penalty, clause)
                    break

            for pattern, conf in _PROFESSION_PATTERNS:
                match = pattern.search(clause)
                if match:
                    add("profession", match.group(1).strip(), conf - penalty, clause)
                    break

            match = _DRINK_EXPLICIT.search(clause)
            if match:
                add("favorite_drink", _canonical(match.group(1), DRINKS), 0.95 - penalty, clause)
            else:
                match = _DRINK_IMPLICIT.search(clause)
                if match:
                    add("favorite_drink", _canonical(match.group(1), DRINKS), 0.75 - penalty, clause)

            match = _FOOD_EXPLICIT.search(clause)
            if match:
                add("favorite_food", match.group(1).strip(), 0.95 - penalty, clause)

            match = _PET.search(clause)
            if match:
                pet = match.group(1) + (f" tên {match.group(2)}" if match.group(2) else "")
                add("pet", pet, 0.9 - penalty, clause)

        if lowered.startswith("nếu"):
            continue
        if any(cue in lowered for cue in ("trả lời", "style", "giải thích")) and any(c in lowered for c in PREFERENCE_CUES):
            for tag in _style_tags(sentence):
                if tag not in styles:
                    styles.append(tag)
        if any(cue in lowered for cue in INTEREST_CUES) and "không thích" not in lowered:
            for match in _TECH_RE.finditer(sentence):
                topic = _canonical(match.group(0), TECH_TOPICS)
                if topic not in interests:
                    interests.append(topic)

    results = [cand for cand in (_best(c) for c in per_key.values()) if cand is not None]
    if styles:
        results.append(FactCandidate("response_style", styles, 0.85, "style"))
    if interests:
        results.append(FactCandidate("interests", interests, 0.8, "interests"))
    return results


def extract_profile_updates(message: str, threshold: float = 0.7) -> dict[str, str]:
    """Stable profile facts confidently present in `message` (key -> value)."""

    updates: dict[str, str] = {}
    for cand in extract_profile_candidates(message):
        if cand.confidence >= threshold:
            updates[cand.key] = LIST_SEPARATOR.join(cand.value) if isinstance(cand.value, list) else cand.value
    return updates


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def _gist(text: str, max_words: int = 25) -> str:
    """Pick the most information-dense sentence (names, numbers) and clip it."""

    sentences = split_sentences(text) or [text.strip()]

    def salience(sentence: str) -> int:
        words = sentence.split()
        return sum(1 for w in words[1:] if any(ch.isupper() or ch.isdigit() for ch in w))

    best = max(sentences, key=salience) if len(sentences) > 1 else sentences[0]
    words = best.split()
    return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: one gist bullet per user message, newest `max_items` kept."""

    bullets = [f"- {_gist(m['content'])}" for m in messages if m.get("role") == "user" and m.get("content", "").strip()]
    return "\n".join(bullets[-max_items:])


@dataclass
class CompactMemoryManager:
    """Short-term memory with compaction for long threads.

    - recent messages are kept verbatim
    - when summary + messages exceed `threshold_tokens`, everything except the
      last `keep_messages` is folded into a bounded bullet summary
    - `compactions` counts how often that happened (for benchmarking)
    """

    threshold_tokens: int
    keep_messages: int
    max_summary_items: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        state = self.context(thread_id)
        state["messages"].append({"role": role, "content": content})
        self._maybe_compact(state)

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0, "compacted_messages": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])

    def context_tokens(self, thread_id: str) -> int:
        state = self.context(thread_id)
        return estimate_tokens(state["summary"]) + sum(estimate_tokens(m["content"]) for m in state["messages"])

    def render(self, thread_id: str) -> str:
        state = self.context(thread_id)
        parts = []
        if state["summary"]:
            parts.append("Summary of earlier conversation:\n" + state["summary"])
        parts += [f"{m['role']}: {m['content']}" for m in state["messages"]]
        return "\n".join(parts)

    def _maybe_compact(self, state: dict[str, object]) -> None:
        messages: list[dict[str, str]] = state["messages"]
        total = estimate_tokens(state["summary"]) + sum(estimate_tokens(m["content"]) for m in messages)
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        cut = len(messages) - self.keep_messages
        old, recent = messages[:cut], messages[cut:]
        previous = [line for line in str(state["summary"]).splitlines() if line.strip()]
        new = [line for line in summarize_messages(old, max_items=self.max_summary_items).splitlines() if line.strip()]
        state["summary"] = "\n".join((previous + new)[-self.max_summary_items:])
        state["messages"] = recent
        state["compactions"] += 1
        state["compacted_messages"] += len(old)
