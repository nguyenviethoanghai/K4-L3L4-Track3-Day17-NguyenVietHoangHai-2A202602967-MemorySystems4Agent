from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import DEFAULT_MODELS, ProviderConfig, normalize_provider

# Which env var holds the API key / base URL for each provider.
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}

DEFAULT_COMPACT_THRESHOLD_TOKENS = 800
DEFAULT_COMPACT_KEEP_MESSAGES = 4
DEFAULT_PROFILE_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths: repo root, dataset directory, state directory.
    - Compact memory: token threshold that triggers compaction and how many
      recent messages are kept verbatim afterwards.
    - Persistent memory: minimum confidence before a fact is written to User.md.
    - Providers: the main chat model and a judge model (used by the benchmark).
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = DEFAULT_PROFILE_CONFIDENCE_THRESHOLD


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(root / ".env", override=False)


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<prefix>_PROVIDER`, `<prefix>_MODEL`, `<prefix>_TEMPERATURE` from env."""

    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv(f"{prefix}_MODEL")
    if not model_name:
        model_name = fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0"))

    key_env = API_KEY_ENV[provider]
    api_key = os.getenv(key_env) if key_env else None
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    url_env = BASE_URL_ENV.get(provider)
    base_url = os.getenv(url_env) if url_env else None

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables (optionally from `.env`) and return a LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = Path(os.getenv("LAB_STATE_DIR", str(root / "state"))).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS)),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES)),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=float(
            os.getenv("PROFILE_CONFIDENCE_THRESHOLD", DEFAULT_PROFILE_CONFIDENCE_THRESHOLD)
        ),
    )
