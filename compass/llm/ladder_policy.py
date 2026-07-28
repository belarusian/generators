"""Model configuration.

One family, many roles. A Family is the complete behavioral signature --
every model role visible in one place. Select with COMPASS_FAMILY, or
override individual roles with env vars.

Switch at runtime with /model.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


# --- Family: the complete behavioral signature ---

@dataclass(frozen=True)
class Family:
    """Every model role in one place.

    The worker is the only required field. Other roles fall back to worker
    when not specified (empty string). Embedding has no worker fallback --
    it's a fundamentally different capability.
    """
    worker: str                        # coding, acting, default for all roles
    oracle: str = ""                   # wisdom (/ask) -- falls back to worker
    vision: str = ""                   # image understanding -- falls back to worker
    embedding: str = ""                # RAG retrieval -- no worker fallback
    learning: str = ""                 # self-reflection -- falls back to worker
    observer: str = ""                 # stall/loop detection -- falls back to worker
    critic: str = ""                   # reviews work -- falls back to observer -> worker


FAMILIES: Dict[str, Family] = {
    "qwen": Family(
        worker    = "qwen3-coder:latest@local",
        oracle    = "qwen3-coder-next:q8_0@big",
        vision    = "qwen3-vl:30b@local",
        embedding = "qwen3-embedding:4b@local",
        learning  = "qwen3-coder:latest@local",
        observer  = "qwen3-coder:latest@local",
        critic    = "qwen3-coder:latest@local",
    ),
    "neo": Family(
        worker    = "qwen3-coder-next:latest@big",
        oracle    = "qwen3-coder-next:latest@big",
        vision    = "qwen3.5:122b@big",
        embedding = "qwen3-embedding:4b@local",
        learning  = "qwen3-coder:latest@local",
        observer  = "qwen3-coder:latest@local",
        critic    = "qwen3-coder:latest@local",
    ),
    "trinity": Family(
        worker    = "qwen3.5:122b@big",
        oracle    = "qwen3.5:122b@big",
        vision    = "qwen3-vl:30b@local",
        embedding = "qwen3-embedding:4b@local",
        learning  = "qwen3-coder:latest@local",
        observer  = "qwen3-coder:latest@local",
        critic    = "qwen3-coder:latest@local",
    ),
    # llama.cpp direct -- no middleman
    # Each role maps to a dedicated llama-server instance.
    # Configure servers via LLAMACPP_SERVERS env var.
    "llamacpp": Family(
        worker    = "llamacpp@coder",
        oracle    = "llamacpp@oracle",
        vision    = "llamacpp@vision",
        embedding = "llamacpp@embed",
        learning  = "llamacpp@coder",
        observer  = "llamacpp@coder",
        critic    = "llamacpp@coder",
    ),
}

DEFAULT_FAMILY = "qwen"


# --- Resolution: env var > family field > fallback ---

_BARE = Family(worker="")  # no family -- everything falls through to env vars


def _get_family() -> Family:
    """Resolve the active family.

    Unset COMPASS_FAMILY -> default family.
    Unknown COMPASS_FAMILY -> bare (all roles fall through to env vars).
    """
    name = os.getenv("COMPASS_FAMILY")
    if name is None:
        return FAMILIES[DEFAULT_FAMILY]
    return FAMILIES.get(name.lower(), _BARE)


def get_model_spec() -> str:
    """Get the worker model spec.

    Resolution: COMPASS_MODEL env var > family worker > default family worker.
    """
    return os.getenv("COMPASS_MODEL") or _get_family().worker or FAMILIES[DEFAULT_FAMILY].worker


def get_oracle_model_spec() -> str:
    """Get the oracle model spec (for wisdom/dream tasks).

    Resolution: ORACLE_MODEL env var > family oracle > COMPASS_MODEL > family worker.
    """
    explicit = os.getenv("ORACLE_MODEL")
    if explicit:
        return explicit
    family = _get_family()
    return family.oracle or get_model_spec()


def get_vision_model_spec() -> str:
    """Get the vision model spec (for image understanding).

    Resolution: VISION_MODEL env var > family vision > COMPASS_MODEL > family worker.
    """
    explicit = os.getenv("VISION_MODEL")
    if explicit:
        return explicit
    family = _get_family()
    return family.vision or get_model_spec()


def get_learning_model_spec() -> str:
    """Get the learning/self-reflection model spec.

    Resolution: LEARNING_MODEL env var > family learning > worker.
    High-volume, low-stakes -- use a fast cheap model.
    """
    return os.getenv("LEARNING_MODEL") or _get_family().learning or get_model_spec()


def get_loop_observer_spec() -> str:
    """Get the loop observer / progress assessor model spec.

    Resolution: LOOP_OBSERVER env var > family observer > worker.
    """
    return os.getenv("LOOP_OBSERVER") or _get_family().observer or get_model_spec()


def get_critic_model_spec() -> str:
    """Get the critic model spec (reviews actor/programmer work).

    Resolution: CRITIC_MODEL env var > family critic > loop observer > worker.
    Supervisory role -- needs a disciplined model that responds cleanly.
    """
    return os.getenv("CRITIC_MODEL") or _get_family().critic or get_loop_observer_spec()


def get_embedding_spec() -> Optional[str]:
    """Get the embedding model spec for RAG.

    Resolution: EMBEDDING_MODEL env var > family embedding > None.
    No worker fallback -- embedding is a different capability.
    """
    return os.getenv("EMBEDDING_MODEL") or _get_family().embedding or None


def get_max_tokens() -> int:
    """Get the default output token budget.

    COMPASS_MAX_TOKENS env var, defaults to 2000.
    """
    return int(os.getenv("COMPASS_MAX_TOKENS", "2000"))


# --- Mutators ---

def set_model_spec(spec: str) -> None:
    """Switch the active model. Updates env var and invalidates cached provider."""
    os.environ["COMPASS_MODEL"] = spec
    from compass.llm.oracle import Oracle
    Oracle.invalidate()


def set_family(name: str) -> None:
    """Switch the active family. Updates both worker and oracle."""
    name_lower = name.lower()
    if name_lower not in FAMILIES:
        raise ValueError(f"Unknown family: {name}. Available: {', '.join(FAMILIES)}")
    os.environ["COMPASS_FAMILY"] = name_lower
    # Clear explicit overrides so family takes effect
    os.environ.pop("COMPASS_MODEL", None)
    os.environ.pop("ORACLE_MODEL", None)
    from compass.llm.oracle import Oracle
    Oracle.invalidate()
