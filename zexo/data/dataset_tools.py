"""Build and validate high-quality Zexo conversation datasets.

The training corpus is intentionally kept as data, not embedded in the trainer.
This module accepts JSONL records with explicit roles, validates them, removes
exact duplicates, renders the same ``System/User/Zexo`` format used by
``ChatSession``, and creates a deterministic held-out evaluation split.

Usage from the repository root::

    python -m zexo.data.dataset_tools

The builder has no dependencies beyond Python's standard library, so it can run
on the same low-power machines used for Zexo training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = DATA_DIR / "quality_dialogues.jsonl"
DEFAULT_TRAIN = DATA_DIR / "zexo_quality_v1_train.txt"
DEFAULT_EVAL = DATA_DIR / "zexo_quality_v1_eval.txt"
DEFAULT_MANIFEST = DATA_DIR / "zexo_quality_v1_manifest.json"

SYSTEM_PROMPT = (
    "You are Zexo, an intelligent, thoughtful, and creative conversational AI assistant. "
    "You communicate with clarity, warmth, and precision. You love helping users explore ideas, "
    "solve challenging problems, explain complex topics simply, and hold engaging, meaningful conversations."
)

_ALLOWED_ROLES = ("user", "assistant")
_ROLE_PREFIX = {"user": "User", "assistant": "Zexo"}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if _CONTROL_CHARS.search(text):
        raise ValueError(f"{field} contains a control character")
    return text


def validate_record(record: object, line_number: int = 0) -> Dict[str, object]:
    """Validate and normalize one JSONL conversation record."""
    if not isinstance(record, dict):
        raise ValueError(f"record {line_number} must be a JSON object")

    raw_id = record.get("id")
    record_id = _clean_text(raw_id, f"record {line_number} id") if raw_id else f"line-{line_number}"
    category = _clean_text(record.get("category", "general"), f"record {record_id} category").lower()
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError(f"record {record_id} messages must contain an even user/assistant sequence")

    normalized: List[Dict[str, str]] = []
    expected = "user"
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"record {record_id} message {idx} must be an object")
        role = str(message.get("role", "")).strip().lower()
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"record {record_id} message {idx} has unsupported role {role!r}")
        if role != expected:
            raise ValueError(f"record {record_id} must alternate user and assistant messages")
        normalized.append({"role": role, "content": _clean_text(message.get("content"), f"record {record_id} message {idx}")})
        expected = "assistant" if expected == "user" else "user"

    return {"id": record_id, "category": category, "messages": normalized}


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    """Read, validate, and deduplicate JSONL conversations."""
    records: List[Dict[str, object]] = []
    seen_ids = set()
    seen_content = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            try:
                record = validate_record(json.loads(raw_line), line_number)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            record_id = str(record["id"])
            if record_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            canonical = json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if digest in seen_content:
                continue
            seen_ids.add(record_id)
            seen_content.add(digest)
            records.append(record)
    if not records:
        raise ValueError(f"No valid conversations found in {path}")
    return records


def render_record(record: Dict[str, object], system_prompt: str = SYSTEM_PROMPT) -> str:
    """Render a validated record in the format consumed by ChatSession."""
    lines = [f"System: {system_prompt.strip()}"]
    for message in record["messages"]:  # type: ignore[index]
        role = message["role"]  # type: ignore[index]
        lines.append(f"{_ROLE_PREFIX[role]}: {message['content']}")  # type: ignore[index]
    # An explicit boundary prevents the last answer in one example from becoming
    # the first continuation target of the next example.
    lines.append("<|endoftext|>")
    return "\n".join(lines)


def _split_by_category(records: Sequence[Dict[str, object]], eval_ratio: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Create a deterministic, category-balanced held-out split."""
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")
    by_category: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        by_category.setdefault(str(record["category"]), []).append(record)

    train: List[Dict[str, object]] = []
    evaluation: List[Dict[str, object]] = []
    for category in sorted(by_category):
        group = sorted(by_category[category], key=lambda item: str(item["id"]))
        n_eval = int(round(len(group) * eval_ratio))
        if eval_ratio > 0 and len(group) > 1:
            n_eval = max(1, n_eval)
            n_eval = min(n_eval, len(group) - 1)
        evaluation.extend(group[:n_eval])
        train.extend(group[n_eval:])
    return train, evaluation


def write_split(records: Iterable[Dict[str, object]], path: Path, supplemental_text: str = "") -> int:
    """Write rendered conversations and optional vetted legacy text."""
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [render_record(record) for record in records]
    if supplemental_text.strip():
        blocks.append(supplemental_text.strip())
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return len(records)


def load_corpus(path: Path) -> str:
    """Load a plain-text corpus or render a validated conversation JSONL file."""
    if path.suffix.lower() == ".jsonl":
        return "\n\n".join(render_record(record) for record in read_jsonl(path)) + "\n"
    return path.read_text(encoding="utf-8", errors="replace")


def _display_path(path: Path) -> str:
    """Return a stable repository-relative path when possible (Python 3.8-safe)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def build_dataset(source: Path = DEFAULT_SOURCE, train_path: Path = DEFAULT_TRAIN, eval_path: Path = DEFAULT_EVAL, manifest_path: Path = DEFAULT_MANIFEST, eval_ratio: float = 0.15, include_legacy: bool = True) -> Dict[str, object]:
    """Build train/eval text files and a reproducibility manifest.

    The curated JSONL records are split by category. The existing technical
    corpus is added to *training only* as supplemental knowledge, so no legacy
    example can accidentally leak into the held-out score.
    """
    records = read_jsonl(source)
    train, evaluation = _split_by_category(records, eval_ratio)
    legacy_path = DATA_DIR / "step1_conversational_expanded.txt"
    legacy_text = legacy_path.read_text(encoding="utf-8", errors="replace") if include_legacy and legacy_path.exists() else ""
    supplemental = ""
    if legacy_text.strip():
        supplemental = f"System: {SYSTEM_PROMPT}\n{legacy_text.strip()}\n<|endoftext|>"
    train_count = write_split(train, train_path, supplemental_text=supplemental)
    eval_count = write_split(evaluation, eval_path)
    categories = Counter(str(record["category"]) for record in records)
    manifest: Dict[str, object] = {
        "format_version": "1",
        "source": _display_path(source),
        "train_file": _display_path(train_path),
        "eval_file": _display_path(eval_path),
        "records": len(records),
        "train_records": train_count,
        "eval_records": eval_count,
        "eval_ratio": eval_ratio,
        "categories": dict(sorted(categories.items())),
        "supplemental_legacy_file": str(legacy_path.relative_to(REPO_ROOT)) if legacy_text else None,
        "supplemental_legacy_words": len(legacy_text.split()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "legacy_sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest() if legacy_text else None,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Zexo train/eval text files from quality JSONL.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", dest="eval_path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    args = parser.parse_args()
    manifest = build_dataset(args.source, args.train, args.eval_path, args.manifest, args.eval_ratio)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
