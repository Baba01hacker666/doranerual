"""Zero-dependency Hugging Face Dataset Downloader & Text Extractor.

Fetches text datasets directly from the Hugging Face Hub using standard library
urllib and json APIs without requiring the heavy `datasets` or `pyarrow` packages.
Supports repository IDs (e.g., 'roneneldan/TinyStories'), direct URLs, automatic
config/split resolution, and intelligent text column detection.
"""

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# Common column names containing natural language text
CANDIDATE_TEXT_COLUMNS = [
    "text",
    "content",
    "story",
    "prompt",
    "sentence",
    "article",
    "document",
    "instruction",
    "output",
    "dialogue",
    "body",
    "description",
]


def is_hf_dataset_identifier(path_or_id: str) -> bool:
    """Check whether a string represents a Hugging Face dataset identifier or URL."""
    s = path_or_id.strip()
    if s.startswith("hf://") or "huggingface.co/datasets/" in s:
        return True
    if Path(s).exists():
        return False
    # Matches 'owner/dataset-name' or single-name datasets like 'wikitext', 'ag_news'
    if re.match(r"^[A-Za-z0-9_\-\.]+/[A-Za-z0-9_\-\.]+$", s):
        return True
    if re.match(r"^[A-Za-z0-9_\-]+$", s) and not s.endswith((".txt", ".csv", ".json", ".bin")):
        return True
    return False


def get_hf_dataset_splits(dataset_name: str, timeout: int = 15) -> List[Dict[str, str]]:
    """Query available configs and splits for a Hugging Face dataset."""
    clean_id = dataset_name.replace("hf://", "").strip()
    encoded_id = urllib.parse.quote(clean_id, safe="/")
    url = f"https://datasets-server.huggingface.co/splits?dataset={encoded_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "doraneural/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("splits", [])
    except Exception as e:
        return []


def _detect_text_column(sample_row: Dict[str, any], explicit_col: Optional[str] = None) -> str:
    """Detect the appropriate text column from a sample dictionary."""
    if explicit_col and explicit_col in sample_row:
        return explicit_col

    keys = list(sample_row.keys())
    # 1. Check exact match in candidate list
    for cand in CANDIDATE_TEXT_COLUMNS:
        for k in keys:
            if k.lower() == cand:
                return k

    # 2. Check substring match in candidate list
    for cand in CANDIDATE_TEXT_COLUMNS:
        for k in keys:
            if cand in k.lower() and isinstance(sample_row[k], str):
                return k

    # 3. Fallback to the string field with maximum length
    str_cols = [(k, len(str(v))) for k, v in sample_row.items() if isinstance(v, (str, list))]
    if str_cols:
        str_cols.sort(key=lambda x: x[1], reverse=True)
        return str_cols[0][0]

    raise ValueError(
        f"Unable to auto-detect text column. Available columns: {keys}. "
        f"Please specify --hf-column explicitly."
    )


def format_row_to_dialogue(row: Dict[str, any], fallback_col: Optional[str] = None) -> Optional[str]:
    """Format any Hugging Face dataset row into clean User/Zexo conversational text."""
    # 1. ChatML messages format: [{"role": "user", "content": "..."}, ...]
    if "messages" in row and isinstance(row["messages"], list):
        turns = []
        for m in row["messages"]:
            if isinstance(m, dict):
                r = m.get("role", "user").lower()
                c = str(m.get("content", "")).strip()
                prefix = "Zexo" if r in ("assistant", "bot", "gpt") else "User"
                if c:
                    turns.append(f"{prefix}: {c}")
        if turns:
            return "\n".join(turns)

    # 2. ShareGPT conversations format: [{"from": "human", "value": "..."}, ...]
    if "conversations" in row and isinstance(row["conversations"], list):
        turns = []
        for m in row["conversations"]:
            if isinstance(m, dict):
                r = m.get("from", "human").lower()
                c = str(m.get("value", "")).strip()
                prefix = "Zexo" if r in ("gpt", "assistant", "chatgpt") else "User"
                if c:
                    turns.append(f"{prefix}: {c}")
        if turns:
            return "\n".join(turns)

    # 3. Instruction + Output format (Alpaca, Dolly, etc.)
    if "instruction" in row and ("output" in row or "response" in row):
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "") or row.get("context", "")).strip()
        out = str(row.get("output", "") or row.get("response", "")).strip()
        if inst and out:
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            return f"User: {user_msg}\nZexo: {out}"

    # 4. Prompt + Completion format
    if "prompt" in row and ("completion" in row or "response" in row):
        p = str(row.get("prompt", "")).strip()
        c = str(row.get("completion", "") or row.get("response", "")).strip()
        if p and c:
            return f"User: {p}\nZexo: {c}"

    # 5. Question + Answer format
    if "question" in row and "answer" in row:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a:
            return f"User: {q}\nZexo: {a}"

    # 6. Fallback to single text column
    if fallback_col and fallback_col in row:
        val = row[fallback_col]
        if isinstance(val, str) and val.strip():
            return val.strip()
        elif isinstance(val, (list, tuple)):
            joined = " ".join(str(item) for item in val if str(item).strip())
            if joined:
                return joined

    return None


def download_hf_dataset(
    dataset_name_or_url: str,
    split: str = "train",
    config: Optional[str] = None,
    text_column: Optional[str] = None,
    max_samples: int = 100,
    target_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    timeout: int = 20,
) -> Path:
    """Download a dataset from Hugging Face Hub and compile it into a plain text corpus.

    Args:
        dataset_name_or_url: HuggingFace dataset ID (e.g. 'roneneldan/TinyStories')
            or direct file URL.
        split: Dataset split ('train', 'validation', 'test').
        config: Optional dataset configuration / subset name.
        text_column: Name of text field. If None, auto-detected from data.
        max_samples: Maximum number of rows/samples to download.
        target_path: Explicit destination path for the saved text corpus.
        cache_dir: Cache directory (default: ~/.cache/doraneural/datasets).
        force_download: If True, bypass cached file.
        timeout: Network timeout in seconds.

    Returns:
        Path to the local plain text file ready for training.
    """
    raw_input = dataset_name_or_url.strip()

    # 1. If it's already an existing local file or directory, return immediately
    if Path(raw_input).exists():
        return Path(raw_input)

    # 2. Handle direct HTTP/HTTPS file downloads (e.g., .txt, .jsonl, .csv)
    if raw_input.startswith(("http://", "https://")) and not "datasets-server" in raw_input:
        if target_path:
            out_file = Path(target_path)
        else:
            base_dir = Path(cache_dir or (Path.home() / ".cache" / "doraneural" / "datasets"))
            base_dir.mkdir(parents=True, exist_ok=True)
            filename = raw_input.split("/")[-1].split("?")[0] or "dataset.txt"
            out_file = base_dir / filename

        if out_file.exists() and out_file.stat().st_size > 0 and not force_download:
            return out_file

        print(f"📥 Downloading direct dataset from URL: {raw_input}")
        req = urllib.request.Request(raw_input, headers={"User-Agent": "doraneural/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(out_file, "wb") as f:
            f.write(resp.read())
        return out_file

    # 3. Hugging Face Hub dataset query
    clean_id = raw_input.replace("hf://", "").replace("datasets/", "").strip()
    sanitized_name = clean_id.replace("/", "_").replace("-", "_")

    if target_path:
        out_file = Path(target_path)
    else:
        base_dir = Path(cache_dir or (Path.home() / ".cache" / "doraneural" / "datasets"))
        base_dir.mkdir(parents=True, exist_ok=True)
        out_file = base_dir / f"{sanitized_name}_{split}_{max_samples}.txt"

    if out_file.exists() and out_file.stat().st_size > 0 and not force_download:
        print(f"📦 Using cached Hugging Face dataset: {out_file} ({out_file.stat().st_size:,} bytes)")
        return out_file

    print(f"🌐 Querying Hugging Face Dataset Server: '{clean_id}' (split={split})...")

    # Resolve config if not provided
    resolved_config = config
    if not resolved_config:
        splits_info = get_hf_dataset_splits(clean_id, timeout=timeout)
        if splits_info:
            # Look for matching split
            matching = [s for s in splits_info if s.get("split") == split]
            if matching:
                resolved_config = matching[0].get("config", "default")
            else:
                resolved_config = splits_info[0].get("config", "default")
                split = splits_info[0].get("split", "train")
        else:
            resolved_config = "default"

    # Paginate and collect rows
    batch_size = 100
    collected_texts: List[str] = []
    detected_col = text_column

    encoded_id = urllib.parse.quote(clean_id, safe="/")
    offset = 0

    while len(collected_texts) < max_samples:
        current_limit = min(batch_size, max_samples - len(collected_texts))
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset={encoded_id}&config={resolved_config}&split={split}"
            f"&offset={offset}&limit={current_limit}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "doraneural/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Failed to fetch dataset '{clean_id}' from Hugging Face (HTTP {e.code}): {err_msg}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error connecting to Hugging Face datasets server: {e}") from e

        rows_data = data.get("rows", [])
        if not rows_data:
            break

        # Detect column on first batch
        if detected_col is None and rows_data:
            first_row = rows_data[0].get("row", {})
            detected_col = _detect_text_column(first_row, explicit_col=text_column)
            print(f"  -> Auto-detected text column: '{detected_col}'")

        for r_entry in rows_data:
            if len(collected_texts) >= max_samples:
                break
            r = r_entry.get("row", {})
            dialogue = format_row_to_dialogue(r, fallback_col=detected_col)
            if dialogue and dialogue.strip():
                collected_texts.append(dialogue.strip())

        offset += len(rows_data)
        print(f"\r  Fetched {len(collected_texts)}/{max_samples} examples...", end="", flush=True)

        # If fewer rows returned than requested, we reached the end of the split
        if len(rows_data) < current_limit:
            break

    print("")

    if not collected_texts:
        raise ValueError(
            f"No valid text samples found in dataset '{clean_id}' under column '{detected_col}'."
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    full_corpus = "\n\n".join(collected_texts)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    print(f"✅ Downloaded {len(collected_texts):,} examples from Hugging Face!")
    print(f"   Corpus saved to: {out_file} ({len(full_corpus):,} chars, {len(full_corpus.split()):,} words)")
    return out_file
