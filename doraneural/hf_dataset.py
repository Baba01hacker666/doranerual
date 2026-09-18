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


USER_ROLES = {"user", "human", "prompter", "client", "q", "question", "input"}
ZEXO_ROLES = {"assistant", "bot", "gpt", "model", "chatgpt", "agent", "zexo", "a", "answer", "output", "system"}


def normalize_dialogue_text(text: str) -> str:
    """Normalize raw dialogue strings containing varying role tags to standard User/Zexo turns."""
    s = text.strip()
    if not s:
        return ""
    # 1. OpenAssistant / Guanaco: ### Human: ... ### Assistant: ...
    if "### Human:" in s or "### human:" in s or "### User:" in s:
        s = re.sub(r'###\s*(Human|User|Prompter):\s*', 'User: ', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*###\s*(Assistant|AI|Bot|Zexo|Model|GPT):\s*', '\nZexo: ', s, flags=re.IGNORECASE)
        return s.strip()
    # 2. ChatML raw tags: <|im_start|>user ... <|im_end|>
    if "<|im_start|>" in s:
        s = re.sub(r'<\|im_start\|>\s*system\s*(.*?)\s*<\|im_end\|>', '', s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r'<\|im_start\|>\s*(user|human|prompter)\s*', 'User: ', s, flags=re.IGNORECASE)
        s = re.sub(r'<\|im_start\|>\s*(assistant|gpt|model|bot|zexo)\s*', '\nZexo: ', s, flags=re.IGNORECASE)
        s = re.sub(r'<\|im_end\|>', '', s)
        return s.strip()
    # 3. Llama / Mistral [INST] [/INST]
    if "[INST]" in s:
        s = re.sub(r'\[INST\]\s*', 'User: ', s)
        s = re.sub(r'\s*\[/INST\]\s*', '\nZexo: ', s)
        return s.strip()
    # 4. Standard Human: / Assistant: or User: / Assistant: lines
    if re.search(r'^(Human|Assistant|AI):', s, flags=re.MULTILINE | re.IGNORECASE):
        s = re.sub(r'^(Human|Prompt|Question):\s*', 'User: ', s, flags=re.MULTILINE | re.IGNORECASE)
        s = re.sub(r'^(Assistant|Response|Answer|AI):\s*', 'Zexo: ', s, flags=re.MULTILINE | re.IGNORECASE)
        return s.strip()
    return s


def format_row_to_dialogue(row: Dict[str, any], fallback_col: Optional[str] = None) -> Optional[str]:
    """Format any Hugging Face dataset row into clean User/Zexo conversational text."""
    # 1. ChatML messages format: [{"role": "user", "content": "..."}, ...]
    if "messages" in row and isinstance(row["messages"], list):
        turns = []
        for m in row["messages"]:
            if isinstance(m, dict):
                r = str(m.get("role", "user")).lower()
                c = str(m.get("content", "")).strip()
                if not c or r in ("system",):
                    continue
                prefix = "Zexo" if r in ZEXO_ROLES else "User"
                turns.append(f"{prefix}: {c}")
        if turns:
            return "\n".join(turns)

    # 2. ShareGPT conversations format: [{"from": "human", "value": "..."}, ...]
    if "conversations" in row and isinstance(row["conversations"], list):
        turns = []
        for m in row["conversations"]:
            if isinstance(m, dict):
                r = str(m.get("from", m.get("role", "human"))).lower()
                c = str(m.get("value", m.get("content", ""))).strip()
                if not c or r in ("system",):
                    continue
                prefix = "Zexo" if r in ZEXO_ROLES else "User"
                turns.append(f"{prefix}: {c}")
        if turns:
            return "\n".join(turns)

    # 3. Instruction + Output/Response/Answer format (Alpaca, Dolly, etc.)
    if "instruction" in row:
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "") or row.get("context", "")).strip()
        out = str(row.get("output", "") or row.get("response", "") or row.get("answer", "")).strip()
        if inst and out:
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            return f"User: {user_msg}\nZexo: {out}"

    # 4. Prompt + Completion/Response format
    if "prompt" in row and ("completion" in row or "response" in row or "answer" in row):
        p = str(row.get("prompt", "")).strip()
        c = str(row.get("completion", "") or row.get("response", "") or row.get("answer", "")).strip()
        if p and c:
            return f"User: {p}\nZexo: {c}"

    # 5. Question + Answer format
    if "question" in row and ("answer" in row or "response" in row):
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "") or row.get("response", "")).strip()
        if q and a:
            return f"User: {q}\nZexo: {a}"

    # 6. Check common single text column names directly (text, content, dialogue, etc.)
    for candidate in ("text", "content", "dialogue", "conversation", "story"):
        if candidate in row and isinstance(row[candidate], str) and row[candidate].strip():
            normalized = normalize_dialogue_text(row[candidate])
            if normalized:
                return normalized

    # 7. Fallback to specified single text column
    if fallback_col and fallback_col in row:
        val = row[fallback_col]
        if isinstance(val, str) and val.strip():
            return normalize_dialogue_text(val)
        elif isinstance(val, (list, tuple)):
            joined = " ".join(str(item) for item in val if str(item).strip())
            if joined:
                return normalize_dialogue_text(joined)

    return None


def download_url_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch content from URL using curl if available, falling back to urllib."""
    import shutil
    import subprocess
    if shutil.which("curl"):
        try:
            cmd = ["curl", "-s", "-L", "--retry", "3", "--max-time", str(timeout), url]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception:
            pass

    req = urllib.request.Request(url, headers={"User-Agent": "doraneural/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_file_to_disk(url: str, out_path: Union[str, Path], timeout: int = 60) -> Path:
    """Download file from URL to disk using curl if available, falling back to urllib."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    import subprocess
    if shutil.which("curl"):
        try:
            # --fail makes curl treat HTTP 4xx/5xx as errors instead of writing
            # the error page body to disk (which would silently corrupt the
            # dataset/tokenizer); --show-error keeps the failure readable.
            cmd = ["curl", "-sS", "-f", "-L", "--retry", "3", "--max-time", str(timeout), "-o", str(p), url]
            proc = subprocess.run(cmd)
            if proc.returncode == 0 and p.exists() and p.stat().st_size > 0:
                return p
        except Exception:
            pass
        # Remove any partial/failed artifact so the urllib fallback re-downloads.
        if p.exists() and p.stat().st_size == 0:
            p.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "doraneural/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(p, "wb") as f:
        f.write(resp.read())
    return p


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
        return download_file_to_disk(raw_input, out_file, timeout=timeout)

    # 3. Hugging Face Hub dataset query
    clean_id = raw_input.replace("hf://", "").replace("datasets/", "").strip()
    sanitized_name = clean_id.replace("/", "_").replace("-", "_")

    if target_path:
        out_file = Path(target_path)
    else:
        base_dir = Path(cache_dir or (Path.home() / ".cache" / "doraneural" / "datasets"))
        base_dir.mkdir(parents=True, exist_ok=True)
        config_suffix = f"_{config}" if config else ""
        samples_suffix = "all" if max_samples <= 0 else str(max_samples)
        out_file = base_dir / f"{sanitized_name}{config_suffix}_{split}_{samples_suffix}.txt"

    if out_file.exists() and out_file.stat().st_size > 0 and not force_download:
        print(f"📦 Using cached Hugging Face dataset: {out_file} ({out_file.stat().st_size:,} bytes)")
        return out_file

    # Fast-path for datasets with direct hosted JSON/JSONL on Hugging Face Hub
    if clean_id in _HF_FAST_PATHS:
        try:
            fast_url = _HF_FAST_PATHS[clean_id]
            print(f"⚡ Fast-path: downloading dataset directly from {fast_url} using urllib...")
            collected = _download_fast_path(fast_url, max_samples=max_samples, timeout=timeout)
            if collected:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                full_corpus = "\n\n".join(collected)
                out_file.write_text(full_corpus, encoding="utf-8")
                print(f"✅ Extracted {len(collected):,} instruction dialogues to {out_file}!")
                print(f"   Corpus saved to: {out_file} ({len(full_corpus):,} chars, {len(full_corpus.split()):,} words)")
                return out_file
        except Exception as e:
            print(f"⚠️ Fast-path failed ({e}), falling back to datasets-server API...")

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
    target_samples = 10000 if max_samples <= 0 else max_samples

    encoded_id = urllib.parse.quote(clean_id, safe="/")
    offset = 0

    while len(collected_texts) < target_samples:
        current_limit = min(batch_size, target_samples - len(collected_texts))
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
            if len(collected_texts) >= target_samples:
                break
            r = r_entry.get("row", {})
            dialogue = format_row_to_dialogue(r, fallback_col=detected_col)
            if dialogue and dialogue.strip():
                collected_texts.append(dialogue.strip())

        offset += len(rows_data)
        progress_total = f"/{max_samples}" if max_samples > 0 else f"/{target_samples} (all)"
        print(f"\r  Fetched {len(collected_texts)}{progress_total} examples...", end="", flush=True)

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


# ── Fast-path registry ────────────────────────────────────────────────────────
# Maps dataset ID → (direct JSON URL, list of fields to form User/Zexo dialogue)
_HF_FAST_PATHS: dict = {
    "yahma/alpaca-cleaned": "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json",
    "tatsu-lab/alpaca":     "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/alpaca_data.json",
    "databricks/databricks-dolly-15k": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
    "timdettmers/openassistant-guanaco": "https://huggingface.co/datasets/timdettmers/openassistant-guanaco/resolve/main/openassistant_best_replies_train.jsonl",
}


def _download_fast_path(url: str, max_samples: int = -1, timeout: int = 90) -> list:
    """Download a known-format HF dataset file and return list of dialogue strings."""
    import urllib.request, json as _json

    req = urllib.request.Request(url, headers={"User-Agent": "doraneural/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    # JSONL (one JSON object per line) vs full JSON array
    if url.endswith(".jsonl"):
        rows = [_json.loads(l) for l in raw.splitlines() if l.strip()]
    else:
        rows = _json.loads(raw)

    limit = len(rows) if max_samples <= 0 else min(max_samples, len(rows))
    collected = []
    for item in rows[:limit]:
        d = format_row_to_dialogue(item)
        if d:
            collected.append(d)
    return collected


def download_and_merge_hf_datasets(
    dataset_ids: list,
    max_samples_each: int = -1,
    target_path=None,
    cache_dir=None,
    timeout: int = 90,
) -> "Path":
    """Download multiple HF datasets and merge them into a single training corpus.

    Args:
        dataset_ids: List of HuggingFace dataset identifiers, e.g.
                     ['yahma/alpaca-cleaned', 'databricks/databricks-dolly-15k']
        max_samples_each: Max samples per dataset (-1 = all).
        target_path: Where to write the merged corpus file.
        cache_dir: Cache base directory.
        timeout: Per-request network timeout.

    Returns:
        Path to merged corpus text file.
    """
    base_dir = Path(cache_dir or (Path.home() / ".cache" / "doraneural" / "datasets"))
    base_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(target_path) if target_path else base_dir / "merged_corpus.txt"

    all_dialogues = []
    for ds_id in dataset_ids:
        ds_id = ds_id.strip()
        if not ds_id:
            continue
        print(f"\n📥 Fetching dataset: {ds_id} ...")

        # Try known fast-path first
        fast_url = _HF_FAST_PATHS.get(ds_id)
        if fast_url:
            try:
                dialogues = _download_fast_path(fast_url, max_samples=max_samples_each, timeout=timeout)
                print(f"  ✅ Fast-path: {len(dialogues):,} examples")
                all_dialogues.extend(dialogues)
                continue
            except Exception as e:
                print(f"  ⚠️  Fast-path failed ({e}), falling back to datasets-server...")

        # Fallback: datasets-server API
        try:
            corpus_path = download_hf_dataset(
                ds_id,
                max_samples=max_samples_each,
                cache_dir=cache_dir,
                timeout=timeout,
            )
            text = corpus_path.read_text(encoding="utf-8", errors="replace")
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            print(f"  ✅ API fallback: {len(chunks):,} examples")
            all_dialogues.extend(chunks)
        except Exception as e:
            print(f"  ❌ Skipping {ds_id}: {e}")

    if not all_dialogues:
        raise ValueError("No examples collected from any dataset!")

    import random
    random.shuffle(all_dialogues)

    merged = "\n\n".join(all_dialogues)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(merged, encoding="utf-8")

    print(f"\n🗂️  Merged corpus: {len(all_dialogues):,} total examples → {out_file}")
    print(f"   Size: {len(merged):,} chars, {len(merged.split()):,} words")
    return out_file
