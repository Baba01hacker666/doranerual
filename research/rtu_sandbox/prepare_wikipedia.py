"""Simple Wikipedia Dataset Preparation Utility.

Supports:
1. Downloading compressed Simple English Wikipedia dump from Wikimedia.
2. Fast streaming extraction & wiki-markup cleaning (pure Python, zero dependencies).
3. Fallback to `wikiextractor` module if available.
4. Target size capping (e.g. 200 MB).
"""

import argparse
import bz2
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET


DEFAULT_DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
)


def clean_wiki_markup(raw_text: str) -> str:
    """Strips mediawiki templates, links, citations, and section formatting."""
    if not raw_text:
        return ""

    # Remove comments
    text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    # Remove ref tags and citations
    text = re.sub(r"<ref.*?(?:/>|</ref>)", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove MediaWiki templates {{ ... }}
    # Run twice to handle single nested templates
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Remove File / Image links [[File:...]]
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    # Normalize wikilinks [[Target|Display]] -> Display, [[Target]] -> Target
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Normalize headers == Section == -> Section.
    text = re.sub(r"=+\s*(.*?)\s*=+", r"\1.", text)
    # Clean external links [http://... title] -> title
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with progress reporting and resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "DoraneuralResearchLab/1.0 (https://github.com/Baba01hacker666/doranerual)"}

    # If partial file exists, check size
    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0

    req = urllib.request.Request(url, headers=headers)
    print(f"📥 Connecting to {url} ...")
    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        if existing_bytes >= total_size and total_size > 0:
            print(f"✓ Archive already fully downloaded ({existing_bytes / (1024*1024):.1f} MB).")
            return

        print(f"⬇️ Downloading ({total_size / (1024*1024):.1f} MB)...")
        downloaded = 0
        t0 = time.perf_counter()

        with open(dest_path, "wb") as f_out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                elapsed = max(1e-4, time.perf_counter() - t0)
                speed = downloaded / (1024 * 1024 * elapsed)
                pct = (downloaded / total_size * 100) if total_size > 0 else 0.0
                print(
                    f"\r  [{pct:5.1f}%] {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB "
                    f"({speed:.1f} MB/s)",
                    end="",
                    flush=True,
                )
    print("\n✓ Download completed.")


def extract_with_wikiextractor(archive_path: Path, output_file: Path, max_bytes: int) -> bool:
    """Attempts to run wikiextractor if available."""
    try:
        import wikiextractor  # noqa: F401
    except ImportError:
        return False

    temp_dir = archive_path.parent / "extracted_wiki_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print("⚡ Found `wikiextractor`. Running WikiExtractor tool...")
    cmd = [
        sys.executable,
        "-m",
        "wikiextractor.WikiExtractor",
        str(archive_path),
        "--no-templates",
        "-o",
        str(temp_dir),
        "-b",
        "100M",
    ]
    subprocess.run(cmd, check=True)

    print(f"🧹 Merging and filtering extracted articles up to {max_bytes / (1024*1024):.1f} MB...")
    written = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.startswith("wiki_"):
                    p = Path(root) / file
                    with open(p, "r", encoding="utf-8", errors="ignore") as in_f:
                        for line in in_f:
                            if line.startswith("<doc ") or line.startswith("</doc>"):
                                continue
                            out_f.write(line)
                            written += len(line.encode("utf-8"))
                            if written >= max_bytes:
                                break
                if written >= max_bytes:
                    break
            if written >= max_bytes:
                break

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    return True


def extract_with_streaming_parser(archive_path: Path, output_file: Path, max_bytes: int) -> int:
    """Extracts text using Python's standard bz2 + iterparse (zero external dependencies)."""
    print(f"🔄 Extracting via built-in streaming parser (target: {max_bytes / (1024*1024):.1f} MB)...")
    written = 0
    articles_count = 0
    t0 = time.perf_counter()

    with open(output_file, "w", encoding="utf-8") as out_f:
        with bz2.BZ2File(archive_path, "rb") as bz2_f:
            for _, elem in ET.iterparse(bz2_f, events=("end",)):
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "text" and elem.text:
                    clean_text = clean_wiki_markup(elem.text)
                    if len(clean_text) > 100 and not clean_text.lower().startswith("#redirect"):
                        chunk = clean_text + "\n\n"
                        out_f.write(chunk)
                        written += len(chunk.encode("utf-8"))
                        articles_count += 1

                        if articles_count % 500 == 0 or written >= max_bytes:
                            elapsed = max(1e-4, time.perf_counter() - t0)
                            print(
                                f"\r  Articles: {articles_count:,} | Extracted: {written / (1024*1024):.1f} MB / "
                                f"{max_bytes / (1024*1024):.1f} MB ({written / (1024*1024*elapsed):.2f} MB/s)",
                                end="",
                                flush=True,
                            )

                    elem.clear()

                if written >= max_bytes:
                    elem.clear()
                    break

    print(f"\n✓ Extracted {articles_count:,} articles ({written / (1024*1024):.1f} MB) to {output_file}")
    return written


def prepare_simplewiki(
    output_path: str = "research/rtu_sandbox/data/simplewiki_clean.txt",
    max_mb: float = 200.0,
    dump_url: str = DEFAULT_DUMP_URL,
    cache_dir: str = "research/rtu_sandbox/data/raw",
) -> Path:
    """Ensures Simple Wikipedia clean text file exists up to requested size."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = int(max_mb * 1024 * 1024)

    if out_p.exists() and out_p.stat().st_size >= target_bytes:
        print(f"✓ Wikipedia dataset already prepared at {out_p} ({out_p.stat().st_size / (1024*1024):.1f} MB).")
        return out_p

    archive_path = Path(cache_dir) / "simplewiki.xml.bz2"
    if not archive_path.exists():
        download_file(dump_url, archive_path)

    # Try wikiextractor first, fallback to pure python streaming
    extracted = extract_with_wikiextractor(archive_path, out_p, target_bytes)
    if not extracted:
        extract_with_streaming_parser(archive_path, out_p, target_bytes)

    return out_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & extract Simple Wikipedia corpus for RTU training")
    parser.add_argument("--output", default="research/rtu_sandbox/data/simplewiki_clean.txt", help="Output text file")
    parser.add_argument("--max-mb", type=float, default=200.0, help="Max MB of clean text to extract")
    parser.add_argument("--dump-url", default=DEFAULT_DUMP_URL, help="Wikimedia dump URL")
    args = parser.parse_args()

    prepare_simplewiki(output_path=args.output, max_mb=args.max_mb, dump_url=args.dump_url)
