"""Regression tests for the versioned Zexo quality-data pipeline."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zexo.data.dataset_tools import build_dataset, load_corpus, read_jsonl


class TestZexoQualityDataset(unittest.TestCase):
    def test_source_is_valid_and_balanced(self):
        source = Path(__file__).resolve().parents[1] / "zexo" / "data" / "quality_dialogues.jsonl"
        records = read_jsonl(source)
        self.assertGreaterEqual(len(records), 30)
        self.assertEqual(len({record["id"] for record in records}), len(records))
        categories = {record["category"] for record in records}
        self.assertGreaterEqual(categories, {"behavior", "programming", "safety", "zexo"})
        for record in records:
            messages = record["messages"]
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[-1]["role"], "assistant")

    def test_builder_creates_disjoint_deterministic_splits(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "zexo" / "data" / "quality_dialogues.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            train_path = out / "train.txt"
            eval_path = out / "eval.txt"
            manifest_path = out / "manifest.json"
            manifest = build_dataset(
                source=source,
                train_path=train_path,
                eval_path=eval_path,
                manifest_path=manifest_path,
                eval_ratio=0.2,
                include_legacy=False,
            )
            self.assertEqual(manifest["records"], manifest["train_records"] + manifest["eval_records"])
            train = train_path.read_text(encoding="utf-8")
            evaluation = eval_path.read_text(encoding="utf-8")
            self.assertIn("<|endoftext|>", train)
            self.assertIn("<|endoftext|>", evaluation)
            self.assertNotEqual(train, evaluation)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["source_sha256"], manifest["source_sha256"])

    def test_jsonl_can_be_loaded_directly(self):
        source = Path(__file__).resolve().parents[1] / "zexo" / "data" / "quality_dialogues.jsonl"
        rendered = load_corpus(source)
        self.assertIn("System: You are Zexo", rendered)
        self.assertIn("User:", rendered)
        self.assertIn("Zexo:", rendered)


if __name__ == "__main__":
    unittest.main()
