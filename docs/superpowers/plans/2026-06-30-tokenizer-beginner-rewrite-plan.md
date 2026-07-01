# Tokenizer Beginner Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Project 01 so the tokenizer mechanics are beginner-friendly, the blog post becomes the primary teaching path, and artifact generation is clearly separated from the tokenizer core.

**Architecture:** Split the current monolithic `main.py` into a small tokenizer core and a separate artifact/export layer, then rewrite the project README, notebook, and portfolio blog to follow the same toy-example-first teaching path. Use `unittest` smoke coverage for the new Python boundaries and existing run/build commands for cross-repo verification.

**Tech Stack:** Python 3, `unittest`, JSON, static HTML/CSS/JS widget assets, Hugo Markdown content

---

## File Structure

- Create: `projects/01-tokenizer-from-scratch/tokenizer.py`
- Create: `projects/01-tokenizer-from-scratch/artifacts.py`
- Create: `projects/01-tokenizer-from-scratch/tests/test_tokenizer.py`
- Create: `projects/01-tokenizer-from-scratch/tests/test_artifacts.py`
- Modify: `projects/01-tokenizer-from-scratch/main.py`
- Modify: `projects/01-tokenizer-from-scratch/README.md`
- Modify: `projects/01-tokenizer-from-scratch/lab.ipynb`
- Modify: `/home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch/content/posts/llm-engineering-from-scratch-tokenizer/index.md`

### Task 1: Extract The Beginner-Facing Tokenizer Core

**Files:**
- Create: `projects/01-tokenizer-from-scratch/tokenizer.py`
- Create: `projects/01-tokenizer-from-scratch/tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing tokenizer-core tests**

```python
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import BPETokenizer, bytes_to_display


class TokenizerCoreTests(unittest.TestCase):
    def test_train_encode_decode_round_trip(self) -> None:
        tokenizer = BPETokenizer().train(
            ["low lower lowest", "low low lower"],
            vocab_size=260,
        )

        token_ids = tokenizer.encode("low lower")

        self.assertEqual(tokenizer.decode(token_ids), "low lower")
        self.assertGreater(len(tokenizer.merges), 0)

    def test_byte_display_keeps_whitespace_and_hex_fallback_clear(self) -> None:
        self.assertEqual(bytes_to_display(b" "), "␠")
        self.assertEqual(bytes_to_display(b"\n"), "\\n")
        self.assertEqual(bytes_to_display(bytes([0xF0])), "0xf0")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest projects/01-tokenizer-from-scratch/tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer'`

- [ ] **Step 3: Write the minimal tokenizer implementation**

```python
from __future__ import annotations

from collections import Counter


def bytes_to_display(piece: bytes) -> str:
    if piece == b" ":
        return "␠"
    if piece == b"\n":
        return "\\n"

    decoded = piece.decode("utf-8", errors="replace")
    if "\ufffd" in decoded:
        return "0x" + piece.hex()
    return decoded.replace(" ", "␠").replace("\n", "\\n")


def piece_to_json(piece: bytes) -> dict[str, object]:
    return {"bytes": list(piece), "text": bytes_to_display(piece)}


class BPETokenizer:
    """Small byte-level BPE tokenizer used by the beginner walkthrough."""

    def __init__(self) -> None:
        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.id_to_token = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []
        self.trace: list[dict[str, object]] = []

    def _initial_symbols(self, text: str) -> list[bytes]:
        return [bytes([byte]) for byte in text.encode("utf-8")]

    def _count_pairs(self, sequences: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
        counts: Counter[tuple[bytes, bytes]] = Counter()
        for sequence in sequences:
            counts.update(zip(sequence, sequence[1:]))
        return counts

    def _merge_sequence(
        self, sequence: list[bytes], pair: tuple[bytes, bytes]
    ) -> list[bytes]:
        merged: list[bytes] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(sequence[index] + sequence[index + 1])
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def _sequence_metrics(
        self, sequences: list[list[bytes]], initial_total_tokens: int
    ) -> dict[str, float | int]:
        total_tokens = sum(len(sequence) for sequence in sequences)
        return {
            "vocab_size": len(self.token_to_id),
            "learned_merges": len(self.merges),
            "total_tokens": total_tokens,
            "compression_ratio": round(total_tokens / initial_total_tokens, 4),
        }

    def train(
        self, corpus: list[str], vocab_size: int, *, capture_trace: bool = False
    ) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("byte-level BPE needs vocab_size >= 256")

        sequences = [self._initial_symbols(text) for text in corpus]
        initial_total_tokens = sum(len(sequence) for sequence in sequences)

        while len(self.token_to_id) < vocab_size:
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break

            best_pair = None
            for pair, _count in pair_counts.most_common():
                merged_token = pair[0] + pair[1]
                if merged_token not in self.token_to_id:
                    best_pair = pair
                    break
            if best_pair is None:
                break

            before = sequences
            merged_token = best_pair[0] + best_pair[1]
            token_id = len(self.token_to_id)
            self.token_to_id[merged_token] = token_id
            self.id_to_token[token_id] = merged_token
            self.merges.append(best_pair)
            sequences = [self._merge_sequence(sequence, best_pair) for sequence in sequences]

            if capture_trace:
                self.trace.append(
                    {
                        "step": len(self.merges),
                        "selected_pair": [
                            piece_to_json(best_pair[0]),
                            piece_to_json(best_pair[1]),
                        ],
                        "merged_token": piece_to_json(merged_token),
                        "frequency": pair_counts[best_pair],
                        "top_pairs": [
                            {
                                "left": piece_to_json(left),
                                "right": piece_to_json(right),
                                "count": count,
                            }
                            for (left, right), count in pair_counts.most_common(8)
                        ],
                        "before_sequences": [
                            [piece_to_json(piece) for piece in sequence]
                            for sequence in before
                        ],
                        "after_sequences": [
                            [piece_to_json(piece) for piece in sequence]
                            for sequence in sequences
                        ],
                        "metrics": self._sequence_metrics(sequences, initial_total_tokens),
                    }
                )

        return self

    def encode(self, text: str) -> list[int]:
        sequence = self._initial_symbols(text)
        for pair in self.merges:
            sequence = self._merge_sequence(sequence, pair)
        return [self.token_to_id[symbol] for symbol in sequence]

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token id: {token_id}")
            pieces.append(self.id_to_token[token_id])
        return b"".join(pieces).decode("utf-8")

    def token_pieces(self, token_ids: list[int]) -> list[dict[str, object]]:
        return [piece_to_json(self.id_to_token[token_id]) for token_id in token_ids]

    def merge_records(self) -> list[dict[str, object]]:
        records = []
        for index, (left, right) in enumerate(self.merges, start=256):
            records.append(
                {
                    "id": index,
                    "left": piece_to_json(left),
                    "right": piece_to_json(right),
                    "token": piece_to_json(left + right),
                }
            )
        return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest projects/01-tokenizer-from-scratch/tests/test_tokenizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/01-tokenizer-from-scratch/tokenizer.py projects/01-tokenizer-from-scratch/tests/test_tokenizer.py
git commit -m "Extract beginner tokenizer core"
```

### Task 2: Separate Artifact Generation From The Tokenizer Core

**Files:**
- Create: `projects/01-tokenizer-from-scratch/artifacts.py`
- Create: `projects/01-tokenizer-from-scratch/tests/test_artifacts.py`
- Modify: `projects/01-tokenizer-from-scratch/main.py`
- Test: `projects/01-tokenizer-from-scratch/tests/test_artifacts.py`

- [ ] **Step 1: Write the failing artifact-boundary tests**

```python
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifacts import build_artifacts


class ArtifactTests(unittest.TestCase):
    def test_build_artifacts_writes_trace_metrics_and_widget_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            widget_dir = root / "widget"
            widget_dir.mkdir()

            result = build_artifacts(
                artifacts_dir=root / "artifacts",
                widget_data_path=widget_dir / "data.json",
            )

            self.assertIn("trace", result)
            self.assertIn("metrics", result)
            self.assertTrue((root / "artifacts" / "trace.json").exists())
            self.assertTrue((root / "artifacts" / "metrics.json").exists())
            self.assertTrue((widget_dir / "data.json").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest projects/01-tokenizer-from-scratch/tests/test_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'artifacts'`

- [ ] **Step 3: Move artifact logic into `artifacts.py` and shrink `main.py`**

```python
# artifacts.py
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import struct
import zlib

from tokenizer import BPETokenizer

DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
DEFAULT_WIDGET_DATA_PATH = Path(__file__).with_name("widget") / "data.json"

TINY_CORPUS = [
    "tokenization turns text into model readable symbols.",
    "models do not see words directly; they see token ids.",
    "byte pair encoding learns common byte patterns.",
    "rare terms like hypermicrotokenization should not become unknown.",
    "python code keeps spaces and indentation meaningful.",
    "math uses symbols like x^2 + y^2 = z^2.",
    "cafe café coffee tokenizer tokenizer tokenizer.",
    "the same phrase the same phrase the same phrase repeats.",
]

TRACE_CORPUS = [
    "low lower lowest",
    "low low lower",
    "token token tokenization",
]

STRESS_EXAMPLES = {
    "common": "tokenization turns text into token ids.",
    "rare": "hypermicrotokenization antidisestablishmentarian-ish words",
    "code": "def tokenize(text):\\n    return text.encode('utf-8')\\n",
    "math": "f(x) = x^2 + 2*x + 1; sum_i p_i log(p_i)",
    "emoji": "I like clean tokenizers ✨ café 👋🏽",
    "multilingual": "Hello नमस्ते こんにちは مرحبا Привет",
}

def summarize_tokenizer(
    tokenizer: BPETokenizer, examples: dict[str, str]
) -> list[dict[str, object]]:
    rows = []
    for name, text in examples.items():
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        assert decoded == text
        rows.append(
            {
                "category": name,
                "characters": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "tokens": len(token_ids),
                "tokens_per_char": round(len(token_ids) / max(1, len(text)), 4),
                "pieces": tokenizer.token_pieces(token_ids),
            }
        )
    return rows

# Copy the remaining helper functions from the current main.py into this module
# without changing their behavior:
# write_json, png_chunk, write_png, make_canvas, draw_rect, draw_bar_chart,
# draw_histogram, write_preview, pack_gif_codes, lzw_encode_indices,
# gif_subblocks, write_preview_gif, and write_failure_gallery.

def build_artifacts(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    widget_data_path: Path = DEFAULT_WIDGET_DATA_PATH,
) -> dict[str, object]:
    artifacts_dir.mkdir(exist_ok=True)

    trace_tokenizer = BPETokenizer().train(TRACE_CORPUS, vocab_size=272, capture_trace=True)
    small = BPETokenizer().train(TINY_CORPUS, vocab_size=265)
    large = BPETokenizer().train(TINY_CORPUS, vocab_size=305)

    small_rows = summarize_tokenizer(small, STRESS_EXAMPLES)
    large_rows = summarize_tokenizer(large, STRESS_EXAMPLES)

    trace_data = {
        "metadata": {
            "project": "01-tokenizer-from-scratch",
            "title": "BPE Merge Microscope",
            "corpus": TRACE_CORPUS,
            "initial_vocab_size": 256,
            "final_vocab_size": len(trace_tokenizer.token_to_id),
        },
        "steps": trace_tokenizer.trace,
        "merges": trace_tokenizer.merge_records(),
        "examples": [{"label": key, "text": value} for key, value in STRESS_EXAMPLES.items()],
    }
    metrics = {
        "small_vocab": {"vocab_size": len(small.token_to_id), "rows": small_rows},
        "large_vocab": {"vocab_size": len(large.token_to_id), "rows": large_rows},
    }

    write_json(artifacts_dir / "trace.json", trace_data)
    write_json(artifacts_dir / "metrics.json", metrics)
    write_json(widget_data_path, trace_data)
    write_failure_gallery(artifacts_dir / "failure_gallery.md", large_rows)
    draw_bar_chart(
        artifacts_dir / "compression_ratio.png",
        [float(row["tokens_per_char"]) for row in large_rows],
        [(23, 105, 170), (11, 107, 87), (138, 90, 0)],
    )
    token_lengths = [len(bytes(piece["bytes"])) for row in large_rows for piece in row["pieces"]]
    draw_histogram(artifacts_dir / "token_length_distribution.png", token_lengths)
    write_preview(artifacts_dir / "preview.png")
    write_preview_gif(artifacts_dir / "preview.gif")
    return {"trace": trace_data, "metrics": metrics}
```

```python
# main.py
from __future__ import annotations

from artifacts import DEFAULT_ARTIFACTS_DIR, TINY_CORPUS, build_artifacts
from tokenizer import BPETokenizer


def print_demo() -> None:
    tokenizer = BPETokenizer().train(TINY_CORPUS, vocab_size=286)
    demo_texts = {
        "seen-ish": "tokenization turns text into token ids.",
        "rare-word": "hypermicrotokenization should stay reversible.",
        "code": "def tokenize(text):\n    return text.encode('utf-8')\n",
        "unicode": "café नमस्ते ✨",
    }

    for name, text in demo_texts.items():
        token_ids = tokenizer.encode(text)
        pieces = [piece["text"] for piece in tokenizer.token_pieces(token_ids)]
        decoded = tokenizer.decode(token_ids)
        assert decoded == text
        print(f"\n{name}")
        print(f"text:      {text!r}")
        print(f"token ids: {token_ids}")
        print(f"pieces:    {pieces}")
        print(f"decoded:   {decoded!r}")

    print(f"\nVocabulary size: {len(tokenizer.token_to_id)}")
    print(f"Learned merges:  {len(tokenizer.merges)}")


def main() -> None:
    print_demo()
    build_artifacts()
    print(f"\nArtifacts written to: {DEFAULT_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and the project entry point**

Run: `python3 -m unittest projects/01-tokenizer-from-scratch/tests/test_tokenizer.py projects/01-tokenizer-from-scratch/tests/test_artifacts.py -v`
Expected: PASS

Run: `python3 projects/01-tokenizer-from-scratch/main.py`
Expected: PASS and print demo output followed by `Artifacts written to:`

- [ ] **Step 5: Commit**

```bash
git add projects/01-tokenizer-from-scratch/artifacts.py projects/01-tokenizer-from-scratch/main.py projects/01-tokenizer-from-scratch/tests/test_artifacts.py
git commit -m "Separate tokenizer artifacts from core logic"
```

### Task 3: Rewrite The Project README And Simplify The Notebook

**Files:**
- Modify: `projects/01-tokenizer-from-scratch/README.md`
- Modify: `projects/01-tokenizer-from-scratch/lab.ipynb`

- [ ] **Step 1: Update the README to point beginners to the blog and explain artifacts plainly**

```markdown
# 01. Tokenizer From Scratch

This project builds a small byte-level BPE tokenizer from scratch and pairs it with a beginner-friendly blog walkthrough.

## Start Here

Begin with the blog post:

- [Tokenizer From Scratch: BPE as Learned Compression](https://avisheksaha.com/posts/llm-engineering-from-scratch-tokenizer/)

Then run the project locally:

```bash
python main.py
```

## What This Project Contains

- `tokenizer.py`: the beginner-facing tokenizer core
- `main.py`: a small demo runner
- `artifacts.py`: optional export logic for charts, JSON, previews, and widget data
- `lab.ipynb`: a lightweight sandbox

## Artifacts

- `trace.json`: merge-by-merge training history used by the widget
- `metrics.json`: summary measurements for selected example texts
- `compression_ratio.png`: token-count comparison across example inputs
- `token_length_distribution.png`: distribution of learned token-piece lengths
- `failure_gallery.md`: stress-case summaries with token previews
- `widget/`: static BPE Merge Microscope demo source

## Further Study

- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Companion Colab notebook](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
```

- [ ] **Step 2: Trim the notebook into a lightweight companion**

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Tokenizer From Scratch Lab\n",
        "\n",
        "This notebook is an optional companion to the blog post. Use it to try a few encode/decode examples and optionally regenerate the exported artifacts."
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Quick Demo\n",
        "\n",
        "Train the tokenizer, encode a short string, and decode it back."
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from artifacts import TINY_CORPUS, build_artifacts\n",
        "from tokenizer import BPETokenizer\n",
        "\n",
        "tokenizer = BPETokenizer().train(TINY_CORPUS, vocab_size=286)\n",
        "text = \"tokenization turns text into token ids.\"\n",
        "token_ids = tokenizer.encode(text)\n",
        "\n",
        "print(token_ids)\n",
        "print([piece[\"text\"] for piece in tokenizer.token_pieces(token_ids)])\n",
        "print(tokenizer.decode(token_ids))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Optional Artifact Export\n",
        "\n",
        "Run this only if you want to regenerate the JSON files, charts, previews, and widget data."
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "artifact_data = build_artifacts()\n",
        "artifact_data[\"metrics\"].keys()"
      ]
    }
  ]
}
```

- [ ] **Step 3: Verify the README and notebook changes**

Run: `python3 -m json.tool projects/01-tokenizer-from-scratch/lab.ipynb >/dev/null`
Expected: PASS

Run: `python3 projects/01-tokenizer-from-scratch/main.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add projects/01-tokenizer-from-scratch/README.md projects/01-tokenizer-from-scratch/lab.ipynb
git commit -m "Rewrite beginner project docs"
```

### Task 4: Rewrite The Portfolio Blog Into A Layered Beginner Lesson

**Files:**
- Modify: `/home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch/content/posts/llm-engineering-from-scratch-tokenizer/index.md`

- [ ] **Step 1: Rewrite the post structure around the approved teaching arc**

```markdown
## Why Tokenization Exists

Language models do not read raw text directly. They read token ids. A tokenizer is the translation layer that turns text into those ids, and the choice of tokenizer affects context length, rare-word handling, code formatting, multilingual text, and even latency.

## Toy BPE First

Start with a short example such as `low lower lowest`.

Round 1:
- split the text into tiny starting pieces
- count adjacent pairs
- merge the most frequent pair

Round 2:
- count the new adjacent pairs
- merge the next most frequent pair

The point of the toy example is not realism. The point is to let the reader watch BPE create larger reusable pieces from repetition.

## From Toy BPE To Byte-Level BPE

The real project starts from UTF-8 bytes instead of character fragments. That one change makes the tokenizer robust: every string can be represented, even when it contains accented text, emoji, or multiple writing systems.

## A Minimal Tokenizer

Introduce only four small ideas from `tokenizer.py`:
- turn text into bytes
- count adjacent pairs
- merge the best pair
- replay the learned merges to encode new text

## What Changes On Real Inputs

Use short plain-language notes for:
- common text compresses more easily
- rare text often falls back to smaller pieces
- code keeps punctuation and spacing
- emoji and multilingual text remain reversible even when the pieces look awkward

## Explore Further

Introduce the extras only after the core lesson:
- `trace.json`: merge-by-merge history for the widget
- `metrics.json`: counts used for the comparison chart
- chart: how the same examples change under different vocabulary sizes
- widget: an optional merge replay tool

## Further Resources

- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Companion Colab notebook](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
```

- [ ] **Step 2: Clarify the chart explanations and artifact framing**

```markdown
For the compression chart, explain the comparison directly:
"This chart compares the same example inputs under two tokenizer vocabulary sizes. Fewer tokens means the tokenizer has learned larger reusable pieces for that kind of text."

For `metrics.json`, explain:
"This file stores the per-example counts used to build the chart and inspect how token counts change by input type."

For `trace.json`, explain:
"This file stores the merge-by-merge training history used by the interactive widget."
```

- [ ] **Step 3: Verify the Hugo post still builds**

Run: `hugo --gc --minify`
Working directory: `/home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git -C /home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch add content/posts/llm-engineering-from-scratch-tokenizer/index.md
git -C /home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch commit -m "Rewrite tokenizer post for beginners"
```

### Task 5: Final Integration Verification

**Files:**
- Modify: none
- Test: `projects/01-tokenizer-from-scratch/tests/test_tokenizer.py`
- Test: `projects/01-tokenizer-from-scratch/tests/test_artifacts.py`

- [ ] **Step 1: Run Python verification**

Run: `python3 -m unittest projects/01-tokenizer-from-scratch/tests/test_tokenizer.py projects/01-tokenizer-from-scratch/tests/test_artifacts.py -v`
Expected: PASS

- [ ] **Step 2: Run the tokenizer demo and artifact export**

Run: `python3 projects/01-tokenizer-from-scratch/main.py`
Expected: PASS and regenerate artifacts without errors

- [ ] **Step 3: Run the portfolio build**

Run: `hugo --gc --minify`
Working directory: `/home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch`
Expected: PASS

- [ ] **Step 4: Review diffs for both repos**

Run: `git status -sb`
Expected: only intended changes in the current repo

Run: `git -C /home/avisaha/sahaavi.github.io/.worktrees/llm-engineering-from-scratch status -sb`
Expected: only intended blog-post changes in the portfolio repo

- [ ] **Step 5: Commit**

```bash
git add projects/01-tokenizer-from-scratch/tests/test_tokenizer.py projects/01-tokenizer-from-scratch/tests/test_artifacts.py projects/01-tokenizer-from-scratch/tokenizer.py projects/01-tokenizer-from-scratch/artifacts.py projects/01-tokenizer-from-scratch/main.py projects/01-tokenizer-from-scratch/README.md
git commit -m "Finish tokenizer beginner rewrite"
```
