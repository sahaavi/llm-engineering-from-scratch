# 01. Tokenizer From Scratch

This project builds a small byte-level BPE tokenizer from scratch and pairs it with a beginner-friendly blog walkthrough.

## Start Here

Begin with the blog post:

- [Tokenizer From Scratch: BPE as Learned Compression](https://avisheksaha.com/posts/llm-engineering-from-scratch-tokenizer/)

## Hard Concept

Tokenization is not a boring preprocessing step. It is a learned compression tradeoff. It decides how text becomes symbols, which affects context length, rare words, code, math, emojis, multilingual behavior, latency, and quality.

## Run

From this folder:

```bash
python main.py
```

That command prints a demo and writes deterministic artifacts under `artifacts/`.

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
- `preview.png` / `preview.gif`: lightweight visual previews
- `widget/`: static BPE Merge Microscope demo source

## What To Notice

- Repeated training text collapses into longer pieces.
- Rare words still round-trip because byte-level fallback never loses information.
- Unicode can decode correctly even when individual byte-level pieces look strange.
- Larger vocabularies reduce token count on familiar patterns, but they do not magically understand every new input.

## Further Study

- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Companion Colab notebook](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
