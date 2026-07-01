# 01. Tokenizer From Scratch

A tiny byte-level BPE tokenizer from scratch with deterministic artifacts: charts, stress cases, preview media, and a static interactive trace.

## Hard Concept

Tokenization is not a boring preprocessing step. It is a learned compression tradeoff. It decides how text becomes symbols, which affects context length, rare words, code, math, emojis, multilingual behavior, latency, and quality.

## Run

From this folder:

```bash
python main.py
```

That command prints a demo and writes deterministic artifacts under `artifacts/`.

## Artifacts

- `trace.json` / `metrics.json`: data for analysis and the interactive widget
- `compression_ratio.png`: token count per character across input types
- `token_length_distribution.png`: byte length distribution of learned token pieces
- `failure_gallery.md`: weird-input examples and what broke or survived
- `preview.png` / `preview.gif`: lightweight visual previews
- `widget/`: static BPE Merge Microscope demo source

## Results

- Repeated training text collapses into longer pieces.
- Rare words still round-trip because byte-level fallback never loses information.
- Unicode can decode correctly even when individual byte-level pieces look strange.
- Larger vocabularies reduce token count on familiar patterns, but they do not magically understand every new input.

## Related Writing

- [Tokenizer From Scratch: BPE as Learned Compression](https://avisheksaha.com/posts/llm-engineering-from-scratch-tokenizer/)
