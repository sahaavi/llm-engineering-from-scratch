# LLM Engineering From Scratch

A project-first series for rebuilding core LLM mechanics from first principles with small runnable implementations, plots, stress cases, and interactive artifacts.

Roadmap inspiration: [Ahmad Osman](https://x.com/TheAhmadOsman) ([@TheAhmadOsman](https://x.com/TheAhmadOsman)) and his article, ["Step-By-Step LLM Engineering Projects (2026 Edition)"](https://x.com/TheAhmadOsman/article/2058745340895870985). This repository uses the article as a roadmap reference; the implementations, experiments, and artifacts are independent.

## Roadmap

| # | Project | Hard concept |
|---:|---|---|
| 1 | Tokenizer from scratch | Tokenization is a learned compression tradeoff that changes context use, rare-word behavior, multilingual text, code, latency, and model quality. |
| 2 | One-hot vectors and learned embeddings | Token IDs become meaningful only when they are mapped into learned geometry. |
| 3 | Positional methods | Attention needs order, and each positional method encodes different assumptions. |
| 4 | Scaled dot-product attention | Attention is weighted retrieval from context. |
| 5 | Multi-head attention | Different heads can learn different relational patterns. |
| 6 | One decoder block | LLM behavior comes from the interaction of attention, MLPs, norms, residuals, and projections. |
| 7 | Mini-former | The training loop teaches how models actually learn. |
| 8 | Language-model objectives | Objective choice shapes capability and failure modes. |

## Projects

| Project | Status | Run |
|---|---|---|
| [01-tokenizer-from-scratch](projects/01-tokenizer-from-scratch/) | Implemented | `python projects/01-tokenizer-from-scratch/main.py` |

## Repository Pattern

Each project stays small and readable:

```text
projects/<number>-<name>/
  README.md
  main.py
  lab.ipynb
  requirements.txt
  widget/
  artifacts/
```

`main.py` is the clean scratch implementation. The notebook is the workshop version. `widget/` contains static interactive demo code. `artifacts/` contains plots, traces, failure galleries, and preview media used by public demos and articles.
