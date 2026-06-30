# LLM Engineering From Scratch

This is a project-first learning series for rebuilding the most important pieces of the LLM stack from first principles.

The loop for every project is:

1. Build it.
2. Plot it.
3. Break it.
4. Explain it.
5. Ship the artifact.

The roadmap is inspired by Ahmad Osman's article, "Step-By-Step LLM Engineering Projects (2026 Edition)." The article is used as inspiration and is not copied into this repository.

## First Wave

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
| [01-tokenizer-from-scratch](projects/01-tokenizer-from-scratch/) | First build | `python projects/01-tokenizer-from-scratch/main.py` |

## Repository Pattern

Each project stays small and readable:

```text
projects/<number>-<name>/
  README.md
  main.py
  lab.ipynb
  notes.md
  writeup.md
  artifacts/
```

`main.py` is the clean scratch implementation. The notebook is the workshop. `artifacts/` contains plots, traces, failure galleries, and preview media used by the blog post.
