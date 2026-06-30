# Tokenizer From Scratch: Blog Draft

## The Hard Concept

Before a model can learn from text, the text has to become symbols. A tokenizer is the rulebook for that conversion. With byte-pair encoding, the rulebook is learned from repeated neighboring symbols.

## Mental Model

BPE starts with tiny pieces. In this project the pieces are raw UTF-8 bytes, so every input is representable. Training repeatedly asks one question:

> Which adjacent pair appears most often?

Then it merges that pair into a new token. Over time, common fragments become single tokens, while rare text can still fall back to smaller byte pieces.

## Build It

The scratch implementation does four things:

1. Count adjacent token pairs.
2. Pick the most frequent pair.
3. Merge every non-overlapping instance of that pair.
4. Replay learned merges to encode new text.

## Interactive Demo

The BPE Merge Microscope replays the training trace step by step. It shows the selected pair, the before/after token tiles, pair frequencies, vocabulary growth, and compression ratio.

## Plot It

The first chart compares tokens per character across common text, rare words, code, math, emoji, and multilingual text.

The second chart shows how many learned tokens are one byte, two bytes, three bytes, and longer.

## Break It

The tokenizer is reversible on all stress examples, but it does not compress all inputs equally. Multilingual and emoji examples often use many byte-level pieces because the tiny corpus did not contain enough similar patterns.

## What Surprised Me

Byte-level BPE separates two ideas that are easy to mix up:

- Reversibility: can the tokenizer recover the exact original text?
- Readability: do individual token pieces look clean to humans?

The tokenizer can be reversible even when token pieces look awkward.

## What I Would Try Next

I would train on a larger corpus, compare against a unigram tokenizer, and measure how vocabulary size changes context usage and encode latency.

## Links

- GitHub repository: https://github.com/sahaavi/llm-engineering-from-scratch
- Interactive demo: portfolio post
