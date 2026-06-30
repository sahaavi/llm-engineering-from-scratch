# Tokenizer From Scratch Notes

## Build It

- [x] Implement byte-level BPE pair counting.
- [x] Implement non-overlapping merge application.
- [x] Train merges from a small corpus.
- [x] Encode and decode real examples.
- [x] Export a trace for the interactive widget.

## Plot It

- [x] Generate a compression chart.
- [x] Generate a token length distribution chart.
- [x] Export metrics as JSON.

## Break It

- [x] Try rare words.
- [x] Try code and indentation.
- [x] Try math notation.
- [x] Try emojis and mixed Unicode.
- [x] Try multilingual text.

## Explain It

Expected: larger vocabularies should compress repeated text better.

Observed: repeated patterns like `tokenization` and `text` compress into longer pieces, but rare/multilingual inputs often fall back to byte-level tokens.

Surprise: byte-level BPE can be perfectly reversible even when individual token pieces are not pretty human-readable strings.

Next: compare against a unigram/SentencePiece-style tokenizer and test larger, more diverse corpora.

## Ship It

- [x] GitHub project folder.
- [x] Static trace for portfolio widget.
- [x] Blog draft in `writeup.md`.
- [x] Portfolio post planned as the public exhibit.
