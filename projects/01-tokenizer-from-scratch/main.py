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

    print(f"\nVocabulary size: {len(tokenizer.vocab)}")
    print(f"Learned merges:  {len(tokenizer.merges)}")


def main() -> None:
    print_demo()
    build_artifacts()
    print(f"\nArtifacts written to: {DEFAULT_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
