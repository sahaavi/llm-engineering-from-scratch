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
