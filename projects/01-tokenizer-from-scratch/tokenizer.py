from __future__ import annotations


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


def text_to_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    counts = {}
    for index in range(len(ids) - 1):
        pair = (ids[index], ids[index + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    new_ids = []
    index = 0
    while index < len(ids):
        if index < len(ids) - 1 and ids[index] == pair[0] and ids[index + 1] == pair[1]:
            new_ids.append(new_id)
            index += 2
        else:
            new_ids.append(ids[index])
            index += 1
    return new_ids


class BPETokenizer:
    """Small byte-level BPE tokenizer used by the beginner walkthrough."""

    def __init__(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_order: list[tuple[int, int]] = []
        self.trace: list[dict[str, object]] = []

    def _most_common_pair(self, counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        best_pair = None
        best_count = -1
        for pair, count in counts.items():
            if count > best_count:
                best_pair = pair
                best_count = count
        if best_pair is None:
            raise ValueError("No pairs to choose from")
        return best_pair

    def _token_to_json(self, token_id: int) -> dict[str, object]:
        return piece_to_json(self.vocab[token_id])

    def _ids_to_json(self, ids: list[int]) -> list[dict[str, object]]:
        pieces = []
        for token_id in ids:
            pieces.append(self._token_to_json(token_id))
        return pieces

    def _sequence_metrics(
        self, ids: list[int], initial_total_tokens: int
    ) -> dict[str, float | int]:
        total_tokens = len(ids)
        return {
            "vocab_size": len(self.vocab),
            "learned_merges": len(self.merges),
            "total_tokens": total_tokens,
            "compression_ratio": round(total_tokens / initial_total_tokens, 4),
        }

    def _trace_step(
        self,
        before: list[int],
        after: list[int],
        pair_counts: dict[tuple[int, int], int],
        best_pair: tuple[int, int],
        new_id: int,
        initial_total_tokens: int,
    ) -> dict[str, object]:
        top_pairs = []
        sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
        for pair, count in sorted_pairs[:8]:
            top_pairs.append(
                {
                    "left": self._token_to_json(pair[0]),
                    "right": self._token_to_json(pair[1]),
                    "count": count,
                }
            )

        return {
            "step": len(self.merge_order),
            "selected_pair": [
                self._token_to_json(best_pair[0]),
                self._token_to_json(best_pair[1]),
            ],
            "merged_token": self._token_to_json(new_id),
            "frequency": pair_counts[best_pair],
            "top_pairs": top_pairs,
            "before_sequences": [self._ids_to_json(before)],
            "after_sequences": [self._ids_to_json(after)],
            "metrics": self._sequence_metrics(after, initial_total_tokens),
        }

    def train(
        self, corpus: list[str], vocab_size: int, *, capture_trace: bool = False
    ) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("byte-level BPE needs vocab_size >= 256")

        training_text = "\n".join(corpus)
        ids = text_to_ids(training_text)
        initial_total_tokens = len(ids)

        while len(self.vocab) < vocab_size:
            pair_counts = count_pairs(ids)
            if not pair_counts:
                break

            best_pair = self._most_common_pair(pair_counts)
            new_id = len(self.vocab)
            new_piece = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            self.merges[best_pair] = new_id
            self.merge_order.append(best_pair)
            self.vocab[new_id] = new_piece

            before = ids
            ids = merge_pair(ids, best_pair, new_id)

            if capture_trace:
                self.trace.append(
                    self._trace_step(
                        before,
                        ids,
                        pair_counts,
                        best_pair,
                        new_id,
                        initial_total_tokens,
                    )
                )

        return self

    def encode(self, text: str) -> list[int]:
        ids = text_to_ids(text)
        for pair in self.merge_order:
            new_id = self.merges[pair]
            ids = merge_pair(ids, pair, new_id)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token id: {token_id}")
            pieces.append(self.vocab[token_id])
        return b"".join(pieces).decode("utf-8")

    def token_pieces(self, token_ids: list[int]) -> list[dict[str, object]]:
        return [self._token_to_json(token_id) for token_id in token_ids]

    def merge_records(self) -> list[dict[str, object]]:
        records = []
        for pair in self.merge_order:
            left, right = pair
            token_id = self.merges[pair]
            records.append(
                {
                    "id": token_id,
                    "left": self._token_to_json(left),
                    "right": self._token_to_json(right),
                    "token": self._token_to_json(token_id),
                }
            )
        return records
