import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import BPETokenizer, bytes_to_display, count_pairs, merge_pair


class TokenizerCoreTests(unittest.TestCase):
    def test_count_pairs_counts_adjacent_integer_pairs(self) -> None:
        self.assertEqual(
            count_pairs([1, 2, 1, 2, 3]),
            {(1, 2): 2, (2, 1): 1, (2, 3): 1},
        )

    def test_merge_pair_replaces_non_overlapping_pairs(self) -> None:
        self.assertEqual(merge_pair([1, 2, 1, 2, 1], (1, 2), 99), [99, 99, 1])

    def test_training_uses_integer_ids_and_vocab(self) -> None:
        tokenizer = BPETokenizer().train(["aaaa"], vocab_size=258)

        self.assertIn((ord("a"), ord("a")), tokenizer.merges)
        self.assertEqual(tokenizer.vocab[256], b"aa")
        self.assertEqual(tokenizer.merge_order[0], (ord("a"), ord("a")))

    def test_train_encode_decode_round_trip(self) -> None:
        tokenizer = BPETokenizer().train(
            ["low lower lowest", "low low lower"],
            vocab_size=260,
        )

        token_ids = tokenizer.encode("low lower")

        self.assertEqual(tokenizer.decode(token_ids), "low lower")
        self.assertGreater(len(tokenizer.merges), 0)

    def test_unicode_round_trip(self) -> None:
        tokenizer = BPETokenizer().train(["café नमस्ते ✨ café"], vocab_size=270)
        text = "café नमस्ते ✨"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_decode_rejects_unknown_token_id(self) -> None:
        tokenizer = BPETokenizer()

        with self.assertRaisesRegex(ValueError, "Unknown token id"):
            tokenizer.decode([999])

    def test_trace_keeps_widget_fields(self) -> None:
        tokenizer = BPETokenizer().train(
            ["low lower lowest"],
            vocab_size=258,
            capture_trace=True,
        )

        step = tokenizer.trace[0]

        self.assertIn("selected_pair", step)
        self.assertIn("merged_token", step)
        self.assertIn("top_pairs", step)
        self.assertIn("before_sequences", step)
        self.assertIn("after_sequences", step)
        self.assertIn("metrics", step)

    def test_byte_display_keeps_whitespace_and_hex_fallback_clear(self) -> None:
        self.assertEqual(bytes_to_display(b" "), "␠")
        self.assertEqual(bytes_to_display(b"\n"), "\\n")
        self.assertEqual(bytes_to_display(bytes([0xF0])), "0xf0")


if __name__ == "__main__":
    unittest.main()
