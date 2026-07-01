import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import BPETokenizer, bytes_to_display


class TokenizerCoreTests(unittest.TestCase):
    def test_train_encode_decode_round_trip(self) -> None:
        tokenizer = BPETokenizer().train(
            ["low lower lowest", "low low lower"],
            vocab_size=260,
        )

        token_ids = tokenizer.encode("low lower")

        self.assertEqual(tokenizer.decode(token_ids), "low lower")
        self.assertGreater(len(tokenizer.merges), 0)

    def test_byte_display_keeps_whitespace_and_hex_fallback_clear(self) -> None:
        self.assertEqual(bytes_to_display(b" "), "␠")
        self.assertEqual(bytes_to_display(b"\n"), "\\n")
        self.assertEqual(bytes_to_display(bytes([0xF0])), "0xf0")


if __name__ == "__main__":
    unittest.main()
