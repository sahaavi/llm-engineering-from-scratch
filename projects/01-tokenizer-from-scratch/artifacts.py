from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import struct
import zlib

from tokenizer import BPETokenizer


DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
DEFAULT_WIDGET_DATA_PATH = Path(__file__).with_name("widget") / "data.json"

TINY_CORPUS = [
    "tokenization turns text into model readable symbols.",
    "models do not see words directly; they see token ids.",
    "byte pair encoding learns common byte patterns.",
    "rare terms like hypermicrotokenization should not become unknown.",
    "python code keeps spaces and indentation meaningful.",
    "math uses symbols like x^2 + y^2 = z^2.",
    "cafe café coffee tokenizer tokenizer tokenizer.",
    "the same phrase the same phrase the same phrase repeats.",
]

TRACE_CORPUS = [
    "low lower lowest",
    "low low lower",
    "token token tokenization",
]

STRESS_EXAMPLES = {
    "common": "tokenization turns text into token ids.",
    "rare": "hypermicrotokenization antidisestablishmentarian-ish words",
    "code": "def tokenize(text):\n    return text.encode('utf-8')\n",
    "math": "f(x) = x^2 + 2*x + 1; sum_i p_i log(p_i)",
    "emoji": "I like clean tokenizers ✨ café 👋🏽",
    "multilingual": "Hello नमस्ते こんにちは مرحبا Привет",
}


def summarize_tokenizer(
    tokenizer: BPETokenizer, examples: dict[str, str]
) -> list[dict[str, object]]:
    rows = []
    for name, text in examples.items():
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        assert decoded == text
        rows.append(
            {
                "category": name,
                "characters": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "tokens": len(token_ids),
                "tokens_per_char": round(len(token_ids) / max(1, len(text)), 4),
                "pieces": tokenizer.token_pieces(token_ids),
            }
        )
    return rows


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    rows = []
    row_width = width * 3
    for y in range(height):
        rows.append(b"\x00" + bytes(pixels[y * row_width : (y + 1) * row_width]))
    payload = b"".join(rows)
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(payload, level=9))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def make_canvas(width: int, height: int, color: tuple[int, int, int]) -> bytearray:
    return bytearray(color * (width * height))


def draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        for x in range(x0, x1):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)


def draw_bar_chart(path: Path, values: list[float], colors: list[tuple[int, int, int]]) -> None:
    width, height = 900, 420
    pixels = make_canvas(width, height, (250, 252, 255))
    draw_rect(pixels, width, height, 55, 40, 60, 360, (42, 54, 71))
    draw_rect(pixels, width, height, 55, 355, 850, 360, (42, 54, 71))
    max_value = max(values) if values else 1.0
    gap = 24
    bar_width = max(24, (760 - gap * (len(values) - 1)) // max(1, len(values)))
    for index, value in enumerate(values):
        x0 = 80 + index * (bar_width + gap)
        x1 = x0 + bar_width
        bar_height = int((value / max_value) * 280)
        y0 = 355 - bar_height
        draw_rect(pixels, width, height, x0, y0, x1, 355, colors[index % len(colors)])
    write_png(path, width, height, pixels)


def draw_histogram(path: Path, values: list[int]) -> None:
    counts = Counter(values)
    bins = list(range(1, max(counts, default=1) + 1))
    chart_values = [counts[value] for value in bins]
    draw_bar_chart(path, chart_values, [(11, 107, 87), (23, 105, 170), (138, 90, 0)])


def write_preview(path: Path) -> None:
    width, height = 900, 420
    pixels = make_canvas(width, height, (16, 20, 24))
    colors = [(103, 183, 220), (125, 211, 176), (240, 184, 74), (255, 255, 255)]
    for i in range(11):
        draw_rect(pixels, width, height, 80 + i * 66, 120, 130 + i * 66, 175, colors[i % 4])
    for i in range(6):
        draw_rect(pixels, width, height, 155 + i * 96, 245, 230 + i * 96, 305, colors[(i + 1) % 4])
    write_png(path, width, height, pixels)


def pack_gif_codes(codes: list[tuple[int, int]]) -> bytes:
    output = bytearray()
    accumulator = 0
    bits = 0
    for code, code_size in codes:
        accumulator |= code << bits
        bits += code_size
        while bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bits -= 8
    if bits:
        output.append(accumulator & 0xFF)
    return bytes(output)


def lzw_encode_indices(indices: list[int], color_count: int) -> bytes:
    min_code_size = max(2, (color_count - 1).bit_length())
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    next_code = end_code + 1
    code_size = min_code_size + 1
    dictionary = {(index,): index for index in range(clear_code)}
    codes: list[tuple[int, int]] = [(clear_code, code_size)]

    current = (indices[0],)
    for index in indices[1:]:
        candidate = current + (index,)
        if candidate in dictionary:
            current = candidate
            continue

        codes.append((dictionary[current], code_size))
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            codes.append((clear_code, code_size))
            dictionary = {(value,): value for value in range(clear_code)}
            next_code = end_code + 1
            code_size = min_code_size + 1
        current = (index,)

    codes.append((dictionary[current], code_size))
    codes.append((end_code, code_size))
    return bytes([min_code_size]) + gif_subblocks(pack_gif_codes(codes))


def gif_subblocks(data: bytes) -> bytes:
    chunks = bytearray()
    for index in range(0, len(data), 255):
        chunk = data[index : index + 255]
        chunks.append(len(chunk))
        chunks.extend(chunk)
    chunks.append(0)
    return bytes(chunks)


def write_preview_gif(path: Path) -> None:
    width, height = 180, 90
    palette = [
        (16, 20, 24),
        (103, 183, 220),
        (125, 211, 176),
        (240, 184, 74),
    ]

    def frame(active: int) -> list[int]:
        pixels = [0] * (width * height)
        for tile in range(9):
            color = 1 + ((tile + active) % 3)
            x0 = 18 + tile * 16
            x1 = x0 + (24 if tile == active else 11)
            y0, y1 = 28, 54
            for y in range(y0, y1):
                for x in range(x0, min(width - 8, x1)):
                    pixels[y * width + x] = color
        return pixels

    data = bytearray(b"GIF89a")
    data.extend(struct.pack("<HH", width, height))
    data.extend(bytes([0b10000001, 0, 0]))
    for red, green, blue in palette:
        data.extend(bytes([red, green, blue]))
    data.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")

    for active in (0, 2, 4, 6):
        data.extend(b"\x21\xf9\x04\x04")
        data.extend(struct.pack("<H", 28))
        data.extend(b"\x00\x00")
        data.extend(b"\x2c")
        data.extend(struct.pack("<HHHH", 0, 0, width, height))
        data.extend(b"\x00")
        data.extend(lzw_encode_indices(frame(active), len(palette)))

    data.extend(b";")
    path.write_bytes(bytes(data))


def write_failure_gallery(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# Failure Gallery\n"]
    lines.append("These examples are called failures because they stress compression, not because decoding loses information.\n")
    for row in rows:
        pieces = row["pieces"]
        piece_preview = " | ".join(piece["text"] for piece in pieces[:28])
        if len(pieces) > 28:
            piece_preview += " | ..."
        lines.append(f"## {row['category']}\n")
        lines.append(f"- Characters: {row['characters']}\n")
        lines.append(f"- UTF-8 bytes: {row['utf8_bytes']}\n")
        lines.append(f"- Tokens: {row['tokens']}\n")
        lines.append(f"- Tokens per character: {row['tokens_per_char']}\n")
        lines.append(f"- Token pieces: `{piece_preview}`\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_artifacts(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    widget_data_path: Path = DEFAULT_WIDGET_DATA_PATH,
) -> dict[str, object]:
    artifacts_dir.mkdir(exist_ok=True)

    trace_tokenizer = BPETokenizer().train(TRACE_CORPUS, vocab_size=272, capture_trace=True)
    small = BPETokenizer().train(TINY_CORPUS, vocab_size=265)
    large = BPETokenizer().train(TINY_CORPUS, vocab_size=305)

    small_rows = summarize_tokenizer(small, STRESS_EXAMPLES)
    large_rows = summarize_tokenizer(large, STRESS_EXAMPLES)

    trace_data = {
        "metadata": {
            "project": "01-tokenizer-from-scratch",
            "title": "BPE Merge Microscope",
            "corpus": TRACE_CORPUS,
            "initial_vocab_size": 256,
            "final_vocab_size": len(trace_tokenizer.token_to_id),
        },
        "steps": trace_tokenizer.trace,
        "merges": trace_tokenizer.merge_records(),
        "examples": [
            {"label": key, "text": value}
            for key, value in STRESS_EXAMPLES.items()
        ],
    }
    metrics = {
        "small_vocab": {"vocab_size": len(small.token_to_id), "rows": small_rows},
        "large_vocab": {"vocab_size": len(large.token_to_id), "rows": large_rows},
    }

    write_json(artifacts_dir / "trace.json", trace_data)
    write_json(artifacts_dir / "metrics.json", metrics)
    write_json(widget_data_path, trace_data)

    write_failure_gallery(artifacts_dir / "failure_gallery.md", large_rows)
    draw_bar_chart(
        artifacts_dir / "compression_ratio.png",
        [float(row["tokens_per_char"]) for row in large_rows],
        [(23, 105, 170), (11, 107, 87), (138, 90, 0)],
    )
    token_lengths = [
        len(bytes(piece["bytes"]))
        for row in large_rows
        for piece in row["pieces"]
    ]
    draw_histogram(artifacts_dir / "token_length_distribution.png", token_lengths)
    write_preview(artifacts_dir / "preview.png")
    write_preview_gif(artifacts_dir / "preview.gif")
    return {"trace": trace_data, "metrics": metrics}
