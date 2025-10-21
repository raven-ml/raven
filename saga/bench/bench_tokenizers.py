from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel, WordPiece
from tokenizers.pre_tokenizers import Whitespace

_ROOT = Path(__file__).resolve().parents[2]
_UBENCH_DIR = _ROOT / "vendor" / "ubench"
if str(_UBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_UBENCH_DIR))

import ubench  # type: ignore


BACKEND_NAME = "HuggingFace"


def create_bpe_tokenizer() -> Tokenizer:
    """Create a simple BPE tokenizer matching the OCaml version."""
    vocab = {
        "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6, "t": 7,
        "i": 8, "n": 9, "g": 10, " ": 11, "!": 12, ".": 13, ",": 14,
        "he": 15, "ll": 16, "lo": 17, "wo": 18, "or": 19, "ld": 20,
        "th": 21, "in": 22, "ng": 23,
    }
    merges = [
        ("h", "e"), ("l", "l"), ("l", "o"), ("w", "o"), ("o", "r"),
        ("l", "d"), ("t", "h"), ("i", "n"), ("n", "g"),
    ]
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))
    return tokenizer


def create_wordpiece_tokenizer() -> Tokenizer:
    """Create a WordPiece tokenizer matching the OCaml version."""
    vocab = {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
        "the": 4, "quick": 5, "brown": 6, "fox": 7,
        "jumps": 8, "over": 9, "lazy": 10, "dog": 11,
        "hello": 12, "world": 13, "how": 14, "are": 15,
        "you": 16, "doing": 17, "today": 18,
        "##ing": 19, "##ed": 20, "##s": 21,
    }
    tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    return tokenizer


def create_wordlevel_tokenizer() -> Tokenizer:
    """Create a word-level tokenizer matching the OCaml version."""
    vocab = {
        "the": 0, "quick": 1, "brown": 2, "fox": 3,
        "jumps": 4, "over": 5, "lazy": 6, "dog": 7,
        "hello": 8, "world": 9, "how": 10, "are": 11,
        "you": 12, "doing": 13, "today": 14, "[UNK]": 15,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


# Encoding benchmarks
class EncodingBenchmarks:
    """Encoding benchmarks - core performance metric."""

    TEXT_SIZES = [
        ("10K chars", "a" * 10000),
        ("100K chars", "a" * 100000),
        ("1M chars", "a" * 1000000),
    ]

    @staticmethod
    def build() -> List[Any]:
        # BPE encoding
        tok_bpe = create_bpe_tokenizer()
        bench_bpe = ubench.bench_param(
            "BPE encode",
            lambda param: tok_bpe.encode(param),
            params=EncodingBenchmarks.TEXT_SIZES
        )

        # WordPiece encoding
        tok_wp = create_wordpiece_tokenizer()
        bench_wordpiece = ubench.bench_param(
            "WordPiece encode",
            lambda param: tok_wp.encode(param),
            params=EncodingBenchmarks.TEXT_SIZES
        )

        # WordLevel encoding
        tok_wl = create_wordlevel_tokenizer()
        bench_wordlevel = ubench.bench_param(
            "WordLevel encode",
            lambda param: tok_wl.encode(param),
            params=EncodingBenchmarks.TEXT_SIZES
        )

        return bench_bpe + bench_wordpiece + bench_wordlevel


# Decoding benchmarks
class DecodingBenchmarks:
    """Decoding benchmarks - important for text generation."""

    TOKEN_COUNTS = [
        ("10K tokens", [i % 10 for i in range(10000)]),
        ("100K tokens", [i % 10 for i in range(100000)]),
    ]

    @staticmethod
    def build() -> List[Any]:
        # BPE decoding
        tok_bpe = create_bpe_tokenizer()
        bench_bpe = ubench.bench_param(
            "BPE decode",
            lambda param: tok_bpe.decode(param),
            params=DecodingBenchmarks.TOKEN_COUNTS
        )

        # WordPiece decoding
        tok_wp = create_wordpiece_tokenizer()
        bench_wordpiece = ubench.bench_param(
            "WordPiece decode",
            lambda param: tok_wp.decode(param),
            params=DecodingBenchmarks.TOKEN_COUNTS
        )

        return bench_bpe + bench_wordpiece


# Batch encoding benchmarks
class BatchBenchmarks:
    """Batch encoding benchmarks - important for real-world usage."""

    SAMPLE_TEXTS = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! How are you doing today?",
        "Machine learning and natural language processing are fascinating fields of study.",
        "This is a longer sentence with multiple clauses, punctuation marks, and various word lengths to test tokenization performance.",
        "1234567890 !@#$%^&*() Testing special characters and numbers in tokenization.",
    ]

    BATCH_SIZES = [("100 items", 100), ("1K items", 1000)]

    @staticmethod
    def build() -> List[Any]:
        # Prepare batch data
        batch_params = [
            (name, [BatchBenchmarks.SAMPLE_TEXTS[i % len(BatchBenchmarks.SAMPLE_TEXTS)] for i in range(size)])
            for name, size in BatchBenchmarks.BATCH_SIZES
        ]

        # BPE batch encoding
        tok_bpe = create_bpe_tokenizer()
        bench_bpe = ubench.bench_param(
            "BPE batch",
            lambda param: tok_bpe.encode_batch(param),
            params=batch_params
        )

        # WordPiece batch encoding
        tok_wp = create_wordpiece_tokenizer()
        bench_wordpiece = ubench.bench_param(
            "WordPiece batch",
            lambda param: tok_wp.encode_batch(param),
            params=batch_params
        )

        return bench_bpe + bench_wordpiece


# Serialization benchmarks
class SerializationBenchmarks:
    """Serialization benchmarks - I/O performance."""

    @staticmethod
    def build() -> List[Any]:
        # to_str (equivalent to to_json)
        tok = create_wordpiece_tokenizer()
        bench_to_json = ubench.bench("to_json", lambda: tok.to_str())

        # from_file
        import tempfile
        import os

        def setup():
            tok = create_wordpiece_tokenizer()
            fd, path = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            tok.save(path)
            return path

        def teardown(path):
            try:
                os.remove(path)
            except:
                pass

        def bench_fn(path):
            Tokenizer.from_file(path)

        bench_from_file = ubench.bench_with_setup(
            "from_file", setup=setup, teardown=teardown, f=bench_fn
        )

        return [bench_to_json, bench_from_file]


# Vocabulary operations benchmarks
class VocabBenchmarks:
    """Vocabulary operations benchmarks."""

    @staticmethod
    def build() -> List[Any]:
        # add_tokens
        new_tokens = [f"token{i}" for i in range(100)]

        def bench_fn():
            tok = create_wordlevel_tokenizer()
            tok.add_tokens(new_tokens)

        return [ubench.bench("add_tokens (100)", bench_fn)]


def build_benchmarks() -> List[Any]:
    """Build all tokenizer benchmarks grouped by category."""
    return [
        ubench.group("Encoding", EncodingBenchmarks.build()),
        ubench.group("Decoding", DecodingBenchmarks.build()),
        ubench.group("Batch", BatchBenchmarks.build()),
        ubench.group("Serialization", SerializationBenchmarks.build()),
        ubench.group("Vocab", VocabBenchmarks.build()),
    ]


def default_config() -> ubench.Config:
    """Create default benchmark configuration."""
    return (
        ubench.Config.default()
        .time_limit(1.0)
        .warmup(1)
        .min_measurements(5)
        .min_cpu(0.01)
        .geometric_scale(1.3)
        .gc_stabilization(False)
        .build()
    )


def main() -> None:
    """Main entry point."""
    benchmarks = build_benchmarks()
    config = default_config()
    ubench.run(benchmarks, config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
