#!/usr/bin/env python3
"""Trim the parity tokenizers to what the parity corpora need.

Run from anywhere, after `packages/brot/bench/download_data.sh`:

    uv run --with tokenizers==0.23.1 python3 packages/brot/test/gen/trim_parity_tokenizers.py

`test_parity.ml` checks brot against the encodings `gen_parity_expected.py`
records from the full tokenizer files, which are tens of megabytes and are
downloaded rather than committed. This writes, for each of them whose licence
allows redistribution, a copy that keeps only the vocabulary and merges the
corpora and `EXTRA` reach, to `support/tokenizers/<name>.json`, and then
checks that the reference gives every recorded row again from the copy, and
the full file's ids for `EXTRA`: the copy encodes and decodes those texts
exactly as the full file does. `test_parity.ml` reads the
copies, so its tests run on a fresh checkout. `support/tokenizers/NOTICE`
records where each one comes from and under which licence.

What a model keeps
------------------

The tokens of every row and every added token are kept with their ids, so a
vocabulary can have gaps. An added token outside a BPE model's vocabulary is
put into it with its id: both the reference and brot number such a token from
the size of the model vocabulary, which the trimmed vocabulary no longer has.

- BPE: a merge that never fires on the corpora can go, since removing a pair
  that is never the lowest-ranked one present changes no step. `bpe_merges`
  replays the model on the corpora's words, and the merges kept are those that
  fire, in their original order, with the symbols they go through.
- WordPiece: the greedy longest match of a word tries every vocabulary entry
  in turn, and a word that fails part way through becomes the unknown token, so
  the pieces kept are every piece the match takes on the corpora's words,
  those of the words that fail included, as `wordpiece_pieces` replays it. A
  removed entry was then never the longest one present.
- Unigram: the ids are positions in the vocabulary, so a piece that goes is
  replaced by a placeholder no text contains, a character of the private use
  plane 15. The pieces of the best segmentations stay, and with them the
  lowest score, which prices the unknown token.

Tokenizers
----------

`COMMITTED` lists the tokenizers whose model card on huggingface.co gives a
licence that allows redistribution, MIT or Apache-2.0; `support/tokenizers/
NOTICE` gives the source, licence and copyright of each copy. `deepseek_v3` is
taken from deepseek-ai/DeepSeek-V3-0324, whose card gives MIT and whose
tokenizer.json is the same file as DeepSeek-V3's. Two are not committed:
`llama` (hf-internal-testing/llama-tokenizer), whose card states no licence,
the tokenizer being Meta's LLaMA one, and `llama3` (NousResearch/Llama-3.2-1B),
under the Llama 3.2 Community License, which binds whoever redistributes it to
its acceptable use policy and attribution terms. Their parity tests read the
downloaded files and skip without them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

from gen_parity_expected import TOKENIZERS, documents, load_tokenizer, rows
from gen_stride_expected import PAIR, SINGLE

COMMITTED = (
    "gpt2",
    "bert_base",
    "roberta_base",
    "t5_base",
    "mistral",
    "qwen2_5",
    "deepseek_v3",
    "gpt_oss",
    "phi4",
    "mistral_nemo",
)

# Texts other suites encode with a trimmed copy and expect the full file's ids
# for: the stride differential of test_tokenization.ml, and
# test_hf_tokenizers.ml.
EXTRA = (SINGLE, *PAIR, "Hello world!", "Hello world", "A quick test")

# The first code point of the supplementary private use area A.
PLACEHOLDER_BASE = 0xF0000


def encodings(tokenizer: Tokenizer, text: str) -> list[list[int]]:
    """The ids of every row `rows` records for corpus `text`."""
    out = []
    for document in documents(text):
        out.append(tokenizer.encode(document, add_special_tokens=False).ids)
        out.append(tokenizer.encode(document, add_special_tokens=True).ids)
        tokenizer.encode_special_tokens = True
        out.append(tokenizer.encode(document, add_special_tokens=False).ids)
        tokenizer.encode_special_tokens = False
    out.append(
        tokenizer.encode(text.removesuffix("\n"), add_special_tokens=False).ids
    )
    return out


def words(tokenizer: Tokenizer, text: str, added: list[str]) -> set[str]:
    """The pre-tokenized words of every document of `text` and of the whole of
    it, whole and cut at every added token, with and without the white space
    an added token's `lstrip` or `rstrip` takes in at the cut. That covers the
    stretches the model is given."""
    found = set()
    for document in [*documents(text), text.removesuffix("\n")]:
        stretches = [document]
        for content in added:
            stretches = [
                piece
                for stretch in stretches
                for piece in stretch.split(content)
            ]
        stripped = {
            strip(piece)
            for piece in stretches
            for strip in (str.lstrip, str.rstrip, str.strip)
        }
        for stretch in {document, *stretches, *stripped}:
            normalized = (
                tokenizer.normalizer.normalize_str(stretch)
                if tokenizer.normalizer
                else stretch
            )
            pieces = tokenizer.pre_tokenizer.pre_tokenize_str(normalized)
            found.update(word for word, _ in pieces)
    return found


def wordpiece_pieces(model: dict, word: str) -> set[str]:
    """The pieces WordPiece's greedy longest match takes on `word`, up to where
    it fails."""
    vocab = model["vocab"]
    prefix = model["continuing_subword_prefix"]
    if len(word) > model["max_input_chars_per_word"]:
        return set()
    taken = set()
    start = 0
    while start < len(word):
        stop = len(word)
        piece = None
        while start < stop:
            candidate = word[start:stop]
            if start > 0:
                candidate = prefix + candidate
            if candidate in vocab:
                piece = candidate
                break
            stop -= 1
        if piece is None:
            break
        taken.add(piece)
        start = stop
    return taken


def merge_parts(merge) -> tuple[str, str]:
    if isinstance(merge, str):
        left, right = merge.split(" ")
        return left, right
    left, right = merge
    return left, right


def bpe_merges(
    model: dict, ranks: dict, word: str
) -> tuple[set[int], set[str]]:
    """The ranks of the merges BPE applies to `word`, and the symbols it goes
    through: from its characters, the lowest-ranked adjacent pair is merged,
    the leftmost of equal ones, until no pair is a merge."""
    vocab = model["vocab"]
    if model.get("ignore_merges") and word in vocab:
        return set(), {word}
    symbols = []
    for char in word:
        if char in vocab:
            symbols.append(char)
        elif model.get("byte_fallback"):
            symbols.extend(f"<0x{byte:02X}>" for byte in char.encode("utf-8"))
        elif model.get("unk_token"):
            symbols.append(model["unk_token"])
    fired = set()
    seen = set(symbols)
    while True:
        best = None
        for i in range(len(symbols) - 1):
            rank = ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, i)
        if best is None:
            return fired, seen
        rank, i = best
        fired.add(rank)
        symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
        seen.add(symbols[i])


def trim_bpe(model: dict, kept: set[str], seen_words: set[str]) -> None:
    for affix in ("continuing_subword_prefix", "end_of_word_suffix"):
        assert not model.get(affix), f"{affix} is not replayed"
    ranks = {
        merge_parts(merge): rank for rank, merge in enumerate(model["merges"])
    }
    needed = set(kept)
    fired = set()
    for word in seen_words:
        word_fired, symbols = bpe_merges(model, ranks, word)
        fired |= word_fired
        needed |= symbols
    model["vocab"] = {
        token: id for token, id in model["vocab"].items() if token in needed
    }
    model["merges"] = [
        merge for rank, merge in enumerate(model["merges"]) if rank in fired
    ]


def trim_wordpiece(model: dict, kept: set[str], seen: set[str]) -> None:
    taken = set(kept)
    for word in seen:
        taken |= wordpiece_pieces(model, word)
    taken.add(model["unk_token"])
    model["vocab"] = {
        token: id for token, id in model["vocab"].items() if token in taken
    }


def trim_unigram(model: dict, kept_ids: set[int]) -> None:
    vocab = model["vocab"]
    lowest = min(score for _, score in vocab)
    kept_ids = kept_ids | {model["unk_id"]}
    trimmed = []
    for id, (piece, score) in enumerate(vocab):
        if id in kept_ids or score == lowest:
            trimmed.append([piece, score])
        else:
            trimmed.append([chr(PLACEHOLDER_BASE + id), 0.0])
    model["vocab"] = trimmed


def trim(name: str, full: Path, parity_dir: Path) -> dict:
    tokenizer = load_tokenizer(full)
    with full.open(encoding="utf-8") as file:
        data = json.load(file)
    model = data["model"]
    added = [token["content"] for token in data["added_tokens"]]
    texts = [
        (parity_dir / f"{corpus}.txt").read_bytes().decode("utf-8")
        for corpus in TOKENIZERS[name]
    ]
    texts.append("\n====\n".join(EXTRA) + "\n")
    ids = {
        id for text in texts for row in encodings(tokenizer, text) for id in row
    }
    ids |= {token["id"] for token in data["added_tokens"]}
    kept = {tokenizer.id_to_token(id) for id in ids}

    seen = {word for text in texts for word in words(tokenizer, text, added)}
    if "merges" in model:
        trim_bpe(model, kept, seen)
        vocab = model["vocab"]
        for token in data["added_tokens"]:
            vocab.setdefault(token["content"], token["id"])
    elif "unk_id" in model:
        trim_unigram(model, ids)
    else:
        trim_wordpiece(model, kept, seen)
    return data


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def lines(values) -> str:
    """A JSON array or object written one member to a line."""
    if isinstance(values, dict):
        members = [f"{compact(k)}:{compact(v)}" for k, v in values.items()]
        return "{\n" + ",\n".join(members) + "\n}"
    return "[\n" + ",\n".join(compact(v) for v in values) + "\n]"


def dump(data: dict) -> str:
    """`data` as JSON with every vocabulary entry, merge and added token on a
    line of its own, so that a change to one shows as a change to one line."""
    model = ",\n".join(
        f"{compact(k)}:{lines(v) if k in ('vocab', 'merges') else compact(v)}"
        for k, v in data["model"].items()
    )
    members = []
    for key, value in data.items():
        if key == "model":
            text = "{\n" + model + "\n}"
        elif key == "added_tokens":
            text = lines(value)
        else:
            text = compact(value)
        members.append(f"{compact(key)}:{text}")
    return "{\n" + ",\n".join(members) + "\n}\n"


def check(name: str, full: Path, trimmed: Path, parity_dir: Path) -> bool:
    """Whether the reference gives every recorded row again from `trimmed`,
    and the ids `full` gives for `EXTRA`."""
    tokenizer = load_tokenizer(trimmed)
    reference = load_tokenizer(full)
    same = True
    for text in EXTRA:
        for specials in (False, True):
            expected = reference.encode(text, add_special_tokens=specials)
            actual = tokenizer.encode(text, add_special_tokens=specials)
            if expected.ids != actual.ids:
                print(f"{trimmed}: {text!r} ids differ", file=sys.stderr)
                same = False
    for corpus in TOKENIZERS[name]:
        text = (parity_dir / f"{corpus}.txt").read_bytes().decode("utf-8")
        recorded = [
            line
            for line in (parity_dir / f"{corpus}.{name}.ids")
            .read_text(encoding="utf-8")
            .removesuffix("\n")
            .split("\n")
            if not line.startswith("#")
        ]
        if rows(tokenizer, text) != recorded:
            print(f"{trimmed}: {corpus}.txt rows differ", file=sys.stderr)
            same = False
    return same


def main() -> int:
    test_root = Path(__file__).resolve().parents[1]
    parity_dir = test_root / "support" / "parity"
    out_dir = test_root / "support" / "tokenizers"
    models_dir = test_root.parent / "bench" / "data"
    out_dir.mkdir(exist_ok=True)

    ok = True
    for name in COMMITTED:
        full = models_dir / f"{name}.json"
        if not full.exists():
            print(
                f"missing {full}; run packages/brot/bench/download_data.sh",
                file=sys.stderr,
            )
            return 1
        data = trim(name, full, parity_dir)
        out = out_dir / f"{name}.json"
        out.write_text(dump(data), encoding="utf-8", newline="\n")
        same = check(name, full, out, parity_dir)
        ok = ok and same
        print(
            f"wrote {out} ({out.stat().st_size} bytes, "
            f"{'same rows' if same else 'ROWS DIFFER'}, "
            f"tokenizers {tokenizers.__version__})"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
