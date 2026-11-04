# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Trim the parity tokenizers to what the parity corpora need.

Run from anywhere, after `packages/brot/bench/download_data.sh`:

    uv run packages/brot/test/gen/trim_parity_tokenizers.py [--check]

`test_parity.ml` checks brot against the encodings `gen_parity_expected.py`
records from the full tokenizer files, which are tens of megabytes and are
downloaded rather than committed. This writes, for each of them whose licence
allows redistribution, a copy that keeps only the part of the vocabulary and
merges the corpora and `EXTRA` reach, to `support/tokenizers/<name>.json`.
It then checks that the reference gives every recorded row again from the
copy, and the full file's ids for `EXTRA`, so the copy encodes and decodes
those texts exactly as the full file does. With `--check` it writes nothing,
fails if a committed copy differs from what it would write, and runs the same
check on the committed copy. The tests read the full file when
it is downloaded, as on CI, and the copy otherwise, so they run on a fresh
checkout. `support/tokenizers/NOTICE` records where each one comes from and
under which licence.

What a model keeps
------------------

The tokens of every row and every added token are kept with their ids, so a
vocabulary can have gaps. An added token outside a BPE model's vocabulary is
put into it with its id: both the reference and brot number such a token from
the size of the model vocabulary, which the trimmed vocabulary no longer has.

A copy keeps more than the corpora's tokens: it keeps every candidate the
algorithm weighs on their words, so that an implementation that picks the
wrong one still gives different ids on the copy, as it would on the full file.
What goes is what no step on these words could look at, so the copy encodes
them as the full file does.

- BPE: every merge whose result is a run of consecutive symbols of a word,
  which takes in every merge any order of merging could apply to it, the ones
  that fire included, in their original order. A merge that is never present
  is never the lowest-ranked one.
- WordPiece: every entry that starts where the greedy longest match of a word
  starts a piece, the one it takes and the shorter ones it passes over, up to
  where a failing word fails, as `wordpiece_pieces` replays it.
- Unigram: every piece in the lattice of a word, that is, every piece that is
  a substring of it. The ids are positions in the vocabulary, so a piece that
  goes is replaced by a placeholder no text contains, a character of the
  private use plane 15. The lowest score stays, as it prices the unknown
  token.

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

from gen_parity_expected import TOKENIZERS, documents, rows
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
    """The entries WordPiece's greedy longest match tries on `word` and finds:
    at each start it reaches, every entry that starts there, the longest of
    which it takes, up to where it fails."""
    vocab = model["vocab"]
    prefix = model["continuing_subword_prefix"]
    if len(word) > model["max_input_chars_per_word"]:
        return set()
    found = set()
    start = 0
    while start < len(word):
        longest = None
        for stop in range(len(word), start, -1):
            candidate = word[start:stop]
            if start > 0:
                candidate = prefix + candidate
            if candidate in vocab:
                found.add(candidate)
                longest = longest or stop
        if longest is None:
            break
        start = longest
    return found


def spans(symbols: list[str], longest: int) -> set[str]:
    """Every concatenation of consecutive `symbols` of at most `longest`
    characters: every token some order of merges could build from them."""
    found = set()
    for i in range(len(symbols)):
        text = ""
        for symbol in symbols[i:]:
            text += symbol
            if len(text) > longest:
                break
            found.add(text)
    return found


def merge_parts(merge) -> tuple[str, str]:
    if isinstance(merge, str):
        left, right = merge.split(" ")
        return left, right
    left, right = merge
    return left, right


def bpe_symbols(model: dict, word: str) -> list[str]:
    """The symbols BPE starts `word` from: its characters, or the bytes of
    those the vocabulary lacks under byte fallback."""
    vocab = model["vocab"]
    symbols = []
    for char in word:
        if char in vocab:
            symbols.append(char)
        elif model.get("byte_fallback"):
            symbols.extend(f"<0x{byte:02X}>" for byte in char.encode("utf-8"))
        elif model.get("unk_token"):
            symbols.append(model["unk_token"])
    return symbols


def trim_bpe(model: dict, kept: set[str], seen_words: set[str]) -> None:
    for affix in ("continuing_subword_prefix", "end_of_word_suffix"):
        assert not model.get(affix), f"{affix} is not replayed"
    longest = max(len(token) for token in model["vocab"])
    reachable = set(kept)
    for word in seen_words:
        reachable |= spans(bpe_symbols(model, word), longest)
    merges = [
        merge
        for merge in model["merges"]
        if "".join(merge_parts(merge)) in reachable
    ]
    needed = reachable | {part for merge in merges for part in merge_parts(merge)}
    model["vocab"] = {
        token: id for token, id in model["vocab"].items() if token in needed
    }
    model["merges"] = merges


def trim_wordpiece(model: dict, kept: set[str], seen: set[str]) -> None:
    taken = set(kept)
    for word in seen:
        taken |= wordpiece_pieces(model, word)
    taken.add(model["unk_token"])
    model["vocab"] = {
        token: id for token, id in model["vocab"].items() if token in taken
    }


def trim_unigram(model: dict, kept_ids: set[int], seen: set[str]) -> None:
    vocab = model["vocab"]
    lowest = min(score for _, score in vocab)
    longest = max(len(piece) for piece, _ in vocab)
    lattice = set()
    for word in seen:
        lattice |= spans(list(word), longest)
    kept_ids = kept_ids | {model["unk_id"]}
    trimmed = []
    for id, (piece, score) in enumerate(vocab):
        if id in kept_ids or piece in lattice or score == lowest:
            trimmed.append([piece, score])
        else:
            trimmed.append([chr(PLACEHOLDER_BASE + id), 0.0])
    model["vocab"] = trimmed


def trim(name: str, full: Path, parity_dir: Path) -> dict:
    tokenizer = Tokenizer.from_file(str(full))
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
        trim_unigram(model, ids, seen)
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
    tokenizer = Tokenizer.from_file(str(trimmed))
    reference = Tokenizer.from_file(str(full))
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
    check_only = "--check" in sys.argv[1:]

    ok = True
    for name in COMMITTED:
        full = models_dir / f"{name}.json"
        if not full.exists():
            print(
                f"missing {full}; run packages/brot/bench/download_data.sh",
                file=sys.stderr,
            )
            return 1
        text = dump(trim(name, full, parity_dir)).encode("utf-8")
        out = out_dir / f"{name}.json"
        if check_only:
            fresh = out.exists() and out.read_bytes() == text
            if not fresh:
                print(f"{out} differs from what the generator writes",
                      file=sys.stderr)
            same = check(name, full, out, parity_dir)
            ok = ok and fresh and same
            continue
        out.write_bytes(text)
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
