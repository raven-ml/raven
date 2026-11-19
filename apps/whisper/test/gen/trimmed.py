"""Whisper's tokenizers over a rank table, as upstream's get_encoding
(whisper/tokenizer.py at 86098128) builds them over the full one: the ranks,
then the specials numbered after them. Imported by the generators beside it.
"""
import base64

import tiktoken
from whisper.tokenizer import LANGUAGES

PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_ranks(path):
    """The ranks of a .tiktoken file, {token bytes: rank}."""
    ranks = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                tok, rank = line.split()
                ranks[base64.b64decode(tok)] = int(rank)
    return ranks


def encoding(name, ranks, languages):
    """get_encoding's construction over [ranks], with the first [languages]
    language tokens."""
    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>", "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:languages]],
        "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
        "<|nospeech|>", "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = {}
    for s in specials:
        special_tokens[s] = n_vocab
        n_vocab += 1
    return tiktoken.Encoding(name=name, explicit_n_vocab=n_vocab, pat_str=PATTERN,
                             mergeable_ranks=ranks, special_tokens=special_tokens)
