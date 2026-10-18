# Reviewing brot

## Design

- A special case for one tokenizer file or model: which stage of the
  pipeline lacks the general rule?
- A new C loop beyond the three in `AGENTS.md`.

## Semantics

References: Hugging Face's tokenizers, whose output brot promises to match;
the Unicode standard for normalisation forms (UAX #15) and RFC 3629 for
UTF-8.

Failure here is a token sequence that differs from Hugging Face's, or
offsets that point at the wrong bytes.

- C kernels: no read past a buffer's end, and agreement with the OCaml
  version on the same input.
- Invalid UTF-8, characters outside the basic plane, every normalisation
  form.
- Added and special tokens around normalisation and pre-tokenization.

## Skip

`bench/data/`, `bench/bench_rust/`, `bench/brot.thumper`.
