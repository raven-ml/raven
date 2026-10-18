# brot

Tokenizers that load and save Hugging Face `tokenizer.json` files: BPE,
WordPiece, Unigram, word-level and character-level.

- Output matches Hugging Face's tokenizers token for token; `test_parity.ml`
  and `test_hf_tokenizers.ml` hold it.
- C covers three hot loops: the byte-level pre-tokenizer walker, the
  pretoken cache probe, and the short-merge BPE miss path. Their OCaml
  versions stay as the reference and as the bytecode and js_of_ocaml
  fallback. Each C kernel is tested against its OCaml version. Everything
  else stays OCaml.
