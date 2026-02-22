# Brot

Fast tokenization library for OCaml.

Brot tokenizes text into token IDs for language models and reverses the
process. It is part of the Raven ecosystem. It loads and saves HuggingFace
`tokenizer.json` files, supports BPE, WordPiece, Unigram, word-level, and
character-level algorithms, and is 1.3-6x faster than HuggingFace
tokenizers on most benchmarks (measured against the Rust crate directly,
not just the Python bindings — see [bench/](bench/)).

## Features

- Tokenization algorithms: BPE, WordPiece, Unigram, word-level, character-level
- HuggingFace compatible: load and save `tokenizer.json` files, load
  vocab/merges model files
- Composable pipeline: normalizer, pre-tokenizer, post-processor, decoder
  — each stage independently configurable
- Rich encoding output: token IDs, string tokens, byte offsets, attention
  masks, type IDs, word IDs, special token masks
- Training: train BPE, WordPiece, Unigram, and word-level tokenizers from
  scratch
- Performance: 1.3-6x faster than HuggingFace tokenizers (Rust native) on
  most benchmarks — see [bench/](bench/) for details

## Quick Start

<!-- $MDX skip -->
```ocaml
open Brot

let () =
  (* Load a pretrained HuggingFace tokenizer *)
  let tokenizer = from_file "tokenizer.json" |> Result.get_ok in

  (* Encode text to token IDs *)
  let encoding = encode tokenizer "Hello world!" in
  let ids = Encoding.ids encoding in
  Printf.printf "Token IDs: ";
  Array.iter (fun id -> Printf.printf "%d " id) ids;
  print_newline ();

  (* Decode back to text *)
  let text = decode tokenizer ids in
  Printf.printf "Decoded: %s\n" text
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
