# `x-gpt2-tokenizer`

Loading a real GPT-2 tokenizer from HuggingFace model files. This example
downloads GPT-2's vocabulary and merges, builds the full byte-level BPE
pipeline, and demonstrates encoding, decoding, and subword inspection.

```bash
dune exec brot/examples/x-gpt2-tokenizer/main.exe
```

## What You'll Learn

- Loading a pre-trained tokenizer from vocabulary and merge files
- Building a byte-level BPE pipeline with `from_model_file`
- Encoding text and inspecting tokens, IDs, and offsets
- Decoding token IDs back to text
- Subword splitting on real vocabulary
- Batch encoding multiple texts

## Key Functions

| Function                   | Purpose                                         |
| -------------------------- | ----------------------------------------------- |
| `Brot.from_model_file`     | Load tokenizer from vocab.json and merges.txt   |
| `Pre_tokenizer.byte_level` | GPT-2 style byte-level pre-tokenizer            |
| `Decoder.byte_level`       | Corresponding byte-level decoder                |
| `Brot.encode`              | Encode text to an `Encoding.t`                  |
| `Brot.decode`              | Decode token IDs back to text                   |
| `Brot.encode_batch`        | Encode multiple texts at once                   |
| `Encoding.tokens`          | Token strings from an encoding                  |
| `Encoding.ids`             | Token IDs from an encoding                      |
| `Encoding.offsets`         | Byte offset pairs mapping tokens to source text |

## Prerequisites

This example downloads GPT-2 model files from HuggingFace on first run
(~1 MB total). Files are cached in `/tmp/brot_gpt2/`.

## Output Walkthrough

```
Vocabulary: 50257 tokens

Text:    "Hello world! GPT-2 is amazing."
Tokens:  ["Hello"; " world"; "!"; " GPT"; "-"; "2"; " is"; " amazing"; "."]
IDs:     [15496; 995; 0; 402; 12; 17; 318; 4998; 13]
Decoded: "Hello world! GPT-2 is amazing."
Round-trip: true

=== Subword Splitting ===

  "tokenization"       -> 3 tokens: ["token", "ization"]
  "transformer"        -> 1 tokens: ["transformer"]
  ...

=== Batch Encoding ===

  "The quick brown fox"          -> 4 tokens
  "jumps over the lazy dog"      -> 5 tokens
  "Machine learning is fun"      -> 4 tokens

=== Token Offsets ===

  Text: "Hello, world!"
  Hello     offsets=(0, 5)  source="Hello"
  ,         offsets=(5, 6)  source=","
  ...
```

## Try It

1. Change the input text and see how GPT-2 tokenizes different sentences.
2. Try words with unusual spellings to see subword splitting in action.
3. Compare the token count for English text vs other languages.

## See Also

- [01-encode-decode](../01-encode-decode/) for basic encoding and decoding
- [05-algorithms](../05-algorithms/) for comparing tokenization algorithms
- [08-decoders](../08-decoders/) for decoder options
