# `01-encode-decode`

Your first tokenizer. This example shows the minimal steps to encode text into
token IDs and decode back.

```bash
dune exec brot/examples/01-encode-decode/main.exe
```

## What You'll Learn

- Creating a BPE tokenizer with `Brot.bpe`
- Encoding text with `Brot.encode`
- Inspecting token strings and IDs with `Encoding.tokens` and `Encoding.ids`
- Decoding token IDs back to text with `Brot.decode`

## Key Functions

| Function          | Purpose                                                |
| ----------------- | ------------------------------------------------------ |
| `bpe`             | Create a BPE tokenizer from vocabulary and merge rules |
| `encode`          | Encode text into an `Encoding.t`                       |
| `Encoding.ids`    | Get the integer token IDs                              |
| `Encoding.tokens` | Get the string token representations                   |
| `decode`          | Convert token IDs back to text                         |

## How BPE Works

BPE (Byte Pair Encoding) iteratively merges the most frequent character pairs.
Given the text `"hello"` and merge rules like `("h","e")`, `("l","l")`,
`("he","l")`, `("ll","o")`, `("hel","lo")`, BPE applies merges in priority
order until no more merges apply, producing `"hello"` as a single token.

## Try It

1. Remove some merge rules and run again to see how the text gets split into
   smaller subword pieces.
2. Add a new word like `"held"` to the vocabulary and encode `"hello held"`.

## Next Steps

Continue to [02-encoding-fields](../02-encoding-fields/) to learn about all the
metadata in an encoding.
