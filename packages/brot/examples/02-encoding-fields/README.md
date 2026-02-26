# `02-encoding-fields`

Understanding encodings. An `Encoding.t` bundles token IDs with alignment
metadata: byte offsets, word indices, type IDs, attention masks, and
special-token flags.

```bash
dune exec brot/examples/02-encoding-fields/main.exe
```

## What You'll Learn

- All parallel arrays in an `Encoding.t` and how they align
- Byte offsets that map each token back to the original text
- Word indices that group subword tokens by source word
- Attention mask (1 = real token, 0 = padding)
- Special tokens mask (1 = special, 0 = content)

## Key Functions

| Function                       | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `Encoding.ids`                 | Token ID array for model input                    |
| `Encoding.tokens`              | String representation of each token               |
| `Encoding.offsets`             | `(start, end)` byte spans in the original text    |
| `Encoding.word_ids`            | Source word index per token (`None` for specials) |
| `Encoding.type_ids`            | Segment IDs (0 or 1 for sentence pairs)           |
| `Encoding.attention_mask`      | 1 for real tokens, 0 for padding                  |
| `Encoding.special_tokens_mask` | 1 for special tokens, 0 for content               |
| `Encoding.length`              | Number of tokens                                  |

## Offsets

Offsets are byte positions `(start, end)` into the original text. You can
extract the original substring with `String.sub text start (end - start)`.
This is essential for highlighting, named entity recognition, and other tasks
that need to map tokens back to source text.

## Try It

1. Add more words to the vocabulary and encode a longer sentence.
2. Encode a text with unknown words and observe the `[UNK]` token.

## Next Steps

Continue to [03-normalizers](../03-normalizers/) to learn how text is cleaned
before tokenization.
