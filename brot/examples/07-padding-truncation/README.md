# `07-padding-truncation`

Padding and truncation for batch processing. Models require uniform sequence
lengths. Padding adds filler tokens; truncation trims long sequences.

```bash
dune exec brot/examples/07-padding-truncation/main.exe
```

## What You'll Learn

- Fixed-length padding with `padding (`Fixed n)`
- Batch-longest padding with `padding `Batch_longest`
- Left vs right padding direction
- Truncation with `truncation max_length`
- Combining padding and truncation
- Using `Encoding.attention_mask` to distinguish real tokens from padding

## Key Functions

| Function                  | Purpose                           |
| ------------------------- | --------------------------------- |
| `Brot.padding`            | Create a padding configuration    |
| `Brot.truncation`         | Create a truncation configuration |
| `Brot.encode_batch`       | Encode multiple texts at once     |
| `Encoding.attention_mask` | 1 for real tokens, 0 for padding  |

## Padding Strategies

| Strategy             | Behavior                                               |
| -------------------- | ------------------------------------------------------ |
| `` `Fixed n ``       | Every sequence padded to exactly `n` tokens            |
| `` `Batch_longest `` | All sequences padded to match the longest in the batch |
| `` `To_multiple n `` | Pad to smallest multiple of `n` >= sequence length     |

## Try It

1. Change the padding direction to `` `Left `` and observe where pad tokens appear.
2. Try `padding (`To_multiple 4)` and see how lengths round up.
3. Truncate from the left with `truncation ~direction:`Left 3`.

## Next Steps

Continue to [08-decoders](../08-decoders/) to learn how tokens are converted
back to text.
