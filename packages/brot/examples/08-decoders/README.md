# `08-decoders`

Decoders convert token strings back to natural text. Different tokenization
schemes require different decoding strategies to produce clean output.

```bash
dune exec brot/examples/08-decoders/main.exe
```

## What You'll Learn

- Per-token decoders: `wordpiece`, `bpe`, `metaspace`, `replace`
- Joining and collapsing decoders: `byte_fallback`, `fuse`
- Composing decoders with `sequence`
- Integrating a decoder with a tokenizer
- Skipping special tokens during decoding

## Key Functions

| Function                | Purpose                                       |
| ----------------------- | --------------------------------------------- |
| `Decoder.wordpiece`     | Strip `##` prefix, space out the other subwords |
| `Decoder.bpe`           | Turn the word-end suffix into the space it marks |
| `Decoder.metaspace`     | Convert markers back to spaces                |
| `Decoder.byte_fallback` | Convert runs of `<0xFF>` back to text         |
| `Decoder.fuse`          | Concatenate all tokens                        |
| `Decoder.replace`       | String replacement in every token             |
| `Decoder.sequence`      | Chain decoders                                |
| `Decoder.decode`        | Apply decoder to token list                   |
| `Brot.decode`           | Full decode through tokenizer                 |

## Per-token vs Collapsing

Every decoder rewrites a token list into another token list, and `decode` is
the concatenation of the result. Most rewrite each token on its own (`bpe`,
`metaspace`, `wordpiece`, `replace`, `strip`); `byte_fallback` and `ctc` also
join or drop tokens; `byte_level` and `fuse` collapse the list into one token
and so hide token boundaries from whatever follows them in a `sequence`.

## Try It

1. Try `Decoder.ctc` for speech recognition CTC output.
2. Compose `byte_fallback` with `fuse` and decode byte tokens.
3. Use `Decoder.strip` to remove leading/trailing characters.

## Next Steps

Continue to [09-training](../09-training/) to learn how to train tokenizers
from scratch.
