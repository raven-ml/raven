# `08-decoders`

Decoders convert token strings back to natural text. Different tokenization
schemes require different decoding strategies to produce clean output.

```bash
dune exec brot/examples/08-decoders/main.exe
```

## What You'll Learn

- Per-token decoders: `wordpiece`, `bpe`, `metaspace`, `byte_fallback`
- Collapsing decoders: `fuse`, `replace`
- Composing decoders with `sequence`
- Integrating a decoder with a tokenizer
- Skipping special tokens during decoding

## Key Functions

| Function                | Purpose                              |
| ----------------------- | ------------------------------------ |
| `Decoder.wordpiece`     | Strip `##` prefix, join subwords     |
| `Decoder.bpe`           | Strip word-end suffix, insert spaces |
| `Decoder.metaspace`     | Convert markers back to spaces       |
| `Decoder.byte_fallback` | Convert `<0xFF>` back to bytes       |
| `Decoder.fuse`          | Concatenate all tokens               |
| `Decoder.replace`       | String replacement                   |
| `Decoder.sequence`      | Chain decoders                       |
| `Decoder.decode`        | Apply decoder to token list          |
| `Brot.decode`           | Full decode through tokenizer        |

## Per-token vs Collapsing

Some decoders transform each token independently (per-token: `bpe`,
`metaspace`, `byte_fallback`), while others combine the entire token list into
a single result (collapsing: `wordpiece`, `fuse`, `replace`). This matters
when composing with `sequence`.

## Try It

1. Try `Decoder.ctc` for speech recognition CTC output.
2. Compose `byte_fallback` with `fuse` and decode byte tokens.
3. Use `Decoder.strip` to remove leading/trailing characters.

## Next Steps

Continue to [09-training](../09-training/) to learn how to train tokenizers
from scratch.
