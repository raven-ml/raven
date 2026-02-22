# `03-normalizers`

Text normalization before tokenization. Normalizers clean and standardize text
so that surface variations (case, accents, whitespace) don't prevent vocabulary
matches.

```bash
dune exec brot/examples/03-normalizers/main.exe
```

## What You'll Learn

- Unicode normalization: `nfc`, `nfkc`
- Text transforms: `lowercase`, `strip_accents`, `strip`, `replace`, `prepend`
- Model-specific normalization: `bert`
- Composing normalizers with `sequence`
- Applying normalizers directly with `Normalizer.apply`
- How normalization affects tokenization results

## Key Functions

| Function                   | Purpose                            |
| -------------------------- | ---------------------------------- |
| `Normalizer.nfc` / `nfkc`  | Unicode normalization forms        |
| `Normalizer.lowercase`     | Unicode case folding               |
| `Normalizer.strip_accents` | Remove combining marks             |
| `Normalizer.strip`         | Strip boundary whitespace          |
| `Normalizer.replace`       | Regex-based replacement            |
| `Normalizer.prepend`       | Prepend a string to non-empty text |
| `Normalizer.bert`          | BERT-specific normalizer           |
| `Normalizer.sequence`      | Compose normalizers left-to-right  |
| `Normalizer.apply`         | Apply a normalizer to a string     |

## Why Normalize?

Without normalization, `"Hello"`, `"hello"`, and `"HELLO"` are three different
tokens. Normalization maps them all to `"hello"` so a single vocabulary entry
covers all cases. Similarly, `"caf\u{00E9}"` and `"cafe"` can be unified by
stripping accents.

## Try It

1. Add `Normalizer.nfkd` and see how it differs from `nfd`.
2. Create a normalizer that replaces email addresses with `<EMAIL>`.
3. Try the BERT normalizer with Chinese characters.

## Next Steps

Continue to [04-pre-tokenizers](../04-pre-tokenizers/) to learn how text is
split into fragments before vocabulary lookup.
