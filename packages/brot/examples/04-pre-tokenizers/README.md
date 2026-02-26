# `04-pre-tokenizers`

Pre-tokenization: splitting text into fragments before vocabulary lookup. Each
fragment carries byte offsets into the original text.

```bash
dune exec brot/examples/04-pre-tokenizers/main.exe
```

## What You'll Learn

- Common pre-tokenizers: `whitespace`, `whitespace_split`, `bert`
- Punctuation and digit handling
- Delimiter-based splitting: `char_delimiter`, `split`, `fixed_length`
- SentencePiece-style `metaspace`
- Composing pre-tokenizers with `sequence`
- Using `Pre_tokenizer.pre_tokenize` to see fragments and offsets

## Key Functions

| Function                         | Purpose                                    |
| -------------------------------- | ------------------------------------------ |
| `Pre_tokenizer.whitespace`       | Pattern-based: `\w+` and `[^\w\s]+` groups |
| `Pre_tokenizer.whitespace_split` | Simple whitespace splitting                |
| `Pre_tokenizer.bert`             | BERT-style: whitespace + punctuation + CJK |
| `Pre_tokenizer.punctuation`      | Isolate punctuation from words             |
| `Pre_tokenizer.digits`           | Split on digit boundaries                  |
| `Pre_tokenizer.char_delimiter`   | Split on a single character                |
| `Pre_tokenizer.split`            | Split on a literal string pattern          |
| `Pre_tokenizer.fixed_length`     | Fixed-length character chunks              |
| `Pre_tokenizer.metaspace`        | Replace spaces with visible markers        |
| `Pre_tokenizer.sequence`         | Chain pre-tokenizers left-to-right         |
| `Pre_tokenizer.pre_tokenize`     | Apply and get `(fragment, offsets)` list   |

## Pre-tokenizer vs Tokenizer

Pre-tokenization happens *before* the vocabulary-based algorithm (BPE,
WordPiece, etc.). It determines the boundaries within which subword splitting
operates. For example, with whitespace pre-tokenization, BPE will never merge
tokens across word boundaries.

## Try It

1. Try `unicode_scripts` on text mixing Latin and CJK characters.
2. Change the punctuation `behavior` to `` `Merged_with_previous `` or
   `` `Removed ``.
3. Create a pre-tokenizer that splits on hyphens.

## Next Steps

Continue to [05-algorithms](../05-algorithms/) to see how different
tokenization algorithms split the same text.
