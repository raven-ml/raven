# `06-special-tokens`

Special tokens and post-processing. Post-processors insert tokens like `[CLS]`
and `[SEP]` after tokenization, and assign type IDs for sentence-pair tasks.

```bash
dune exec brot/examples/06-special-tokens/main.exe
```

## What You'll Learn

- Defining special tokens with `Brot.special`
- BERT-style post-processing: `[CLS] A [SEP]` and `[CLS] A [SEP] B [SEP]`
- Sentence-pair encoding with `encode ~pair`
- Type IDs: 0 for first sequence, 1 for second
- Template-based post-processing for custom formats
- Skipping special tokens with `~add_special_tokens:false`

## Key Functions

| Function                       | Purpose                                     |
| ------------------------------ | ------------------------------------------- |
| `Brot.special`                 | Define a special token configuration        |
| `Post_processor.bert`          | BERT-style `[CLS] A [SEP] B [SEP]`          |
| `Post_processor.template`      | Template-based with `$A`, `$B` placeholders |
| `Brot.encode ~pair`            | Encode a sentence pair                      |
| `Encoding.type_ids`            | Segment type IDs (0 or 1)                   |
| `Encoding.special_tokens_mask` | 1 for special tokens, 0 for content         |

## BERT Post-processing

For a single sentence: `[CLS] tokens [SEP]`
For a sentence pair: `[CLS] A_tokens [SEP] B_tokens [SEP]`

Type IDs distinguish the two sequences:
- First sequence (including `[CLS]` and first `[SEP]`): type_id = 0
- Second sequence (including final `[SEP]`): type_id = 1

## Try It

1. Try the `roberta` post-processor with `<s>` and `</s>` tokens.
2. Create a custom template with different special tokens.
3. Encode a pair and check that `type_ids` correctly separates the segments.

## Next Steps

Continue to [07-padding-truncation](../07-padding-truncation/) to learn about
preparing batches with uniform sequence lengths.
