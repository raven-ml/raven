# `10-bert-pipeline`

Complete BERT-style tokenizer pipeline. Assembles all stages: normalizer,
pre-tokenizer, WordPiece algorithm, post-processor, decoder, special tokens,
padding, and truncation.

```bash
dune exec brot/examples/10-bert-pipeline/main.exe
```

## What You'll Learn

- Assembling a full tokenization pipeline
- How all stages work together end-to-end
- Single sentence and sentence-pair encoding
- Batch encoding with padding
- Sentence-pair batch encoding with `encode_pairs_batch`
- Decoding with and without special tokens
- Inspecting tokenizer configuration with `Brot.pp`

## Key Functions

| Function                           | Purpose                                       |
| ---------------------------------- | --------------------------------------------- |
| `Brot.wordpiece`                   | Full pipeline constructor                     |
| `Normalizer.bert`                  | BERT normalizer (lowercase, clean, CJK)       |
| `Pre_tokenizer.bert`               | BERT pre-tokenizer (whitespace + punctuation) |
| `Post_processor.bert`              | Insert `[CLS]` and `[SEP]` tokens             |
| `Decoder.wordpiece`                | Reverse `##` prefix joining                   |
| `Brot.encode ~pair`                | Encode a sentence pair                        |
| `Brot.encode_pairs_batch`          | Batch-encode sentence pairs                   |
| `Brot.decode ~skip_special_tokens` | Decode without `[CLS]`/`[SEP]`                |
| `Brot.pp`                          | Pretty-print tokenizer configuration          |

## The Full Pipeline

```
Input text
  |
  v
Normalizer.bert     -- lowercase, clean control chars, pad CJK
  |
  v
Pre_tokenizer.bert  -- split on whitespace, isolate punctuation
  |
  v
WordPiece model     -- greedy longest-match subword splitting
  |
  v
Post_processor.bert -- insert [CLS] and [SEP], set type_ids
  |
  v
Encoding.t          -- ids, tokens, offsets, type_ids, attention_mask
```

## Try It

1. Encode text with accented characters and see the normalizer at work.
2. Change `Post_processor.bert` to `Post_processor.roberta` with `<s>` and
   `</s>` tokens for a RoBERTa-style pipeline.
3. Use `save_pretrained` to export the tokenizer and reload it with
   `from_file`.

## Further Reading

- [gpt2_tokenizer](../gpt2_tokenizer/) -- loading a real GPT-2 tokenizer
  from HuggingFace model files
