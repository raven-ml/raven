# `05-algorithms`

Five tokenization algorithms compared side-by-side. Each algorithm splits text
differently based on its strategy.

```bash
dune exec brot/examples/05-algorithms/main.exe
```

## What You'll Learn

- **BPE** (Byte Pair Encoding): merge-based subwords (GPT-2, RoBERTa)
- **WordPiece**: greedy longest-match with `##` prefix (BERT)
- **Unigram**: probabilistic segmentation (T5, mBART)
- **Word-level**: one token per word, no subword splitting
- **Character-level**: one token per byte, no vocabulary needed

## Key Functions

| Function          | Purpose                                |
| ----------------- | -------------------------------------- |
| `Brot.bpe`        | BPE tokenizer from vocab + merge rules |
| `Brot.wordpiece`  | WordPiece tokenizer from vocab         |
| `Brot.unigram`    | Unigram tokenizer from vocab + scores  |
| `Brot.word_level` | Word-level tokenizer from vocab        |
| `Brot.chars`      | Character-level tokenizer (no vocab)   |
| `Brot.vocab_size` | Number of vocabulary entries           |

## Algorithm Comparison

| Algorithm  | Subwords?           | Unknown handling         | Vocabulary              |
| ---------- | ------------------- | ------------------------ | ----------------------- |
| BPE        | Yes (merges)        | Falls back to characters | `(string * int) list`   |
| WordPiece  | Yes (`##` prefix)   | `[UNK]` token            | `(string * int) list`   |
| Unigram    | Yes (probabilistic) | Lowest-score fallback    | `(string * float) list` |
| Word-level | No                  | `<unk>` token            | `(string * int) list`   |
| Chars      | No                  | N/A (all bytes valid)    | None needed             |

## Try It

1. Add more merge rules to the BPE tokenizer and see how it affects splitting.
2. Try encoding a word not in the WordPiece vocabulary.
3. Change the Unigram scores and observe how probabilities affect splitting.

## Next Steps

Continue to [06-special-tokens](../06-special-tokens/) to learn about special
tokens and post-processing.
