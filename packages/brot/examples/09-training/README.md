# `09-training`

Training tokenizers from scratch. Given a text corpus, each algorithm learns a
vocabulary tailored to the data.

```bash
dune exec brot/examples/09-training/main.exe
```

## What You'll Learn

- Training BPE, WordPiece, word-level, and Unigram tokenizers
- Controlling vocabulary size with `~vocab_size`
- Adding special tokens during training
- Inspecting the learned vocabulary

## Key Functions

| Function               | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `Brot.train_bpe`       | Train a BPE tokenizer (learns merge rules)       |
| `Brot.train_wordpiece` | Train a WordPiece tokenizer (learns subwords)    |
| `Brot.train_wordlevel` | Train a word-level tokenizer (collects words)    |
| `Brot.train_unigram`   | Train a Unigram tokenizer (learns probabilities) |
| `Brot.vocab_size`      | Check learned vocabulary size                    |
| `Brot.token_to_id`     | Look up a token's ID                             |

## Training Data

Training data is provided as `` `Seq (List.to_seq texts) `` for in-memory text
or `` `Files ["path1"; "path2"] `` for files (one sentence per line).

## Try It

1. Add more sentences to the corpus and see how the vocabulary changes.
2. Train with a smaller `~vocab_size` and observe more subword splitting.
3. Use `~min_frequency:2` to exclude rare words.

## Next Steps

Continue to [10-bert-pipeline](../10-bert-pipeline/) to assemble a complete
BERT-style tokenizer pipeline.
