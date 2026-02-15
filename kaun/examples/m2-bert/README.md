# `m2-bert`

This example shows how to load a pretrained BERT encoder with Kaun, extract
vector representations, and run lightweight probes on top of them.

- **Source**: [`main.ml`](./main.ml)
- **Downloads**: Weights and vocabulary files are cached on first run

```bash
cd kaun/examples/m2-bert
dune exec kaun/examples/m2-bert/main.exe
```

### Program structure

`bert.ml` performs three steps:

1. Validate a tiny configuration block (`model_id`, `max_length`,
   `similarity_threshold`).
2. Instantiate the encoder and tokenizer via `Kaun_models.Bert.from_pretrained`
   and `Bert.Tokenizer.create`.
3. Run two probes:
   - **Sentence similarity** – pool CLS and mean embeddings, compute cosine
     scores for a few sentence pairs, and compare them against the configured
     threshold.
   - **Word in context** – locate the same surface word in two sentences,
     extract its contextual embedding in each sentence, and report how far they
     drift.

### Sample output

```
Configuration:
  model_id              : bert-base-uncased
  max_length            : 96
 similarity_threshold  : 0.80

=== Sentence Similarity ===
1. "The cat curls up on the sofa"
   "A feline naps on the couch"
   cosine(CLS) : 0.88
   cosine(mean): 0.87 → Similar (expected Similar)

=== Word-in-Context (Polysemy) ===
Word    Context A                                | Context B                                | Cosine
--------------------------------------------------------------------------------------------------------
bank    The bank raised interest rates           | We sat on the bank of the river          | 0.41 ✓
         token positions: 2 vs 5 (max 96)
```

### Experiments

- Adjust `Config.similarity_threshold` to see how sensitive the decision rule
  is to the chosen cutoff.
- Try different pooling strategies in `sentence_embeddings` (e.g. max pooling)
  and observe how the ranking changes.
- Replace the hard-coded sentences with your own pairs to test domain-specific
  wording.

### Next steps

- Continue to [**`m3-gpt2`**](../m3-gpt2#readme) to generate text with GPT-2.
- Revisit [**`m1-mnist-fit`**](../m1-mnist-fit#readme) if you want to relate
  pretrained inference to supervised training workloads.

[Back to the examples index](../#readme)
