# `m2-bert`

This example exercises Kaun’s BERT integration. It loads `bert-base-uncased`
from Hugging Face, tokenizes input text, and evaluates a few inspection tasks to
illustrate how to work with contextual embeddings.

- **Source**: [`bert.ml`](./bert.ml)
- **Downloads**: Weights and vocabulary files are cached on first run

```bash
cd kaun/example/m2-bert
dune exec kaun/example/m2-bert/bert.exe
```

### Program structure

`bert.ml` performs three steps:

1. Validate the small configuration block (`model_id`, `max_length`,
   `similarity_threshold`).
2. Instantiate the model and tokenizer via `Kaun_models.Bert.from_pretrained` and
   `Bert.Tokenizer.create`.
3. Run two analysis routines:
   - **Sentence similarity** – encodes pairs of sentences, extracts CLS and
     mean-pooled embeddings, computes cosine similarity, and reports decisions
     relative to the configured threshold.
   - **Word-in-context** – encodes sentences containing the same surface word in
     different contexts, selects the corresponding token embeddings, and measures
     how far the representations diverge.
   - A Hugging Face regression test now lives under
     `kaun/test/bert/test_bert_matches_hf.ml`.

### Sample output

```
Configuration:
  Model: bert-base-uncased
  Max length: 128
  Similarity threshold: 0.80

=== Sentence Similarity Task ===
1. Sentence 1: "The cat sits on the mat"
   Sentence 2: "A feline rests on the rug"
   Similarity (CLS): 0.8321
   Similarity (Mean): 0.8576
   Expected: Similar | Predicted: Similar | ✓

=== Word-in-Context (Polysemy Verification) ===
plant    | The chemical plant employs hundreds
         | She watered the plant in her garden
         | CosineSim: 0.3625 | ✓
```

### Experiments

- Adjust `Config.similarity_threshold` to see how sensitive the decision rule
  is to the chosen cutoff.
- Switch the pooling strategy in `extract_sentence_embedding` (CLS, mean, max)
  and observe how it alters the similarity rankings.
- Use `Bert.Tokenizer.encode_batch` to process multiple sentence pairs at once;
  this highlights Kaun’s handling of batched inputs.

### Next steps

- Continue to [**`m3-gpt2`**](../m3-gpt2#readme) to generate text with GPT-2.
- Revisit [**`m1-mnist-fit`**](../m1-mnist-fit#readme) if you want to relate
  pretrained inference to supervised training workloads.

[Back to the examples index](../#readme)
