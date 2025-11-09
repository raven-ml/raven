# `m3-gpt2`

This example demonstrates GPT-2 inference in Kaun. It shows how to load
`gpt2` weights and either inspect logits or sample continuations for a prompt.

- **Sources**: [`gpt2.ml`](./gpt2.ml), [`generate.ml`](./generate.ml)
- **Downloads**: GPT-2 weights and vocabulary are fetched from Hugging Face

```bash
cd kaun/example/m3-gpt2

# load parameters and dump tensor stats
dune exec kaun/example/m3-gpt2/gpt2.exe

# sample from the model (temperature 0.8, top-k 50 by default)
dune exec kaun/example/m3-gpt2/generate.exe -- "Once upon a time"
```

### Components

- `gpt2.ml` shows how to call `Kaun_models.Gpt2.from_pretrained`, inspect the
  parameter tree, and print logits for a fixed prompt.
- `generate.ml` tokenizes input text with `Saga` utilities, runs greedy or
  stochastic sampling under Kaun, and prints the continuation.

### Sample generation

```
Prompt: Once upon a time
---
Once upon a time in a small village, |
the people gathered around the storyteller |
to listen to tales of forgotten heroes...
```

### Next steps

- Jump back to [**`m2-bert`**](../m2-bert#readme) if you want to explore
  embedding-based analysis instead of generation.
- Continue into your own Kaun scripts by copying the sampling loop from
  `generate.ml`. A Hugging Face parity test now lives in
  `kaun/test/gpt2/test_gpt2_compare_with_python.ml`.

[Back to the examples index](../#readme)
