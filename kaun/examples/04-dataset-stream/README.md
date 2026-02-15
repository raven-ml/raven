# `04-dataset-stream`

Streams an infinite sequence of synthetic `(x, sin x)` pairs through the Kaun
`Dataset` pipeline. The example shows how to build a lazy pipeline from a
`Seq.t`, shuffle, batch, prefetch, and reset it for another pass.

- **Source**: [`main.ml`](./main.ml)

```bash
cd kaun/examples/04-dataset-stream
dune exec kaun/examples/04-dataset-stream/main.exe
```

### What you’ll see

- The dataset reports an `Infinite` cardinality.
- The first three mini-batches are materialised lazily, printing their shapes
  and the first sample in each batch.
- After `Dataset.reset` the pipeline emits another batch from the same stream
  without rebuilding it.

This pattern is useful when you want on-the-fly data generation (noise, data
augmentation, etc.) without ever loading a full dataset into memory.

[Back to the examples index](../#readme)
