# Kaun Examples

These numbered folders form a lightweight tour of Kaun. Start with the
shallow end (manual training loops), then climb toward large pretrained
transformers. Each example README explains what to expect, how to run it, and
where to look next.

| Example                 | What it shows                                                            |
| ----------------------- | ------------------------------------------------------------------------ |
| `01-xor`                | Minimal supervised loop that trains a two-layer perceptron on XOR        |
| `02-xor-eval`           | Extends XOR with reusable predict/accuracy helpers                       |
| `03-optimizer-schedule` | Comparing learning-rate schedules with Adam                              |
| `04-dataset-stream`     | Building infinite pipelines with `Kaun.Dataset` (shuffle/batch/prefetch) |
| `05-checkpointing`      | Saving and restoring params/optimiser state with `Kaun.Checkpoint`       |
| `m0-mnist-basics`       | MNIST CNN with explicit batching, gradients, and metrics                 |
| `m1-mnist-fit`          | The same MNIST CNN driven by `Training.fit` and dataset helpers          |
| `m2-bert`               | BERT embedding inspections (sentence similarity, word-in-context)        |
| `m3-gpt2`               | GPT-2 forward inspection and sampling CLI                                |
