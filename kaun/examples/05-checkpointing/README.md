# `05-checkpointing`

Demonstrates how to persist and restore Kaun models with the built-in
checkpointing utilities. The script trains the XOR perceptron, writes a
checkpoint containing model parameters and optimizer state, simulates a crash
by reinitialising everything, then reloads the checkpoint and continues
training.

- **Source**: [`main.ml`](./main.ml)

```bash
cd kaun/examples/05-checkpointing
dune exec kaun/examples/05-checkpointing/main.exe
```

### What it shows

- Creating a checkpoint repository with `Kaun.Checkpoint.create_repository`.
- Serialising model parameters (`Checkpoint.Snapshot.ptree`) and optimizer
  state (`Optimizer.serialize`).
- Restoring both pieces (`Snapshot.to_ptree`, `Optimizer.restore`) to resume
  training exactly where it left off.

Checkpoints are written to `checkpointing_repo/` in the example directory so you
can inspect the manifest and artifacts after the run.

[Back to the examples index](../#readme)
