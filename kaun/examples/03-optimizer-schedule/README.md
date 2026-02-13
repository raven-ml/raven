# `03-optimizer-schedule`

Compares different learning-rate schedules by fitting a 1D linear regression
with Adam. Each schedule runs from the same initial parameters, logs the
learning rate every 40 steps, and prints the fitted weights at the end.

- **Source**: [`main.ml`](./main.ml)

```bash
cd kaun/examples/03-optimizer-schedule
dune exec kaun/examples/03-optimizer-schedule/main.exe
```

### Experiments

The script evaluates three schedules:

- `constant` – fixed step size.
- `exp_decay` – exponential decay with rate 0.6.
- `cosine` – cosine decay with a 10% floor.

Each run reports the learning-rate samples and the recovered line `y ≈ wx + b`
so you can see how quickly the optimiser converges under different schedules.

[Back to the examples index](../#readme)
