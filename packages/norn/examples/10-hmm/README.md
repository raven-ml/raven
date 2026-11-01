# `10-hmm`

A hidden Markov model's states are discrete, so a gradient sampler cannot
move them. `Dist.hmm` sums them out with the forward algorithm, which leaves
a density over the continuous parameters that NUTS samples. This example
fits a series that switches between two regimes, then reads each step's
regime from `Dist.membership`.

```bash
dune exec packages/norn/examples/10-hmm/main.exe
```

## What You'll Learn

- Observing a sequence through `Dist.hmm ~initial ~transition n d`
- Building transition logits from sampled stay probabilities
- Keeping regimes labelled with `Dist.sorted`
- Each step's state's posterior with `Dist.membership`

## Key Functions

| Function                              | Purpose                                 |
| ------------------------------------- | --------------------------------------- |
| `Dist.hmm ~initial ~transition n d`   | `n` observations, the states summed out |
| `Dist.membership d y`                 | Each step's state's posterior           |
| `Dist.sorted n d`                     | Ordered draws, for labelled components  |
