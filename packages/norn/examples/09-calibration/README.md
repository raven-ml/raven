# `09-calibration`

Ranks of prior truths among posterior draws are uniform when the posterior is
right. This example tests that with exact posterior draws and with posteriors
that are too narrow or too wide.

```bash
dune exec packages/norn/examples/09-calibration/main.exe
```

## What You'll Learn

- Simulating truths and data with `Norn_model.simulate`
- `Diag.rank` of a truth among draws
- `Diag.rank_uniformity`, an exact test

## Key Functions

| Function                              | Purpose                      |
| ------------------------------------- | ---------------------------- |
| `Diag.rank u ~truth d`                | The rank of a truth          |
| `Diag.rank_uniformity u ~draws ranks` | The p-value of uniform ranks |
