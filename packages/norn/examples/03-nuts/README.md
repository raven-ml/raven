# `03-nuts`

A density over a record of your own, written by hand, sampled with NUTS over
four chains, then diagnosed and summarised.

```bash
dune exec packages/norn/examples/03-nuts/main.exe
```

## What You'll Learn

- Making a record a structure with `Nx.Ptree.instantiate`
- Writing a density over chains with `Dist.factors`
- `Nuts.init`, `warmup` and `sample`
- Draws as values of your record, and `Diag.rhat` per field
- `Summary.v` and its table

## Key Functions

| Function                      | Purpose                      |
| ----------------------------- | ---------------------------- |
| `Nuts.init u lp start`        | Chains at `start`            |
| `Nuts.warmup u lp k ~steps s` | Tune step sizes and geometry |
| `Nuts.sample u lp k ~draws s` | Draws and statistics         |
| `Diag.rhat u d`               | R-hat per element            |
| `Summary.v u ~stats d`        | A table with findings        |

## Next Steps

Continue to [04-models](../04-models/).
