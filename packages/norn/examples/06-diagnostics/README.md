# `06-diagnostics`

Chains that start in different modes of a bimodal target never meet. This
example shows R-hat catching it, and reads the summary's findings as data.

```bash
dune exec packages/norn/examples/06-diagnostics/main.exe
```

## What You'll Learn

- `Diag.rhat`, `Diag.ess_bulk` and `Diag.ebfmi`
- `Summary.findings` as values to match on
- `Summary.pp_finding`

## Key Functions

| Function                         | Purpose                 |
| -------------------------------- | ----------------------- |
| `Diag.rhat`, `ess_bulk`, `ebfmi` | Diagnostics             |
| `Summary.findings s`             | Findings as data        |
| `Summary.pp_finding`             | A finding as a sentence |

## Next Steps

Continue to [07-evidence](../07-evidence/).
