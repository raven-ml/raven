# `04-models`

`norn.model` turns one function that draws every random variable into densities,
starting points, simulations and predictions. This example fits a line to data
simulated from the same model.

```bash
dune exec packages/norn/examples/04-models/main.exe
```

## What You'll Learn

- Writing a model with `Norn_model.v` and `sample`
- The site table, `Norn_model.pp`
- Fixing sites with `fix` and simulating data
- Fitting in coordinates with `log_density` and `init`
- Mapping draws back with `Draws.map` and `constrain`
- Predicting with `predict`

## Key Functions

| Function                                 | Purpose                        |
| ---------------------------------------- | ------------------------------ |
| `Norn_model.v dtype latent observed gen` | A model                        |
| `Norn_model.sample d`                    | A random variable              |
| `Norn_model.log_density m y`             | The posterior over coordinates |
| `Norn_model.init m y ~chains k`          | Finite starting points         |
| `Norn_model.constrain m c`               | Values of coordinates          |
| `Norn_model.simulate`, `predict`         | Draw data                      |

## Next Steps

Continue to [05-hmc-and-ensemble](../05-hmc-and-ensemble/).
