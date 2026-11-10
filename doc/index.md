# Documentation

Welcome to Raven's documentation. Raven is an ecosystem of OCaml libraries for numerical computing, machine learning, and data science.

## Start Here

- **[Quickstart](manual/quickstart.md)** — zero to gradient in 5 minutes
- **[Coming from Python](manual/coming-from-python.md)** — map NumPy, PyTorch, pandas concepts to Raven
- **[Ecosystem Overview](manual/ecosystem-overview.md)** — how the libraries relate and which to use

## Libraries

| | Library | Like | What it does |
|-|---------|------|-------------|
| | [**rig**](../packages/rig/doc/index.md) | CUDA driver API | Devices such as GPUs, and the order of the work they run |
| | [**nx**](../packages/nx/doc/index.md) | NumPy | N-dimensional arrays on the host and on devices |
| ᛏ | [**tolk**](../packages/tolk/doc/index.md) | tinygrad | Minimal ML compiler for GPU tensor computation |
| ᚱ | [**rune**](../packages/rune/doc/index.md) | JAX | Automatic differentiation and functional transformations |
| ᚴ | [**kaun**](../packages/kaun/doc/index.md) | PyTorch / Flax | Neural networks and training |
| ᚹ | [**vega**](../packages/vega/doc/index.md) | Optax | Gradient-based optimizers and schedules |
| ᛄ | [**jera**](../packages/jera/doc/index.md) | SciPy | Numerical methods: systems, minima, integrals, interpolation, differential equations |
| ᚾ | [**norn**](../packages/norn/doc/index.md) | NumPyro / Stan | Probabilistic inference: models, samplers, diagnostics |
| ᚨ | [**brot**](../packages/brot/doc/index.md) | HF Tokenizers | Fast tokenization for language models |
| ᛃ | [**talon**](../packages/talon/doc/index.md) | pandas / Polars | Dataframes with typed queries, CSV and Parquet |
| ᛞ | [**hugin**](../packages/hugin/doc/index.md) | Matplotlib | Data visualization and plotting |
| ᛈ | [**quill**](../packages/quill/doc/index.md) | Jupyter + IPython | Interactive REPL and markdown notebooks |
| ᛗ | [**munin**](../packages/munin/doc/index.md) | W&B / MLflow | Local experiment tracking with live TUI dashboard |

Packages that release on their own schedule, such as Fehu, Sowilo and Ymir, live in the repository's [`contrib/`](https://github.com/raven-ml/raven/tree/main/contrib) directory.

## Project

- [Installation](manual/installation.md) — system dependencies, opam setup, building from source
- [Roadmap](roadmap.md) — what works today and what's coming
- [Updates](blog/index.md) — progress reports on Raven's development
- [Introduction](manual/introduction.md) — vision and philosophy
- [Support Raven](support-raven.md) — sponsorship and contributing
