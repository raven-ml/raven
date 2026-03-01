# agents.md

raven is an ecosystem of packages that brings modern machine learning capabilities to ocaml. it provides familiar equivalent of python packages.

## philosophy

raven is inspired by unix's philosophy of doing one thing well, and tinygrad's philosophy of minimalism and clarity. while our scope is larger than tinygrad's, we aim for the same beautiful and minimal code that covers python equivalent use cases.

- strive for the "right", principled implementations and designs that stand the test of time.
- every line must have purpose. choose clarity over cleverness.
- public apis stay small and modern. no legacy layers, no extra knobs.
- do not maintain compatibility for its own sake. breaking changes are fine when they move us toward the correct design.
- focus on _modern_ numerical computing and machine learning. old or classic apis from numpy, pandas, jax, etc are out of scope.
- minimize api surface as much as possible and offer the most elegant apis that cover user needs.

## projects

- **nx**: n-dimensional arrays with pluggable backend architecture - equivalent to numpy.

  the backend interface is defined at `packages/nx/lib/core/backend.mli`. NEVER add a backend operation without being asked to do so.
  frontend apis are defined in a single file `packages/nx/lib/frontend.ml` using the backend operations.
  nx comes with a default c backend in `packages/nx/lib/backend_c/`.

- **rune**: tensor computation with automatic differentiation and jit compilation - equivalent to jax.

  rune is architected as a backend for nx in `packages/rune/lib/nx_rune.ml`, where each backend operation raises an effect, or, if the effect is unhandled, falls back to the nx c backend.

  this allows us to provide an nx-like api, while providing additional features such as automatic differentiation and jit compilation:
  - for automatic differentiation in `packages/rune/lib/autodiff.ml`, effects are caught once re-executed, alongside their gradient calculations in the effect handler, the new calls are not caught by the effect handler (unless the user nests `grad` calls), so the operations are executed as normal on the c backend.
  - for jit compilation, all effects are handled to build a computation graph, which is then jitted using `rune.jit`.
  - and similar for other features such as debug, vmap.

- **kaun**: neural networks and training utilities built on rune - equivalent to flax.

  kaun builds on rune to provide high-level neural network abstractions such as ptree, layers, optimizer, training loops, datasets, metrics, etc.

  it also provides ready-to-use models in `packages/kaun/lib/kaun-models` and datasets in `packages/kaun/lib/kaun-datasets`.

- **fehu**: reinforcement learning environment and algorithms built on rune and kaun - equivalent to gym and stable baselines.
- **talon**: dataframe library for data manipulation and analysis - equivalent to pandas and polars.
- **brot**: tokenization and text processing - equivalent to huggingface tokenizers and parts of huggingface transformers.
- **hugin**: visualization library for plotting and rendering - equivalent to matplotlib and plotly.
- **quill**: interactive computing environment for ocaml - equivalent to jupyter notebooks and ipython.
- **sowilo**: image processing and computer vision built on rune - equivalent to opencv with differentiable operations.

## project structure

- packages live in `packages/` such as `packages/nx/`, `packages/rune/`, `packages/kaun/`, `packages/sowilo/`, `packages/talon/`, `packages/hugin/`, `packages/quill/`, and `packages/fehu/`, each with `lib/` sources and `test/` suites.
- documentation assets live under `www/` (static site).

## guidelines

- modules and variants are `Capitalized_snake_case`. values and functions use `snake_case`.
- docstrings are only used in `mli` files. they start with `(** [function_name args...] ... *)`.
- operations that match on dtypes need explicit type annocations, e.g. `let nonzero (type a b) (t : (a, b) t) =`.

## performance

- keep allocations to a minimum. allocate outside of loops and reuse buffers when possible.
- prefer loop-based implementations over higher-order functions for performance-critical code.
- use unsafe Bigarray and Bytes functions (e.g. `Bigarray.Array1.unsafe_get`) when safety checks are redundant.

## important rules

- NEVER stage or commit changes unless explicitly requested
- NEVER run `dune clean`
- NEVER use the `--force` argument
- NEVER run dune build with DUNE_CACHE=disabled
- NEVER try to remove the dune lock file
- NEVER git checkout or reset any file unless explicitly requested
- NEVER add new backend operations to nx unless explicitly requested
- NEVER hide warnings and NEVER hide unused variables by adding an underscore. ALWAYS treat warnings as errors that need a proper fix.
