# The Raven Ecosystem

Raven's libraries share one data type: `Nx.t`, the
n-dimensional array. Each library does one thing, and they compose
through tensors.

## How the Libraries Fit Together

```
                         ┌───────────┐
                         │   Kaun    │  neural networks
                         │  (Flax)   │
                         └─────┬─────┘
                               │
                         ┌─────┴─────┐
                         │   Rune    │  autodiff, vmap
                         │  (JAX)    │
                         └─────┬─────┘
                               │
  ┌────────────────────────────┴────────────────────────────┐
  │                          Nx                              │
  │                       (NumPy)                            │
  └──┬──────────────┬──────────────┬──────────────┬─────────┘
     │              │              │              │
 ┌───┴────┐    ┌────┴───┐    ┌────┴───┐    ┌─────┴────┐
 │ Talon  │    │  Brot  │    │ Hugin  │    │  Quill   │
 │(Polars)│    │(HF Tok)│    │(Mpl)   │    │(Jupyter) │
 └────────┘    └────────┘    └────────┘    └──────────┘
```

**Nx** is the foundation — every library operates on `Nx.t` tensors.

**Rune** adds functional transformations on top of Nx: `grad`, `jvp`,
`vmap`. Your Nx code becomes differentiable without changes.

**Kaun** builds on Rune to provide layers, losses, initializers, data
batching, metrics, checkpoints, and HuggingFace Hub integration. Models
are typed records you define; optimizers come from **Vega**.

**Talon**, **Brot**, **Hugin**, and **Quill** each use Nx directly for
their domain.

## Which Library Do I Need?

| I want to... | Use |
|---|---|
| Work with numerical arrays | [Nx](../packages/nx/doc/index.md) |
| Compute gradients | [Rune](../packages/rune/doc/index.md) |
| Train neural networks | [Kaun](../packages/kaun/doc/index.md) |
| Tokenize text for language models | [Brot](../packages/brot/doc/index.md) |
| Manipulate tabular data | [Talon](../packages/talon/doc/index.md) |
| Create plots and visualizations | [Hugin](../packages/hugin/doc/index.md) |
| Run code interactively (REPL or notebooks) | [Quill](../packages/quill/doc/index.md) |

---

## Nx: N-Dimensional Arrays

Nx provides the numerical foundation for the entire ecosystem.
NumPy-like operations on n-dimensional arrays with 19 data types
(float16 through complex128), broadcasting, slicing, linear algebra,
FFT, and I/O.

```ocaml
open Nx

let x = linspace Float32 0. 10. 100
let y = sin x
let mean_y = mean y
```

[Nx documentation →](../packages/nx/doc/index.md)

## Rune: Automatic Differentiation

Functional transformations for Nx tensors: reverse-mode AD (grad,
vjp), forward-mode AD (jvp), and vectorising maps (vmap). Operates on
`Nx.t` values directly using OCaml 5 effect handlers — no special
tensor type needed. Transformations work over any typed parameter
structure through the `Nx.Ptree.S` interface; primed variants take a
single tensor.

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let f x = add (mul x x) (sin x)
let f' = grad' f
let f'' = grad' f'
```

[Rune documentation →](../packages/rune/doc/index.md)

## Kaun: Neural Networks

Building blocks for neural networks: layers as plain records with pure
apply functions, losses, initializers, data batching, metrics,
checkpoints, and HuggingFace Hub integration. A model is a typed record
you define — there is no layer object and no trainer. Training steps
compose `Rune.value_and_grad` with a Vega optimizer update.

<!-- $MDX skip -->
```ocaml
open Kaun

type 'a mlp = { l1 : 'a Linear.t; l2 : 'a Linear.t }

let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))

let params =
  { l1 = Linear.init ~inputs:784 ~outputs:128;
    l2 = Linear.init ~inputs:128 ~outputs:10 }
```

[Kaun documentation →](../packages/kaun/doc/index.md)

## Brot: Tokenization

Fast, HuggingFace-compatible tokenization supporting BPE, WordPiece,
Unigram, word-level, and character-level algorithms. Composable
pipeline (normalizer → pre-tokenizer → model → post-processor →
decoder) with training from scratch.

```ocaml
open Brot

let tokenizer = from_file "tokenizer.json" |> Result.get_ok
let encoding = encode tokenizer "Hello, world!"
let ids = Encoding.ids encoding
```

[Brot documentation →](../packages/brot/doc/index.md)

## Talon: DataFrames

Type-safe tabular data with heterogeneous columns, an applicative Row
system for row-wise operations, and vectorized aggregations backed by
Nx.

```ocaml
open Talon

let df = create [
  "name", Col.string_list ["Alice"; "Bob"; "Charlie"];
  "score", Col.float64_list [85.5; 92.0; 78.5];
]

let () = print df
```

[Talon documentation →](../packages/talon/doc/index.md)

## Hugin: Visualization

Publication-quality 2D and 3D plots using Cairo rendering. Takes Nx
tensors as input. Line plots, scatter, bar charts, contour plots,
image display.

<!-- $MDX skip -->
```ocaml
open Hugin
open Nx

let fig = figure () in
let ax = subplot fig in
let _ = Plotting.plot ax ~x ~y ~label:"sin(x)" in
show fig
```

[Hugin documentation →](../packages/hugin/doc/index.md)

## Quill: Interactive Computing

Interactive REPL and markdown notebooks. Launch `quill` for a toplevel
with syntax highlighting, completion, and history, or open a markdown
file for a full notebook experience. Terminal UI, web frontend, and
batch mode with all Raven libraries pre-loaded.

<!-- $MDX skip -->
```bash
quill                    # interactive REPL
quill notebook.md        # notebook TUI
quill serve notebook.md  # web frontend
quill run notebook.md    # batch evaluation
```

[Quill documentation →](../packages/quill/doc/index.md)

## Contrib Packages

The repository's [`contrib/`](https://github.com/raven-ml/raven/tree/main/contrib) directory holds packages built on the
core libraries that release on their own schedule and install separately
from `opam install raven`:

- [Norn](https://github.com/raven-ml/raven/tree/main/contrib/norn): MCMC sampling with automatic gradients
- [Fehu](https://github.com/raven-ml/raven/tree/main/contrib/fehu): reinforcement learning environments
- [Sowilo](https://github.com/raven-ml/raven/tree/main/contrib/sowilo): differentiable computer vision

## Getting Started

1. **New to Raven?** Start with the [Quickstart](quickstart.md)
2. **Coming from Python?** Read [Coming from Python](coming-from-python.md)
3. **Want a specific library?** Use the table above to find the right docs
