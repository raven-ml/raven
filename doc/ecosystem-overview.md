# The Raven Ecosystem

Raven's libraries share one data type: `Nx.t`, the
n-dimensional array. Each library does one thing, and they compose
through tensors.

## How the Libraries Fit Together

```
       ┌───────────┐                  ┌───────────┐
       │   Kaun    │  neural networks │   Norn    │  inference
       │  (Flax)   │                  │ (NumPyro) │
       └─────┬─────┘                  └─────┬─────┘
             │                        ┌─────┴─────┐
             │                        │   Jera    │  numerical methods
             │                        │  (SciPy)  │
             │                        └─────┬─────┘
       ┌─────┴──────────────────────────────┴──────┐
       │                    Rune                   │  autodiff, vmap
       │                   (JAX)                   │
       └───────────────────────┬───────────────────┘
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
batching, metrics, and HuggingFace Hub integration. Models
are typed records you define; optimizers come from **Vega**.

**Jera** builds on Rune to solve systems, find minima, integrate,
interpolate and solve differential equations, with answers Rune
differentiates. **Norn** builds on both for probabilistic inference.

**Talon**, **Brot**, **Hugin**, and **Quill** each use Nx directly for
their domain.

## Which Library Do I Need?

| I want to... | Use |
|---|---|
| Work with numerical arrays | [Nx](../packages/nx/doc/index.md) |
| Compute gradients | [Rune](../packages/rune/doc/index.md) |
| Solve equations, integrate, interpolate, fit | [Jera](../packages/jera/doc/index.md) |
| Fit a probabilistic model | [Norn](../packages/norn/doc/index.md) |
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
apply functions, losses, initializers, data batching, metrics, and
HuggingFace Hub integration. A model is a typed record
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

## Jera: Numerical Methods

Linear and nonlinear systems, minima with bounds, integrals,
interpolation, and ordinary, stiff, differential-algebraic, delay and
stochastic differential equations. A problem is an OCaml function over
tensors. A solve returns its answer with a status per lane and an error
estimate against a stated tolerance, and Rune differentiates the answer
through its own equation.

<!-- $MDX skip -->
```ocaml
open Jera

let pendulum _t (q, p) = (p, Nx.neg (Nx.sin q))

let q, _ =
  Ode.solve Nx.Ptree.(pair tensor tensor) Ode.tsit5
    ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12) ~budget:1000 pendulum
    ~t0:(Nx.scalar Nx.float64 0.) ~t1:(Nx.scalar Nx.float64 10.)
    (Nx.scalar Nx.float64 1., Nx.scalar Nx.float64 0.)
  |> Solution.get
```

[Jera documentation →](../packages/jera/doc/index.md)

## Norn: Probabilistic Inference

Distributions, bijectors, and samplers (NUTS, HMC, ensemble, nested
sampling, tempered SMC) over a structure of your own type, with
diagnostics as values of that structure. `norn.model` turns one
generative function into the densities the samplers need.

<!-- $MDX skip -->
```ocaml
let state = Norn.Nuts.init params log_density start
let state = Norn.Nuts.warmup params log_density key ~steps:500 state
let _, draws, stats = Norn.Nuts.sample params log_density key ~draws:500 state
let () = Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v params ~stats draws)
```

[Norn documentation →](../packages/norn/doc/index.md)

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

Tables of named, typed columns, with one null state for every type. A query
is a plan of verbs (`filter`, `derive`, `aggregate`, `join`, `sort`) over a
table or a CSV or Parquet file, checked against the columns before any data
is read. A numeric column becomes an Nx tensor without a copy.

<!-- $MDX skip -->
```ocaml
open Talon

let mean_score =
  Query.(
    of_table df
    |> aggregate ~by:[ "team" ] Expr.[ "mean" := mean (Col.float "score") ]
    |> run)
```

[Talon documentation →](../packages/talon/doc/index.md)

## Hugin: Visualization

Figures as values. Marks (lines, dots, rects, areas, text, images, contours)
read Nx tensors through channels, compose into layers, grids and facets, and
render to PNG, SVG or PDF with a bundled font.

<!-- $MDX skip -->
```ocaml
open Hugin

let x = Nx.linspace Nx.float32 0. 6.28 100

let () =
  line ~x:(num x) ~y:(num (Nx.sin x)) ()
  |> title (Text.v "sin(x)")
  |> save "sine.svg"
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

- [Fehu](https://github.com/raven-ml/raven/tree/main/contrib/fehu): reinforcement learning environments
- [Sowilo](https://github.com/raven-ml/raven/tree/main/contrib/sowilo): differentiable computer vision
- [Ymir](https://github.com/raven-ml/raven/tree/main/contrib/ymir): astronomy: units, frames, FITS, photometry, cosmology

## Getting Started

1. **New to Raven?** Start with the [Quickstart](quickstart.md)
2. **Coming from Python?** Read [Coming from Python](coming-from-python.md)
3. **Want a specific library?** Use the table above to find the right docs
