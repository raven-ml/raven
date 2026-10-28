# The Raven Ecosystem

Raven's libraries share one data type: `Nx.t`, the
n-dimensional array. Each library does one thing, and they compose
through tensors.

## How the Libraries Fit Together

```
 ┌──────────┐  ┌───────────┐
 │   Kaun   │  │   Norn    │  inference
 │  (Flax)  │  │ (NumPyro) │
 └────┬─────┘  └─────┬─────┘
      │        ┌─────┴─────┐
      │        │   Jera    │  numerical methods
      │        │  (SciPy)  │
      │        └─────┬─────┘
 ┌────┴──────────────┴─────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌─────────┐
 │          Rune           │ │ Talon  │ │  Brot  │ │ Hugin │ │  Quill  │
 │          (JAX)          │ │(Polars)│ │(HF Tok)│ │ (Mpl) │ │(Jupyter)│
 └────────────┬────────────┘ └───┬────┘ └───┬────┘ └───┬───┘ └────┬────┘
 ┌────────────┴──────────────────┴──────────┴──────────┴──────────┴────┐
 │                              Nx (NumPy)                             │
 └──────────────────────────────────┬──────────────────────────────────┘
                              ┌─────┴─────┐
                              │    Rig    │  devices and their work
                              └───────────┘
```

**Nx** is the foundation — every library operates on `Nx.t` tensors.

**Rig** sits under Nx. It opens devices such as GPUs, holds their memory,
and orders the work they run. Libraries that face devices directly build
on it.

**Rune** adds functional transformations on top of Nx: `grad`, `jvp`,
`vmap`. Your Nx code becomes differentiable without changes.

**Kaun** builds on Nx to provide layers, losses, initializers, data
batching, metrics, and HuggingFace Hub integration; its models are
differentiated with Rune in your own code. Models
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
| Run compiled work on a GPU or other device | [Rig](../packages/rig/doc/index.md) |
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

Nx provides the numerical foundation for the entire ecosystem:
operations on n-dimensional arrays with 21 data types (4-bit floats
and integers through complex128), broadcasting, slicing, linear
algebra, FFT, and I/O.

A value's type carries the device set it lies on. A value made from
numbers is a host value, `(_, _, Nx.host) Nx.t`, which lies in the
process's memory and which the host computes on. `Nx.devices` mints a
set from devices rig opens, with a brand of its own, so values of two
sets never meet by mistake, and `Nx.place` moves a value onto it, whole
on each device or cut across them. A set minted without `~kernels` over
GPUs computes nothing eagerly, and an operation there raises: its work
runs under `Rune.jit`. Random draws take a key, `~key`, or the next key
of a `Nx.Rng.with_key` scope ([nx.mli](../packages/nx/lib/nx.mli),
Device sets and Rng).

```ocaml
let x = Nx.linspace Nx.float32 0. 10. 100
let y = Nx.sin x
let mean_y = Nx.mean y
```

[Nx documentation →](../packages/nx/doc/index.md)

## Rig: Devices

Devices such as GPUs, and the order of the work they run. A program
opens a device, allocates buffers on it, copies between devices, and
submits work; rig orders the work of all devices by the memory it uses.
Apple GPUs run through Metal, NVIDIA GPUs through CUDA, and AMD GPUs
through Linux's `amdgpu`. Files and the devices of other machines open
as devices too.

<!-- $MDX skip -->
```ocaml
let gpu = Result.get_ok (Rig.open_ (Rig_metal.gpu 0))
let file = Result.get_ok (Rig_disk.of_file "weights.bin")
let weights = Rig.Buffer.create gpu (Rig.Buffer.length file)
let () = Rig.Buffer.copy ~src:file ~dst:weights
```

[Rig documentation →](../packages/rig/doc/index.md)

## Rune: Automatic Differentiation

Functional transformations for Nx tensors: reverse-mode AD (grad,
vjp), forward-mode AD (jvp), and vectorising maps (vmap). Operates on
`Nx.t` values directly using OCaml 5 effect handlers — no special
tensor type needed. Transformations work over any typed parameter
structure through the `Nx.Ptree.S` interface; primed variants take a
single tensor.

<!-- $MDX skip -->
```ocaml
let f x = Nx.add (Nx.mul x x) (Nx.sin x)
let f' x = Rune.grad' f x
let f'' x = Rune.grad' f' x
```

`Rune.jit` compiles a function through tolk into kernels for the
device its values lie on, and composes with the other transformations,
so a compiled function differentiates and maps like any other. A
program over values on several devices runs as its operations
evaluated by nx, which needs devices nx computes on eagerly
([rune.mli, jit](../packages/rune/lib/rune.mli)).

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

type ('a, 'd) mlp = { l1 : ('a, 'd) Linear.t; l2 : ('a, 'd) Linear.t }

let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))

let params =
  let k = Nx.Rng.split (Nx.Rng.key 0) in
  { l1 = Linear.init ~key:k.(0) ~inputs:784 ~outputs:128;
    l2 = Linear.init ~key:k.(1) ~inputs:128 ~outputs:10 }
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
  |> title "sin(x)"
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
