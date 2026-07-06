# The Raven Ecosystem

Raven is nine libraries that share one data type: `Nx.t`, the
n-dimensional array. Each library does one thing, and they compose
through tensors.

## How the Libraries Fit Together

```
                         ┌───────────┐
                         │   Kaun    │  neural networks
                         │  (Flax)   │
                         └─────┬─────┘
                               │
  ┌───────────┐          ┌─────┴─────┐          ┌───────────┐
  │  Sowilo   │          │   Rune    │          │   Fehu    │
  │ (OpenCV)  ├──────────┤  (JAX)    ├──────────┤(Gymnasium)│
  └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
        │                      │                      │
  ┌─────┴──────────────────────┴──────────────────────┴─────┐
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

**Sowilo**, **Fehu**, **Talon**, **Brot**, **Hugin**, and **Quill** each
use Nx directly for their domain. Sowilo and Fehu operations are
compatible with Rune's `grad` and `vmap` since they are plain Nx
operations under the hood.

## Which Library Do I Need?

| I want to... | Use |
|---|---|
| Work with numerical arrays | [Nx](/docs/nx/) |
| Compute gradients | [Rune](/docs/rune/) |
| Train neural networks | [Kaun](/docs/kaun/) |
| Tokenize text for language models | [Brot](/docs/brot/) |
| Manipulate tabular data | [Talon](/docs/talon/) |
| Process and transform images | [Sowilo](/docs/sowilo/) |
| Build RL environments and agents | [Fehu](/docs/fehu/) |
| Create plots and visualizations | [Hugin](/docs/hugin/) |
| Run code interactively (REPL or notebooks) | [Quill](/docs/quill/) |

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

[Nx documentation →](/docs/nx/)

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

[Rune documentation →](/docs/rune/)

## Kaun: Neural Networks

Building blocks for neural networks: layers as plain records with pure
apply functions, losses, initializers, data batching, metrics,
checkpoints, and HuggingFace Hub integration. A model is a typed record
you define — there is no layer object and no trainer. Training steps
compose `Rune.value_and_grad` with a Vega optimizer update.

<!-- $MDX skip -->
```ocaml
open Kaun

type mlp = { l1 : Linear.t; l2 : Linear.t }

let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))

let params =
  { l1 = Linear.init ~inputs:784 ~outputs:128;
    l2 = Linear.init ~inputs:128 ~outputs:10 }
```

[Kaun documentation →](/docs/kaun/)

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

[Brot documentation →](/docs/brot/)

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

[Talon documentation →](/docs/talon/)

## Sowilo: Computer Vision

Differentiable image processing: geometric transforms (resize, crop,
flip), spatial filters (Gaussian blur, Sobel, Canny), color space
conversions, and morphological operations. All operations are plain Nx
computations, so they compose with `Rune.grad` and `Rune.vmap`.

<!-- $MDX skip -->
```ocaml
open Sowilo

let processed =
  img
  |> to_float
  |> resize ~height:224 ~width:224 ~mode:Bilinear
  |> normalize ~mean:[|0.485; 0.456; 0.406|] ~std:[|0.229; 0.224; 0.225|]
```

[Sowilo documentation →](/docs/sowilo/)

## Fehu: Reinforcement Learning

RL environments (CartPole, MountainCar, GridWorld), type-safe
observation/action spaces, vectorized environments, trajectory
collection, replay buffers, and generalized advantage estimation.

<!-- $MDX skip -->
```ocaml
open Fehu

let env = Fehu_envs.cartpole () in
let obs, _info = Env.reset env in
let obs, reward, terminated, truncated, _info =
  Env.step env (Space.sample (Env.action_space env))
```

[Fehu documentation →](/docs/fehu/)

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

[Hugin documentation →](/docs/hugin/)

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

[Quill documentation →](/docs/quill/)

## Getting Started

1. **New to Raven?** Start with the [Quickstart](/docs/quickstart/)
2. **Coming from Python?** Read [Coming from Python](/docs/coming-from-python/)
3. **Want a specific library?** Use the table above to find the right docs
