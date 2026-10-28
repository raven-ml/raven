# Coming from Python

This page maps Python scientific computing concepts to their Raven equivalents. It assumes you already know OCaml basics.

## Library Mapping

| Python | Raven | Notes |
|--------|-------|-------|
| NumPy | [Nx](../packages/nx/doc/index.md) | N-dimensional arrays, broadcasting, linear algebra, FFT |
| JAX | [Rune](../packages/rune/doc/index.md) | Functional transformations: `grad`, `jvp`, `vmap` |
| PyTorch / Flax | [Kaun](../packages/kaun/doc/index.md) | Layers, losses, data batching; training loops you write |
| Optax | [Vega](../packages/vega/doc/index.md) | Optimizers and learning-rate schedules |
| HuggingFace Tokenizers | [Brot](../packages/brot/doc/index.md) | BPE, WordPiece, Unigram; HF-compatible |
| pandas / Polars | [Talon](../packages/talon/doc/index.md) | Typed tables and queries; CSV and Parquet |
| Matplotlib | [Hugin](../packages/hugin/doc/index.md) | Figures from tensors, rendered to PNG, SVG and PDF |
| Jupyter + IPython | [Quill](../packages/quill/doc/index.md) | Interactive REPL and markdown notebooks |

## Key Differences

### Explicit Types

NumPy casts types silently. Nx does not.

```python
# Python: silently upcasts int + float -> float
a = np.array([1, 2, 3])
b = a + 1.5  # works
```

<!-- $MDX skip -->
```ocaml
(* OCaml: types must match *)
let a = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |]
(* Nx.add a (Nx.scalar Nx.float32 1.5)  -- type error *)

(* Cast explicitly *)
let a_f = Nx.cast Nx.float32 a
let b = Nx.add_s a_f 1.5
```

### Array Literals

NumPy uses Python lists. Nx uses OCaml arrays with `[| |]` syntax.

```python
x = np.array([[1, 2], [3, 4]])
```

<!-- $MDX skip -->
```ocaml
let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
```

### Slicing

NumPy uses `[]` with `:`. Nx uses the `slice` function with index constructors, which are `Nx`'s, so a slice is written inside a local open.

```python
x[0:2, :]           # first two rows
x[:, 1]             # second column
x[::2]              # every other row
```

<!-- $MDX skip -->
```ocaml
Nx.(slice [ R (0, 2); A ] x)          (* first two rows *)
Nx.(slice [ A; I 1 ] x)               (* second column *)
Nx.(slice [ Rs (0, dim 0 x, 2) ] x)   (* every other row *)
```

### No Separate Tensor Type

In PyTorch, `torch.Tensor` is different from `numpy.ndarray`. In Raven, Rune operates directly on `Nx.t` values. There is no wrapper type.

```python
# PyTorch: convert between types
x_np = np.array([1.0, 2.0])
x_torch = torch.from_numpy(x_np)
x_torch.requires_grad_(True)
```

<!-- $MDX skip -->
```ocaml
(* Raven: just use Nx tensors directly *)
let x = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |]
let gradient = Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) x
```

### Functional Transformations

JAX users will find Rune familiar. PyTorch users: think of `grad` as a function transformer, not a method on tensors.

```python
# JAX style
grad_fn = jax.grad(loss_fn)
grads = grad_fn(params)

# PyTorch style
loss = loss_fn(params)
loss.backward()
grads = params.grad
```

<!-- $MDX skip -->
```ocaml
(* Rune: JAX-style functional transforms over your own typed record *)
let params_ptree = Nx.Ptree.instantiate (module Params)
let grads = Rune.grad params_ptree loss_fn params

(* Or compute value and gradient together *)
let loss, grads = Rune.value_and_grad params_ptree loss_fn params
```

Where JAX registers pytrees, Rune takes a structure: `Params` implements `Nx.Ptree.S` with one function, `walk`, over your record's fields, `Nx.Ptree.instantiate` makes it a value, and gradients come back with the same type as the parameters.

### Module-Based Layers

Kaun layers are plain records with `init` and `apply` functions, not classes with `forward` — and a model is a record you define, not a container object.

```python
# PyTorch
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)
    def forward(self, x):
        return self.linear(x)
model = Model()
```

<!-- $MDX skip -->
```ocaml
(* Kaun: a model is a typed record of layer records *)
type ('a, 'd) model = { linear : ('a, 'd) Kaun.Linear.t }

let apply p x = Kaun.Linear.apply p.linear x

let params =
  { linear = Kaun.Linear.init ~key:(Nx.Rng.key 0) ~inputs:784 ~outputs:10 }
```

Parameters are plain data (records of Nx tensors), not hidden inside objects. A constructor draws them from the key it is given.

### Random Numbers

NumPy and PyTorch draw from a global generator that a seed resets. Nx draws from keys: a draw is a function of its key, dtype and shape. A sampler takes `~key`, or else the next key of the enclosing `Nx.Rng.with_key` scope; outside every scope, the next key of the domain's own, seeded from the system's entropy. `Nx.Rng.split` makes independent keys from one ([nx.mli, Rng](../packages/nx/lib/nx.mli)).

```python
np.random.seed(0)
x = np.random.normal(size=3)
```

<!-- $MDX skip -->
```ocaml
let x = Nx.Rng.normal ~key:(Nx.Rng.key 0) Nx.float32 [| 3 |]

let y =
  Nx.Rng.with_key (Nx.Rng.key 1) (fun () -> Nx.Rng.uniform Nx.float32 [| 3 |])
```

### Devices

In PyTorch, a tensor's device is a runtime attribute. In Raven, a value's type carries its device set. A value made from numbers is a host value, of type `(_, _, Nx.host) Nx.t`, and `Nx.devices` mints a set with a brand of its own from devices that rig opens. `Nx.place` moves a value onto a set, as `Nx.Ptree.place` moves a model. A set minted without `~kernels` over GPUs computes nothing eagerly, and an operation there raises, so a function over GPU values runs compiled, under `Rune.jit`, one device per compiled program ([nx.mli, Device sets](../packages/nx/lib/nx.mli); [rune.mli, jit](../packages/rune/lib/rune.mli)).

```python
x = torch.tensor([1.0, 2.0]).to("cuda")
y = x * 2
```

<!-- $MDX skip -->
```ocaml
module Gpu = (val Nx.devices [ Result.get_ok (Rig.open_ (Rig_cuda.gpu 0)) ])

let x = Nx.place Gpu.on (Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |])
let double = Rune.jit' (fun x -> Nx.mul_s x 2.)
let y = double x (* a Gpu.d value *)
```

PyTorch updates a tensor in place, as `x.add_(1)` does. Nx never changes a value. A program that wants a result to reuse a value's memory gives the value up at the call. `Nx.donate x`, or `Nx.Ptree.donate` for a structure, lets the operation or compiled call that reads it write its result there, and the donated value is dead afterwards ([nx.mli, Donation](../packages/nx/lib/nx.mli)).

### DataFrames

pandas computes each operation as it is called, and finds a wrong column name
when the line that uses it runs. Talon builds a query from typed column
handles, checks every verb against its input's columns before reading any
data, and computes the query when it runs.

```python
# pandas
df["bmi"] = df["weight"] / df["height"] ** 2
```

<!-- $MDX skip -->
```ocaml
(* Talon: a typed expression in a query *)
let weight = Talon.Col.float "weight" and height = Talon.Col.float "height"

let with_bmi df =
  Talon.Query.(df |> derive Talon.Expr.[ "bmi" := weight /. (height *. height) ])
```

## Detailed Comparisons

Each library has a dedicated comparison page with side-by-side code examples:

- [Nx vs NumPy](../packages/nx/doc/05-numpy-comparison.md)
- [Kaun vs PyTorch/Flax](../packages/kaun/doc/05-pytorch-comparison.md)
- [Vega vs Optax](../packages/vega/doc/04-optax-comparison.md)
- [Brot vs HuggingFace Tokenizers](../packages/brot/doc/06-hf-tokenizers-comparison.md)
- [Talon vs pandas](../packages/talon/doc/05-pandas-comparison.md)
- [Sowilo vs OpenCV](https://github.com/raven-ml/raven/blob/main/contrib/sowilo/doc/04-opencv-comparison.md) (contrib)
- [Fehu vs Gymnasium](https://github.com/raven-ml/raven/blob/main/contrib/fehu/doc/04-gymnasium-comparison.md) (contrib)
