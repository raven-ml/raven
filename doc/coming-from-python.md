# Coming from Python

This page maps Python scientific computing concepts to their Raven equivalents. It assumes you already know OCaml basics.

## Library Mapping

| Python | Raven | Notes |
|--------|-------|-------|
| NumPy | [Nx](../packages/nx/doc/index.md) | N-dimensional arrays, broadcasting, linear algebra, FFT |
| JAX | [Rune](../packages/rune/doc/index.md) | Functional transformations: `grad`, `jvp`, `vmap` |
| PyTorch / Flax | [Kaun](../packages/kaun/doc/index.md) | Layers, optimizers, training loops |
| HuggingFace Tokenizers | [Brot](../packages/brot/doc/index.md) | BPE, WordPiece, Unigram; HF-compatible |
| pandas / Polars | [Talon](../packages/talon/doc/index.md) | Type-safe DataFrames |
| Matplotlib | [Hugin](../packages/hugin/doc/index.md) | 2D/3D plotting with Cairo |
| Gymnasium | [Fehu](../packages/fehu/doc/index.md) | RL environments and training utilities |
| OpenCV | [Sowilo](../packages/sowilo/doc/index.md) | Differentiable image processing |
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
let a = Nx.create Nx.Int32 [|3|] [|1l; 2l; 3l|]
(* Nx.add a (Nx.scalar Nx.Float32 1.5)  -- type error *)

(* Cast explicitly *)
let a_f = Nx.cast Nx.Float32 a
let b = Nx.add a_f (Nx.scalar Nx.Float32 1.5)
```

### Array Literals

NumPy uses Python lists. Nx uses OCaml arrays with `[| |]` syntax.

```python
x = np.array([[1, 2], [3, 4]])
```

<!-- $MDX skip -->
```ocaml
let x = Nx.create Nx.Float32 [|2; 2|] [|1.; 2.; 3.; 4.|]
```

### Slicing

NumPy uses `[]` with `:`. Nx uses the `slice` function with index constructors.

```python
x[0:2, :]           # first two rows
x[:, 1]             # second column
x[::2]              # every other element
```

<!-- $MDX skip -->
```ocaml
Nx.slice [R (0, 2); A] x      (* first two rows *)
Nx.slice [A; I 1] x            (* second column *)
Nx.slice [S (0, -1, 2)] x     (* every other element *)
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
let x = Nx.create Nx.Float32 [|2|] [|1.0; 2.0|]
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
let grads = Rune.grad (module Params) loss_fn params

(* Or compute value and gradient together *)
let loss, grads = Rune.value_and_grad (module Params) loss_fn params
```

Where JAX registers pytrees, Rune takes a first-class module: `Params` implements `Nx.Ptree.S` (three one-line traversals over your record's tensor leaves), and gradients come back with the same type as the parameters.

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
type 'a model = { linear : 'a Kaun.Linear.t }

let apply p x = Kaun.Linear.apply p.linear x
let params = { linear = Kaun.Linear.init ~inputs:784 ~outputs:10 }
```

Parameters are plain data (records of Nx tensors), not hidden inside objects.

### DataFrames

pandas uses string-based column access. Talon provides type-safe row operations via an applicative.

```python
# pandas
df['bmi'] = df['weight'] / df['height'] ** 2
```

<!-- $MDX skip -->
```ocaml
(* Talon: type-safe row computation *)
let df = Talon.with_column df "bmi" Nx.Float64
  Talon.Row.(map2 (number "weight") (number "height")
    ~f:(fun w h -> w /. (h *. h)))
```

## Detailed Comparisons

Each library has a dedicated comparison page with side-by-side code examples:

- [Nx vs NumPy](../packages/nx/doc/05-numpy-comparison.md)
- [Rune vs JAX](../packages/rune/doc/04-jax-comparison.md)
- [Kaun vs PyTorch/Flax](../packages/kaun/doc/05-pytorch-comparison.md)
- [Brot vs HuggingFace Tokenizers](../packages/brot/doc/06-hf-tokenizers-comparison.md)
- [Talon vs pandas](../packages/talon/doc/03-pandas-comparison.md)
- [Hugin vs Matplotlib](../packages/hugin/doc/05-matplotlib-comparison.md)
- [Sowilo vs OpenCV](../packages/sowilo/doc/04-opencv-comparison.md)
- [Fehu vs Gymnasium](../packages/fehu/doc/04-gymnasium-comparison.md)
