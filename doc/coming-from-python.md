# Coming from Python

This page maps Python scientific computing concepts to their Raven equivalents. It assumes you already know OCaml basics.

## Library Mapping

| Python | Raven | Notes |
|--------|-------|-------|
| NumPy | [Nx](/docs/nx/) | N-dimensional arrays, broadcasting, linear algebra, FFT |
| JAX | [Rune](/docs/rune/) | Functional transformations: `grad`, `jvp`, `vmap` |
| PyTorch / Flax | [Kaun](/docs/kaun/) | Layers, optimizers, training loops |
| HuggingFace Tokenizers | [Brot](/docs/brot/) | BPE, WordPiece, Unigram; HF-compatible |
| pandas / Polars | [Talon](/docs/talon/) | Type-safe DataFrames |
| Matplotlib | [Hugin](/docs/hugin/) | 2D/3D plotting with Cairo |
| Gymnasium | [Fehu](/docs/fehu/) | RL environments and training utilities |
| OpenCV | [Sowilo](/docs/sowilo/) | Differentiable image processing |
| Jupyter | [Quill](/docs/quill/) | Markdown files as notebooks |

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
let a_f = Nx.astype Nx.Float32 a
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
let gradient = Rune.grad (fun x -> Nx.sum (Nx.mul x x)) x
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
(* Rune: JAX-style functional transforms *)
let grad_fn = Rune.grad loss_fn
let grads = grad_fn params

(* Or compute value and gradient together *)
let loss, grads = Rune.value_and_grad loss_fn params
```

### Module-Based Layers

Kaun layers are records with `init` and `apply`, not classes with `forward`.

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
(* Kaun: compose layer records *)
let model = Kaun.Layer.sequential [
  Kaun.Layer.linear ~in_features:784 ~out_features:10 ();
]
let vars = Kaun.Layer.init model ~dtype:Nx.Float32
```

Parameters are plain data (`Ptree.t` — a tree of Nx tensors), not hidden inside objects.

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

- [Nx vs NumPy](/docs/nx/numpy-comparison/)
- [Rune vs JAX](/docs/rune/jax-comparison/)
- [Kaun vs PyTorch/Flax](/docs/kaun/pytorch-comparison/)
- [Brot vs HuggingFace Tokenizers](/docs/brot/hf-tokenizers-comparison/)
- [Talon vs pandas](/docs/talon/pandas-comparison/)
- [Hugin vs Matplotlib](/docs/hugin/matplotlib-comparison/)
- [Sowilo vs OpenCV](/docs/sowilo/opencv-comparison/)
- [Fehu vs Gymnasium](/docs/fehu/gymnasium-comparison/)
