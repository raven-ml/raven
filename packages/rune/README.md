# Rune

Transformations of OCaml functions over [nx](../nx/) values: automatic
differentiation, vectorizing maps, loops and compilation.

Rune differentiates ordinary OCaml functions over ordinary OCaml values:
functions compute with plain `Nx.t` values, and parameters are your own
typed records. Write once how to walk a record's parts, an `Nx.Ptree.S`
module with one `walk` and a line per field, and every transformation
takes and returns values of that record type.

## The Core Idea

```ocaml
type 'a params = { w : 'a; b : 'a }

module Params = struct
  type ('a, _) t = 'a params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let params_ptree = Nx.Ptree.instantiate (module Params)

let loss x y p =
  Nx.mean (Nx.square (Nx.sub (Nx.add (Nx.matmul x p.w) p.b) y))

let x = Nx.Rng.normal Nx.float32 [| 32; 3 |]
let y = Nx.Rng.normal Nx.float32 [| 32; 1 |]
let params = { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }

(* The gradient of [loss x y] at [params], a value of the same record type. *)
let grads = Rune.grad params_ptree (loss x y) params
```

Leaves may mix dtypes: one forward and one backward pass give a gradient
for each, of its leaf's dtype. Records nest into records, so models
compose structurally: see [kaun](../kaun/) for neural-network layers
built this way, and [vega](../vega/) for optimizers over the same
structures.

## Features

- **Reverse mode**: `grad` and `value_and_grad` for scalar objectives;
  `vjp` returns the result and its pullback, applied to any number of
  cotangents; `value_and_grad_aux` returns an objective's auxiliary data;
  `linear_transpose` transposes a linear function
- **Forward mode**: `jvp` for Jacobian-vector products in one forward
  pass; `linearize` keeps the tangent map to apply again
- **Vectorizing map**: `vmap` lifts a per-example function to batched
  inputs, mapping axis 0 of every argument its signature lists
- **Composition**: transformations nest: `vmap` of `grad` is per-sample
  gradients, `jvp` of `grad` a Hessian-vector product, `grad` of `grad`
  a second derivative
- **Jacobians and Hessians**: `jacfwd'` and `jacrev'`; a Hessian is
  `jacfwd' (grad' f)`
- **Loops**: `scan` folds a step over rows and `iterate` repeats one
  until a condition holds; each traces its step once, at the call, and
  has derivatives in either mode
- **Gradient checkpointing**: `remat` recomputes a function in the
  backward pass, trading compute for memory
- **Custom rules**: `custom_jvp` gives a function a tangent map that
  serves both modes at every order, `custom_vjp` a pullback, and
  `custom_transpose` a linear function its transpose
- **Totals and lanes**: `Total` collects a write-only sum that code
  anywhere inside a function adds to, once per execution; `axis` names a
  map, and `lanes` and `lane_index` read its lanes
- **Gradient checking**: `check_grads` compares reverse mode with finite
  differences, and `check_transpose` a transpose with its function
- **Compilation**: `jit` traces a function once per abstract value of its
  arguments and runs the program compiled on the devices its values lie
  on (the host, CUDA, Metal, AMD); a call consumes the arguments the
  caller donates (`Nx.donate`), and its results may take their memory
- **Programs**: `program`, `pp_program` and `eval_shape` show what a
  function traces, without computing it
- **Stopping gradients**: `detach` holds a value constant
- **Structures**: every transformation takes the structures it walks,
  so results may be structures too; primed forms (`grad'`, `vmap'`,
  ...) serve functions of one array

## Quick Start

Gradient descent is `value_and_grad` plus a record update, compiled with
`jit`; donating the parameters lets the update take their memory:

```ocaml
let step =
  Rune.jit
    Nx.Ptree.(params_ptree @-> returns (pair params_ptree tensor))
    (fun p ->
      let l, g = Rune.value_and_grad params_ptree (loss x y) p in
      ({ w = Nx.sub p.w (Nx.mul_s g.w 0.1); b = Nx.sub p.b (Nx.mul_s g.b 0.1) }, l))

let params, l = step (Nx.Ptree.donate params_ptree params)
```

Per-sample gradients compose `vmap` with `grad`. The signature says that
the mapped function takes two arrays and returns the parameters'
structure; each argument is mapped over its axis 0:

```ocaml
let per_sample =
  Rune.vmap
    Nx.Ptree.(tensor @-> tensor @-> returns params_ptree)
    (fun x y -> Rune.grad params_ptree (loss x y) params)
    x y
```

A loop's step is traced once per call, eagerly too: it branches with
`Nx.where` and draws from the keys it receives:

```ocaml
let walk, () =
  Rune.scan Nx.Ptree.tensor (Nx.Ptree.v Nx.Rng.walk) Nx.Ptree.unit
    ~init:(Nx.zeros Nx.float32 [| 2 |])
    ~f:(fun x key -> (Nx.add x (Nx.Rng.normal ~key Nx.float32 [| 2 |]), ()))
    (Nx.Rng.split_batch ~n:100 (Nx.Rng.key 0))
```

See the [guide](doc/index.md) and the [API reference](lib/rune.mli) for
the full contracts.

## Examples

- [`01-gradient-descent`](examples/01-gradient-descent): fit a linear
  model by differentiating a function of a typed record
- [`02-per-sample-grads`](examples/02-per-sample-grads): per-example
  gradients via `vmap` of `grad`, checked against the loop
- [`03-hessian`](examples/03-hessian): Newton's method with the
  Hessian `jacfwd' (grad' f)`, a matrix-free Hessian-vector product, and
  `check_grads`

Run any of them with `dune exec`, e.g.

```sh
dune exec packages/rune/examples/01-gradient-descent/main.exe
```

## Limitations

Rune raises rather than return a wrong derivative:

- **Derivatives with no definition raise.** The tangents of
  `Nx.qr ~factors:Complete` of a tall matrix, of
  `Nx.svd ~factors:Complete` of a non-square one, and of a block scale of
  `Nx.contract` that depends on the parameters raise `Invalid_argument`;
  `detach` the input if gradients should not flow through.
- **A construct's functions read no values.** `scan`, `iterate`,
  `remat`, `jit` and the custom rules trace their functions, so
  `Nx.item` of a value computed from their arguments, or a draw without
  `~key`, inside them raises.
- **Eager loops over small arrays pay per operation.** An eager `scan`
  evaluates its traced step per row, which costs more per operation than
  computing it directly; `jit` removes that cost.

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
