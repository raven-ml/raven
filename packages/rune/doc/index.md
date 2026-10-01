# rune

Rune provides functional transformations — automatic differentiation, vectorizing maps, and friends — for ordinary OCaml functions over ordinary OCaml values. There is no special tensor type (functions compute with plain `Nx.t` tensors) and no runtime tree encoding: parameters are your own typed records. Write once how to walk a record's parts — an `Nx.Ptree.S` module with one `walk`, a line per field — and every transformation works on it directly, preserving its type.

## Features

- **Reverse mode** — `grad`, `value_and_grad` for backpropagation; `vjp` returns the result and its pullback; `value_and_grad_aux` returns an objective's auxiliary data
- **Forward mode** — `jvp` for Jacobian-vector products in a single forward pass
- **Vectorizing map** — `vmap` lifts a per-example function to batched inputs, mapping axis 0 of every argument
- **Composable** — transformations nest freely: `vmap` of `grad` is per-sample gradients, `jvp` of `grad` is a Hessian-vector product, `grad` of `grad` is second order
- **Jacobians and Hessians** — `jacfwd'` and `jacrev'`; a Hessian is `jacfwd' (grad' f)`
- **Gradient checkpointing** — `remat` trades compute for memory in the backward pass
- **Custom rules** — `custom_jvp` gives a function a tangent map that serves both modes at every order, and `custom_vjp` a pullback
- **Interpreters of Nx's operations** — transformations intercept Nx operations as they run; no tracing, no graph

## Quick Start

A parameter structure is a record plus a `walk` over its fields. The gradient of a function of the record is a value of the same record type:

```ocaml
type 'a params = { w : 'a; b : 'a }

module Params = struct
  type 'a t = 'a params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let params_ptree = Nx.Ptree.instantiate (module Params)

let () =
  let x = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 2.; 3. |] in
  let loss p =
    Nx.mean (Nx.square (Nx.sub (Nx.add (Nx.matmul x p.w) p.b) y))
  in
  let params =
    { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  let g = Rune.grad params_ptree loss params in
  Format.printf "dw has shape %a, db has shape %a@."
    Nx.pp_shape (Nx.shape g.w) Nx.pp_shape (Nx.shape g.b)
```

Leaves may mix dtypes freely: a single forward and backward pass produces gradients for all of them, each with its leaf's dtype. Records nest into records, so models compose structurally — see [kaun](../../kaun/doc/index.md) for neural-network layers built this way, and [vega](../../vega/doc/index.md) for optimizers that consume the same structures.

## Next Steps

- [Getting Started](01-getting-started.md) — installation, first gradients, gradient descent on a record
- [Transformations](02-transformations.md) — complete guide to grad, vjp, jvp, vmap, remat, custom rules, and control flow
- [How It Works](03-how-it-works.md) — interpreters, forward mode and its transpose
- [JAX Comparison](04-jax-comparison.md) — mapping JAX vocabulary to rune
- [Compilation](05-compilation.md) — `jit` signatures, consumption, devices, the persistent cache, beam search and debugging
