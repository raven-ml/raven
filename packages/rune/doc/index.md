# rune

Rune provides functional transformations — automatic differentiation, vectorizing maps, and friends — for ordinary OCaml functions over ordinary OCaml values. There is no special tensor type (functions compute with plain `Nx.t` tensors) and no runtime tree encoding: parameters are your own typed records. Declare once how to traverse a record's tensor leaves — the `Nx.Ptree.S` interface, three one-liners — and every transformation works on it directly, preserving its type.

## Features

- **Reverse mode** — `grad`, `value_and_grad`, `vjp` for backpropagation; `vjp_fun` returns a reusable pullback; `_aux` variants thread auxiliary data out of the objective
- **Forward mode** — `jvp` for Jacobian-vector products in a single forward pass
- **Vectorizing map** — `vmap` lifts a per-example function to batched inputs, with `in_axes`/`out_axis` control
- **Composable** — transformations nest freely: `vmap` of `grad` is per-sample gradients, `jvp` of `grad` powers `hvp`, `grad` of `grad` is second order
- **Jacobians and Hessians** — `jacfwd'`, `jacrev'`, `hessian'`, and matrix-free `hvp`
- **Gradient checkpointing** — `remat` trades compute for memory in the backward pass
- **Custom rules** — `custom_vjp` and `custom_jvp` override differentiation where you know a better rule
- **Effect-based** — OCaml 5 effect handlers intercept Nx operations; no tracing, no graph

## Quick Start

A parameter structure is a record plus three one-line traversals. The gradient of a function of the record is a value of the same record type:

```ocaml
type params = { w : Nx.float32_t; b : Nx.float32_t }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
    { w = f w; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
    f w;
    f b
end

let () =
  let x = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 2.; 3. |] in
  let loss p =
    Nx.mean (Nx.square (Nx.sub (Nx.add (Nx.matmul x p.w) p.b) y))
  in
  let params =
    { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  let g = Rune.grad (module Params) loss params in
  Printf.printf "dw has shape %s, db has shape %s\n"
    (Nx.shape_to_string (Nx.shape g.w))
    (Nx.shape_to_string (Nx.shape g.b))
```

Leaves may mix dtypes freely: a single forward and backward pass produces gradients for all of them, each with its leaf's dtype. Records nest into records, so models compose structurally — see [kaun](/docs/kaun/) for neural-network layers built this way, and [vega](/docs/vega/) for optimizers that consume the same structures.

## Next Steps

- [Getting Started](01-getting-started/) — installation, first gradients, gradient descent on a record
- [Transformations](02-transformations/) — complete guide to grad, vjp, jvp, vmap, remat, custom rules, and control flow
- [How It Works](03-how-it-works/) — effects, handlers, and the tape
- [JAX Comparison](04-jax-comparison/) — mapping JAX vocabulary to rune
