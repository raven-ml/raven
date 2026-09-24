# Rune

Functional transformations — automatic differentiation, vectorizing maps,
and friends — for OCaml, inspired by [JAX](https://github.com/jax-ml/jax).

Rune differentiates ordinary OCaml functions over ordinary OCaml values:
there is no special tensor type (functions compute with plain
[Nx](../nx/) tensors) and no runtime tree encoding (parameters are your
own typed records). Write once how to walk a record's parts, an
`Nx.Ptree.S` module with one `walk` and a line per field, and every
transformation works on it directly, preserving its type.

## The Core Idea

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
let loss p = Nx.mean (Nx.square (Nx.sub (Nx.add (Nx.matmul x p.w) p.b) y))

(* The gradient of [loss] at [params], a value of the same record type. *)
let grads = Rune.grad params_ptree loss params
```

Leaves may mix dtypes freely: a single forward and backward pass
produces gradients for all of them, each with its leaf's dtype. Records
nest into records, so models compose structurally — see
[kaun](../kaun/) for neural-network layers built this way, and
[vega](../vega/) for optimizers that consume the same structures.

## Features

- **Reverse mode** — `grad`, `value_and_grad`, `vjp` for scalar
  objectives and explicit cotangents; `vjp_fun` returns a reusable
  pullback; `_aux` variants thread non-differentiated data out of the
  objective
- **Forward mode** — `jvp` for Jacobian-vector products in a single
  forward pass
- **Vectorizing map** — `vmap` lifts a per-example function to batched
  inputs, mapping axis 0 of every argument its signature lists
- **Composition** — transformations nest freely: `vmap` of `grad` is
  per-sample gradients, `jvp` of `grad` powers `hvp`, `grad` of `grad`
  is second order
- **Jacobians and Hessians** — `jacfwd'`, `jacrev'`, `hessian'`, and
  matrix-free `hvp`
- **Gradient checkpointing** — `remat` recomputes a sub-computation in
  the backward pass, trading compute for memory
- **Custom rules** — `custom_vjp` and `custom_jvp` override
  differentiation for a function you know a better rule for
- **Gradient checking** — `check_grads` compares reverse mode against
  finite differences
- **Control flow** — `scan`, `cond`, `while_loop` combinators with
  staging-ready signatures
- **Compilation** — `jit` traces a function once per key and replays it
  as fused kernels, on the host, CUDA, or Metal via `~devices`; an argument
  marked `consumes` in its signature is given up by each call, and its
  storage is reused for the results
- **Debugging** — `with_debug` logs every tensor operation; `detach`
  and `no_grad` stop gradient flow
- **Structures** — every transformation takes the structures it walks,
  so results may be structures too; primed variants (`grad'`, `vmap'`,
  ...) serve single-tensor functions

## Quick Start

Gradient descent is `value_and_grad` plus a record update:

```ocaml
let step p =
  let l, g = Rune.value_and_grad params_ptree loss p in
  ({ w = Nx.sub p.w (Nx.mul_s g.w lr); b = Nx.sub p.b (Nx.mul_s g.b lr) }, l)
```

Per-sample gradients compose `vmap` with `grad`. The signature says that
the mapped function takes two tensors and returns the parameters'
structure; each argument is mapped over its axis 0:

```ocaml
let per_sample =
  Rune.vmap
    Nx.Ptree.(tensor @-> tensor @-> returns params_ptree)
    (fun x y -> Rune.grad params_ptree (loss x y) params)
    xs ys
```

See the [API reference](lib/rune.mli) for the full contracts.

## Examples

- [`01-gradient-descent`](examples/01-gradient-descent) — fit a linear
  model by differentiating a function of a typed record
- [`02-per-sample-grads`](examples/02-per-sample-grads) — per-example
  gradients via `vmap` of `grad`, checked against the loop
- [`03-hessian`](examples/03-hessian) — Newton's method with
  `hessian'`, matrix-free `hvp'`, and `check_grads`

Run any of them with `dune exec`, e.g.

```sh
dune exec packages/rune/examples/01-gradient-descent/main.exe
```

## Limitations

Rune aims to fail loudly rather than return wrong gradients.
Current gaps:

- **Ops without differentiation rules raise.** Reverse mode has no rule
  for `svd`, `eig`, `eigh`, `psum`, and `mod`; forward mode additionally
  lacks `qr`. Differentiating through them raises `Invalid_argument` —
  `detach` the input if gradients should not flow through. (`cholesky`,
  reverse-mode `qr`, and the whole FFT family are supported.)
- **`vmap` has no rule for decomposition ops** (`cholesky`, `qr`,
  `svd`, `eig`, `eigh`) over batched inputs.
- **Implicit RNG under `vmap` draws identical values for every lane** —
  the RNG key is a constant of the map. Thread distinct randomness in as
  mapped inputs instead.
- **`jit` unrolls `scan`**, so a recurrence's compile time grows with
  its sequence length.

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
