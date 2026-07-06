# Rune

Functional transformations — automatic differentiation, vectorizing maps,
and friends — for OCaml, inspired by [JAX](https://github.com/jax-ml/jax).

Rune differentiates ordinary OCaml functions over ordinary OCaml values:
there is no special tensor type (functions compute with plain
[Nx](../nx/) tensors) and no runtime tree encoding (parameters are your
own typed records). Declare once how to traverse a record's tensor
leaves — the `Nx.Ptree.S` interface, three one-liners — and every
transformation works on it directly, preserving its type.

## The Core Idea

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

let loss p = Nx.mean (Nx.square (Nx.sub (Nx.add (Nx.matmul x p.w) p.b) y))

(* The gradient of [loss] at [params] — a value of type [params]. *)
let grads = Rune.grad (module Params) loss params
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
  inputs; `in_axes`/`out_axis` control which axes are mapped
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
- **Debugging** — `with_debug` logs every tensor operation; `detach`
  and `no_grad` stop gradient flow
- **Structured outputs** — `vjp2`, `jvp2`, `vmap2` for functions
  returning a parameter structure rather than one tensor; primed
  variants (`grad'`, `vmap'`, ...) for single-tensor functions

## Quick Start

Gradient descent is `value_and_grad` plus a record update:

```ocaml
let step p =
  let l, g = Rune.value_and_grad (module Params) loss p in
  ({ w = Nx.sub p.w (Nx.mul_s g.w lr); b = Nx.sub p.b (Nx.mul_s g.b lr) }, l)
```

Per-sample gradients compose `vmap2` with `grad`:

```ocaml
let per_sample =
  Rune.vmap2
    (module Example)
    (module Params)
    (fun ex -> Rune.grad (module Params) (loss ex) params)
    batch
```

See the [API reference](lib/rune.mli) for the full contracts.

## Examples

- [`01-gradient-descent`](examples/01-gradient-descent) — fit a linear
  model by differentiating a function of a typed record
- [`02-per-sample-grads`](examples/02-per-sample-grads) — per-example
  gradients via `vmap2` of `grad`, checked against the loop
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
  for `svd`, `eig`, `eigh`, `rfft`, `irfft`, `psum`, and `mod`; forward
  mode additionally lacks `qr`. Differentiating through them raises
  `Invalid_argument` — `detach` the input if gradients should not flow
  through. (`cholesky` and reverse-mode `qr` are supported.)
- **`vmap` has no rule for `fft`-family and decomposition ops**
  (`fft`, `ifft`, `rfft`, `irfft`, `cholesky`, `qr`, `svd`, `eig`,
  `eigh`) over batched inputs.
- **Implicit RNG under `vmap` draws identical values for every lane** —
  the RNG key is a constant of the map. Thread distinct randomness in as
  mapped inputs instead.
- **In-place mutation** (`set_item`, `set_slice`, `blit`, `assign`)
  raises during differentiation; write the update functionally.
- **No JIT yet.** Everything runs eagerly; `scan`/`cond`/`while_loop`
  are designed so a future `jit` can stage them without unrolling.

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
