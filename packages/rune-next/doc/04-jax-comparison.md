# Rune-next vs. JAX — A Practical Comparison

This guide explains how rune-next's transformations relate to [JAX](https://docs.jax.dev/), focusing on:

* How core concepts map (grad, vjp, jvp, vmap, pytrees, custom rules, remat)
* Where the APIs feel similar vs. deliberately different
* What rune-next does not have yet, stated honestly

If you already use JAX, this should be enough to become productive in rune-next quickly.

---

## 1. Big-Picture Differences

| Aspect | JAX (Python) | rune-next (OCaml) |
| --- | --- | --- |
| Language | Dynamic, interpreted | Statically typed, compiled |
| Array type | `jax.Array` | `Nx.t` (no separate rune-next tensor type) |
| Array library | `jax.numpy` | Nx |
| AD mechanism | Tracing + XLA compilation | OCaml 5 effect handlers, eager |
| Parameter containers | Pytrees (registered runtime trees) | `Nx.Ptree.S` — your own typed records |
| Reverse mode | `jax.grad`, `jax.value_and_grad` | `grad`, `value_and_grad`, `_aux` variants |
| VJP | `jax.vjp` | `vjp`, `vjp_fun` (reusable pullback), `vjp2` |
| Forward mode | `jax.jvp` | `jvp`, `jvp_aux`, `jvp2` |
| Vectorizing map | `jax.vmap` | `vmap`, `vmap2`, `vmap'` |
| Custom rules | `jax.custom_vjp`, `jax.custom_jvp` | `custom_vjp`, `custom_jvp` |
| Checkpointing | `jax.checkpoint` / `jax.remat` | `remat` |
| Jacobians / Hessians | `jacfwd`, `jacrev`, `hessian` | `jacfwd'`, `jacrev'`, `hessian'`, `hvp` |
| Control flow | `lax.scan`, `lax.cond`, `lax.while_loop` (required under `jit`) | `scan`, `cond`, `while_loop` (optional, staging-ready) plus ordinary OCaml control flow |
| Gradient stopping | `jax.lax.stop_gradient` | `detach`, `no_grad` |
| Gradient checking | `jax.test_util.check_grads` | `check_grads` |
| Randomness | Explicit splittable keys (`jax.random`) | Implicit scoped RNG (`Nx.Rng.run`) |
| JIT compilation | `jax.jit` | Not yet implemented |
| Devices | `jax.device_put`, GPU/TPU | CPU only |

---

## 2. Pytrees → Ptree.S

This is the deepest difference. JAX flattens arbitrary registered containers into lists of leaves at runtime:

```python
import jax

params = {"w": w, "b": b}          # any registered pytree
grads = jax.grad(loss)(params)      # same pytree of gradients
```

Rune-next has no runtime tree. A parameter structure is a record you define, and you tell the library how to traverse its tensor leaves by implementing `Nx.Ptree.S` — three hand-written one-liners, no ppx, no registration table:

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
```

Every transformation takes the module as a first-class argument and preserves the type: the gradient of a function of `params` *is* a `params`. There is no `tree_map` because `Params.map` is `tree_map`, specialized to your type — and `Rune_next.Ptree` (that is, `Nx.Ptree.t`) is the stock dynamic instance for structures only known at runtime, the closest analogue of a raw pytree.

Where JAX distinguishes leaves by position in a flattened list, rune-next leaves keep their record field names, dtypes, and shapes in the type. Mixed dtypes work: a single backward pass produces a gradient for every leaf, each with its leaf's dtype.

---

## 3. Reverse Mode

**JAX**

```python
def loss(params):
    return jnp.mean((x @ params["w"] + params["b"] - y) ** 2)

grads = jax.grad(loss)(params)
loss_value, grads = jax.value_and_grad(loss)(params)
```

**rune-next**

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 2.; 3. |] in
  let loss p =
    Nx.mean (Nx.square (Nx.sub (Nx.add (Nx.matmul x p.w) p.b) y))
  in
  let params =
    { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  let grads = Rune_next.grad (module Params) loss params in
  let loss_value, grads' =
    Rune_next.value_and_grad (module Params) loss params
  in
  ignore (grads, loss_value, grads')
```

Both require a scalar output. JAX's `argnums` has no equivalent: differentiate with respect to *the* parameter structure and close over everything else. For a function of one tensor, `grad'` skips the module argument.

### Auxiliary outputs

JAX uses a flag; rune-next has dedicated `_aux` variants:

```python
(loss, aux), grads = jax.value_and_grad(f, has_aux=True)(params)
```

<!-- $MDX skip -->
```ocaml
let loss, grads, aux = Rune_next.value_and_grad_aux (module Params) f params
```

---

## 4. VJP and JVP

JAX's `jax.vjp` returns a pullback closure; rune-next offers both that shape (`vjp_fun`) and a one-shot version (`vjp`) that takes the cotangent directly:

```python
y, pullback = jax.vjp(f, x)
grads = pullback(ct)
```

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let ct = Nx.ones Nx.float32 [| 3 |] in

  (* One-shot: *)
  let y, g = Rune_next.vjp' f x ct in

  (* Reusable pullback, as in JAX: *)
  let y', pullback = Rune_next.vjp_fun' f x in
  let g' = pullback ct in
  ignore (y, g, y', g')
```

Forward mode is nearly identical in both:

```python
y, tangent = jax.jvp(f, (x,), (v,))
```

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = Nx.ones Nx.float32 [| 3 |] in
  let y, tangent = Rune_next.jvp' f x v in
  ignore (y, tangent)
```

For functions returning a structure rather than one tensor, use `vjp2`/`jvp2` with a second module describing the output.

---

## 5. vmap and Per-Sample Gradients

```python
f_batched = jax.vmap(f)                      # map axis 0 of every input
jax.vmap(f, in_axes=(0, None))               # hold the second input fixed
per_sample = jax.vmap(jax.grad(loss))(batch) # per-example gradients
```

`in_axes` translates directly: `Some i` for a mapped axis, `None` for a constant, one entry per leaf in traversal order:

```ocaml
let () =
  let f v = Nx.sum (Nx.mul v v) in
  let batch = Nx.ones Nx.float32 [| 10; 5 |] in
  let results = Rune_next.vmap' f batch in
  ignore results (* shape [10] *)
```

<!-- $MDX skip -->
```ocaml
(* Hold the second leaf fixed: *)
let ys =
  Rune_next.vmap ~in_axes:[ Some 0; None ] (module Pair) f pairs

(* Per-sample gradients: vmap2 of grad. *)
let per_sample =
  Rune_next.vmap2
    (module Example)
    (module Params)
    (fun ex -> Rune_next.grad (module Params) (loss ex) params)
    batch
```

Two honest caveats relative to `jax.vmap`:

- rune-next's `vmap` has no batching rule for the `fft` family and matrix decompositions (`cholesky`, `qr`, `svd`, `eig`, `eigh`); those raise on batched inputs.
- Implicit RNG inside the mapped function draws identical values for every lane (JAX avoids this by making you thread keys; in rune-next, thread randomness in as mapped inputs).

---

## 6. Custom Rules

The correspondence is direct. JAX:

```python
@jax.custom_vjp
def f(x): return jnp.square(x)

def f_fwd(x): return jnp.square(x), x
def f_bwd(res, ct): return (ct * 2 * res,)
f.defvjp(f_fwd, f_bwd)
```

rune-next packs the same three pieces into one call — the forward function returns the residual alongside its result:

```ocaml
module Vec = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) v = f v
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) v = f v
end

let f x =
  Rune_next.custom_vjp
    (module Vec)
    ~fwd:(fun x -> (Nx.square x, x))
    ~bwd:(fun res ct -> Nx.mul ct (Nx.mul_s res 2.0))
    x

let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  Nx.print_data (Rune_next.grad' (fun v -> Nx.sum (f v)) x)
```

`custom_jvp` mirrors `jax.custom_jvp` the same way. One difference to know: in rune-next a `custom_vjp` raises if differentiated in forward mode (and vice versa) — define both rules if you need both modes, where JAX can sometimes transpose a JVP rule automatically.

---

## 7. Gradient Checkpointing

`jax.checkpoint` (a.k.a. `jax.remat`) maps to `remat`:

```python
y = jax.checkpoint(expensive)(x)
```

```ocaml
let () =
  let expensive v = Nx.mean (Nx.square (Nx.sin v)) in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let g =
    Rune_next.grad' (fun v -> Rune_next.remat (module Vec) expensive v) x
  in
  ignore g
```

Both recompute the wrapped function during the backward pass instead of retaining its intermediates; gradients are unchanged.

---

## 8. Control Flow

In JAX, Python control flow breaks under `jit`, so `lax.cond`/`lax.scan`/`lax.while_loop` are mandatory inside compiled functions. In rune-next everything runs eagerly, so ordinary OCaml control flow works inside every transformation:

```ocaml
let () =
  let f x = if Nx.item [] x > 0.0 then x else Nx.neg x in
  ignore (Rune_next.grad' f (Nx.scalar Nx.float32 2.0))
```

Rune-next still provides `scan`, `cond`, and `while_loop` — not because you need them today, but because their signatures are staging-ready: code written with them differentiates and vectorizes now, and a future `jit` can stage them as structured control flow instead of unrolled traces. `lax.scan`'s carry-and-stacked-outputs contract translates directly:

```python
final, ys = jax.lax.scan(f, init, xs)
```

<!-- $MDX skip -->
```ocaml
let final, ys = Rune_next.scan (module Carry) ~f ~init xs
```

---

## 9. Jacobians, Hessians, HVPs

| JAX | rune-next |
| --- | --- |
| `jax.jacfwd(f)(x)` | `jacfwd' f x` |
| `jax.jacrev(f)(x)` | `jacrev' f x` |
| `jax.hessian(f)(x)` | `hessian' f x` |
| `jvp`-of-`grad` HVP recipe | `hvp (module P) f params v` / `hvp' f x v` |

JAX's docs derive the Hessian-vector product as `jvp` of `grad`; rune-next ships that composition as `hvp`, matrix-free, for any parameter structure.

---

## 10. Gradient Checking

```python
from jax.test_util import check_grads
check_grads(f, (x,), order=1)
```

```ocaml
let () =
  let f v = Nx.sum (Nx.mul v v) in
  let module V64 = struct
    type t = Nx.float64_t

    let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) v = f v
    let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
    let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) v = f v
  end in
  match
    Rune_next.check_grads
      (module V64)
      f
      (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |])
  with
  | Ok () -> print_endline "ok"
  | Error msg -> print_endline msg
```

Both compare autodiff against finite differences along directions rather than element by element. Use float64 for reliable results.

---

## 11. Randomness

JAX threads explicit splittable keys. Rune-next uses Nx's implicit scoped RNG: wrap the program in `Nx.Rng.run ~seed` for reproducibility, and `Nx.rand`/`Nx.randn` draw from the ambient scope:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  ignore (Nx.randn Nx.float32 [| 3 |])
```

The trade-off surfaces under `vmap`: with explicit keys you would pass one key per lane; with the implicit scope, in-function draws are identical across lanes, so per-lane randomness must be a mapped input.

---

## 12. What Rune-next Does Not Have (Yet)

| JAX feature | Status in rune-next |
| --- | --- |
| `jax.jit` | Not implemented. Everything runs eagerly; `scan`/`cond`/`while_loop` are designed so a future `jit` can stage them. |
| GPU/TPU, `jax.device_put` | Not implemented. CPU only. |
| `jax.pmap` / distributed | Not implemented. |
| Full op coverage under AD | Reverse mode raises on `svd`, `eig`, `eigh`, `rfft`, `irfft`, `psum`, `mod`; forward mode additionally on `qr`. `detach` inputs where gradients should not flow. |
| Full op coverage under `vmap` | The `fft` family and decompositions raise on batched inputs. |
| `jax.random` keys | Implicit scoped RNG instead; see §11. |
| Donation, sharding, `pjit` | Not applicable without a compiler. |

Rune-next's failure model is deliberate: operations without a rule raise `Invalid_argument` rather than silently producing zero or wrong gradients.

---

## 13. Quick Cheat Sheet

| Task | JAX | rune-next |
| --- | --- | --- |
| Gradient | `jax.grad(f)(params)` | `grad (module P) f params` |
| Gradient (one tensor) | `jax.grad(f)(x)` | `grad' f x` |
| Value + gradient | `jax.value_and_grad(f)(params)` | `value_and_grad (module P) f params` |
| Auxiliary output | `value_and_grad(f, has_aux=True)` | `value_and_grad_aux (module P) f params` |
| Parameter container | pytree registration | `Ptree.S` record + 3 one-line traversals |
| Dynamic tree | pytree | `Rune_next.Ptree.t` |
| VJP | `jax.vjp(f, x)` then call | `vjp_fun (module P) f params` / `vjp'` |
| JVP | `jax.jvp(f, (x,), (v,))` | `jvp (module P) f params v` / `jvp'` |
| Batch map | `jax.vmap(f)(batch)` | `vmap (module P) f batch` / `vmap' f batch` |
| Axis control | `in_axes=(0, None)` | `~in_axes:[ Some 0; None ]` |
| Per-sample grads | `vmap(grad(f))` | `vmap2` of `grad` |
| Custom reverse rule | `@jax.custom_vjp` | `custom_vjp (module P) ~fwd ~bwd` |
| Custom forward rule | `@jax.custom_jvp` | `custom_jvp (module P) ~f ~jvp` |
| Rematerialization | `jax.checkpoint(f)` | `remat (module P) f` |
| Jacobian | `jacfwd` / `jacrev` | `jacfwd'` / `jacrev'` |
| Hessian | `jax.hessian(f)(x)` | `hessian' f x` |
| HVP | `jvp`-of-`grad` recipe | `hvp` / `hvp'` |
| Scan | `jax.lax.scan(f, init, xs)` | `scan (module C) ~f ~init xs` |
| Stop gradient | `jax.lax.stop_gradient(x)` | `detach x` |
| Block region from AD | — | `no_grad (fun () -> ...)` |
| Gradient check | `check_grads(f, (x,), 1)` | `check_grads (module P) f params` |
| Debug tracing | `jax.debug.print` | `with_debug (fun () -> ...)` |
| JIT | `jax.jit(f)` | Not yet available |
