# Rune vs. JAX — A Practical Comparison

This guide explains how rune's transformations relate to [JAX](https://docs.jax.dev/), focusing on:

* How core concepts map (grad, vjp, jvp, vmap, pytrees, custom rules, remat)
* Where the APIs feel similar vs. deliberately different
* What rune does not have yet, stated honestly

If you already use JAX, this should be enough to become productive in rune quickly.

---

## 1. Big-Picture Differences

| Aspect | JAX (Python) | rune (OCaml) |
| --- | --- | --- |
| Language | Dynamic, interpreted | Statically typed, compiled |
| Array type | `jax.Array` | `Nx.t` (no separate rune tensor type) |
| Array library | `jax.numpy` | Nx |
| AD mechanism | Tracing + XLA compilation | Interpreters of Nx's operations, eager; reverse mode transposes forward mode |
| Parameter containers | Pytrees (registered runtime trees) | `Nx.Ptree.S` — your own typed records |
| Reverse mode | `jax.grad`, `jax.value_and_grad` | `grad`, `value_and_grad`, `value_and_grad_aux` |
| VJP | `jax.vjp` | `vjp` (result and pullback) |
| Forward mode | `jax.jvp` | `jvp` |
| Vectorizing map | `jax.vmap` | `vmap`, `vmap'` |
| Custom rules | `jax.custom_vjp`, `jax.custom_jvp` | `custom_jvp` (serves both modes), `custom_vjp` |
| Checkpointing | `jax.checkpoint` / `jax.remat` | `remat` |
| Jacobians / Hessians | `jacfwd`, `jacrev`, `hessian` | `jacfwd'`, `jacrev'`; a Hessian is `jacfwd' (grad' f)` |
| Control flow | `lax.scan`, `lax.cond`, `lax.while_loop` (required under `jit`) | `scan` (optional, compiled as a loop) plus ordinary OCaml control flow |
| Gradient stopping | `jax.lax.stop_gradient` | `detach` |
| Gradient checking | `jax.test_util.check_grads` | `check_grads` |
| Randomness | Typed splittable keys (`jax.random.key`) | Keys of a private type (`Nx.Rng.t`), or a scope (`Nx.Rng.with_key`) |
| JIT compilation | `jax.jit` | `jit` — traces once per key (every tensor's path, dtype and shape, and what its structure reports); CPU, CUDA, or Metal |
| Devices | `jax.device_put`, GPU/TPU | `Nx.place` with `Rune.device`; compiled functions run on CPU, CUDA, Metal |

---

## 2. Pytrees → Nx.Ptree

This is the deepest difference. JAX flattens arbitrary registered containers into lists of leaves at runtime:

```python
import jax

params = {"w": w, "b": b}          # any registered pytree
grads = jax.grad(loss)(params)      # same pytree of gradients
```

Rune has no runtime tree. A parameter structure is a record you define, and an `Nx.Ptree.S` module walks it with one function, `walk`, a line per field; no ppx, no registration table:

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
```

Every transformation takes the structure and preserves the type: the gradient of a function of `params` *is* a `params`. `Nx.Ptree.map`, `map2` and `fold` are `tree_map` and its kin, and they pass each tensor its path (`w`, `b`), which is also its checkpoint name.

Where JAX distinguishes leaves by position in a flattened list, rune tensors keep their record field names, dtypes, and shapes in the type. Mixed dtypes work: a single backward pass produces a gradient for every leaf, each with its leaf's dtype.

---

## 3. Reverse Mode

**JAX**

```python
def loss(params):
    return jnp.mean((x @ params["w"] + params["b"] - y) ** 2)

grads = jax.grad(loss)(params)
loss_value, grads = jax.value_and_grad(loss)(params)
```

**rune**

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
  let grads = Rune.grad params_ptree loss params in
  let loss_value, grads' = Rune.value_and_grad params_ptree loss params in
  ignore (grads, loss_value, grads')
```

Both require a scalar output. JAX's `argnums` has no equivalent: differentiate with respect to *the* parameter structure and close over everything else. For a function of one tensor, `grad'` takes no structure.

### Auxiliary outputs

JAX uses a flag; rune has a dedicated function that takes the auxiliary value's structure:

```python
(loss, aux), grads = jax.value_and_grad(f, has_aux=True)(params)
```

<!-- $MDX skip -->
```ocaml
let loss, grads, aux = Rune.value_and_grad_aux params_ptree aux_ptree f params
```

---

## 4. VJP and JVP

JAX's `jax.vjp` returns a pullback closure, and so does rune's:

```python
y, pullback = jax.vjp(f, x)
grads = pullback(ct)
```

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let ct = Nx.ones Nx.float32 [| 3 |] in
  let y, pullback = Rune.vjp' f x in
  let g = pullback ct in
  ignore (y, g)
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
  let y, tangent = Rune.jvp' f x v in
  ignore (y, tangent)
```

`vjp` and `jvp` take the structure of the result beside that of the parameters, so a function returning a structure takes one cotangent or tangent per tensor of its result.

---

## 5. vmap and Per-Sample Gradients

```python
f_batched = jax.vmap(f)                      # map axis 0 of every input
jax.vmap(f, in_axes=(0, None))               # hold the second input fixed
per_sample = jax.vmap(jax.grad(loss))(batch) # per-example gradients
```

Rune's `vmap` maps axis 0 of every tensor of the arguments its signature lists; a value held fixed is captured, and another axis is moved to the front with `Nx.moveaxis`:

```ocaml
let () =
  let f v = Nx.sum (Nx.mul v v) in
  let batch = Nx.ones Nx.float32 [| 10; 5 |] in
  let results = Rune.vmap' f batch in
  ignore results (* shape [10] *)
```

<!-- $MDX skip -->
```ocaml
(* Hold the second input fixed by capturing it: *)
let ys = Rune.vmap Nx.Ptree.(tensor @-> returns tensor) (fun x -> f x y) xs

(* Per-sample gradients: vmap of grad. *)
let per_sample =
  Rune.vmap
    Nx.Ptree.(tensor @-> tensor @-> returns params_ptree)
    (fun x y -> Rune.grad params_ptree (loss x y) params)
    xs ys
```

Two honest caveats relative to `jax.vmap`:

- rune's `vmap` has no batching rule for the matrix decompositions (`cholesky`, `qr`, `svd`, `eig`, `eigh`); those raise on batched inputs.
- Implicit RNG inside the mapped function draws identical values for every lane (JAX avoids this by making you thread keys; in rune, thread randomness in as mapped inputs).

---

## 6. Custom Rules

A JAX `custom_vjp` takes a forward function returning residuals and a backward function receiving them:

```python
@jax.custom_vjp
def f(x): return jnp.square(x)

def f_fwd(x): return jnp.square(x), x
def f_bwd(res, ct): return (ct * 2 * res,)
f.defvjp(f_fwd, f_bwd)
```

rune's rule is one function returning the result and its pullback; the pullback's closure holds what JAX calls residuals:

```ocaml
let f x =
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
    (fun x -> (Nx.square x, fun ct -> Nx.mul ct (Nx.mul_s x 2.0)))
    x

let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  Printf.printf "%s\n"
    (Nx.to_string (Rune.grad' (fun v -> Nx.sum (f v)) x))
```

A `custom_jvp` rule returns the result and its tangent map. As in JAX, reverse mode transposes the map, so one rule serves `jvp` and `grad`, and every enclosing differentiation applies it too. A `custom_vjp` whose result holds a tensor raises under `jvp`: give the function a `custom_jvp` rule if it needs forward mode.

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
    Rune.grad' (Rune.remat Nx.Ptree.(tensor @-> returns tensor) expensive) x
  in
  ignore g
```

Both recompute the wrapped function during the backward pass instead of retaining its intermediates; gradients are unchanged.

---

## 8. Control Flow

In JAX, Python control flow breaks under `jit`, so `lax.cond`/`lax.scan`/`lax.while_loop` are mandatory inside compiled functions. In rune everything runs eagerly, so ordinary OCaml control flow works inside every transformation:

```ocaml
let () =
  let f x = if Nx.item [] x > 0.0 then x else Nx.neg x in
  ignore (Rune.grad' f (Nx.scalar Nx.float32 2.0))
```

Rune still provides `scan` because it gives a loop a structure the compiler can see: `jit` compiles `scan` as a loop, forward and reverse, and rejects a branch on a traced value. `lax.scan`'s carry-and-stacked-outputs contract translates directly, with one structure each for the carry, the rows and the outputs, where JAX infers pytrees:

```python
final, ys = jax.lax.scan(f, init, xs)
```

<!-- $MDX skip -->
```ocaml
let final, ys = Rune.scan' ~f ~init xs (* single tensors *)
let final, ys = Rune.scan carry rows outputs ~f ~init xs
```

---

## 9. Jacobians, Hessians, HVPs

| JAX | rune |
| --- | --- |
| `jax.jacfwd(f)(x)` | `jacfwd' f x` |
| `jax.jacrev(f)(x)` | `jacrev' f x` |
| `jax.hessian(f)(x)` | `jacfwd' (grad' f) x` |
| `jvp`-of-`grad` HVP recipe | `snd (jvp p p (grad p f) params v)` / `snd (jvp' (grad' f) x v)` |

Both are compositions: the Hessian is forward mode over the gradient, and the Hessian-vector product is matrix-free.

---

## 10. Gradient Checking

```python
from jax.test_util import check_grads
check_grads(f, (x,), order=1)
```

```ocaml
let () =
  let f v = Nx.sum (Nx.mul v v) in
  match
    Rune.check_grads Nx.Ptree.tensor f
      (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |])
  with
  | Ok () -> print_endline "ok"
  | Error msg -> print_endline msg
```

Both compare autodiff against finite differences along directions rather than element by element. Use float64 for reliable results.

---

## 11. Randomness

JAX threads typed splittable keys, and so does Nx: `Nx.Rng.t` is a private type that only `Nx.Rng` builds, so arithmetic on a key does not type-check. `Nx.Rng.split` and `Nx.Rng.fold_in` derive keys, and each distribution has a keyed sampler. Nx adds a scope: inside `Nx.Rng.with_key`, `Nx.rand` and `Nx.randn` draw successive subkeys of its root.

```ocaml
let () =
  let keys = Nx.Rng.split (Nx.Rng.key 0) in
  ignore (Nx.Rng.normal keys.(0) Nx.float32 [| 3 |]);
  Nx.Rng.with_key keys.(1) @@ fun () -> ignore (Nx.randn Nx.float32 [| 3 |])
```

Under `vmap`, the counterpart of mapping over `jax.random.split(key, n)` is mapping over `Nx.Rng.split_batch ~n key`, walked with `Nx.Rng.ptree`; each lane sees one key. A key or a scope that the mapped function closes over draws identical values in every lane.

```ocaml
let () =
  let noise =
    Rune.vmap
      Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Nx.Rng.normal k Nx.float32 [| 3 |])
      (Nx.Rng.split_batch ~n:4 (Nx.Rng.key 0))
  in
  assert (Nx.shape noise = [| 4; 3 |])
```

---

## 12. What Rune Does Not Have (Yet)

| JAX feature | Status in rune |
| --- | --- |
| `jax.jit` | `jit s f` compiles to fused kernels for the signature `s`, cached per key: each tensor's path, dtype and shape, and the data the structures report (a window, a list's length). It compiles `scan` as a loop and rejects a branch on a traced value. |
| GPU/TPU, `jax.device_put` | Eager execution is CPU-only; `jit ~devices:[ Rune.device "CUDA" ]` (or `"METAL"`) runs compiled steps on GPU. `Nx.place (Nx.Placement.device (Rune.device "METAL"))` holds a tensor's bytes on a device, and a compiled function that captures it uses that buffer with no upload. |
| `jax.pmap` / distributed | `jit` over values placed on several devices, as `jax.jit` over sharded inputs: the function sees whole values and a reduction over a split axis is an allreduce. A per-device computation is `vmap` over an axis split one slice per device; there is no `shard_map`. |
| Full op coverage under AD | Every operation has a forward rule and a batching rule. The tangents of a complete SVD of a non-square matrix and of a complete QR factorisation of a tall one raise; `detach` inputs where derivatives should not flow. |
| `jax.random` keys | Implicit scoped RNG instead; see §11. |
| Donation, sharding, `pjit` | An argument marked `consumes` in `jit`'s signature is given up by the call, like `donate_argnums`, and results reuse its storage; a consumed value raises on use. Sharding is a placement (`Nx.Placement.sharded`), and operands split differently raise instead of being resharded. |

Rune's failure model is deliberate: a derivative with no definition raises `Invalid_argument` rather than silently producing zero or wrong gradients.

---

## 13. Quick Cheat Sheet

| Task | JAX | rune |
| --- | --- | --- |
| Gradient | `jax.grad(f)(params)` | `grad p f params` |
| Gradient (one tensor) | `jax.grad(f)(x)` | `grad' f x` |
| Value + gradient | `jax.value_and_grad(f)(params)` | `value_and_grad p f params` |
| Auxiliary output | `value_and_grad(f, has_aux=True)` | `value_and_grad_aux p aux f params` |
| Parameter container | pytree registration | `Nx.Ptree.S` record with one `walk` |
| VJP | `jax.vjp(f, x)` then call | `vjp p q f params` / `vjp' f x`, then call |
| JVP | `jax.jvp(f, (x,), (v,))` | `jvp p q f params v` / `jvp'` |
| Batch map | `jax.vmap(f)(batch)` | `vmap Nx.Ptree.(tensor @-> returns tensor) f batch` / `vmap' f batch` |
| Axis control | `in_axes=(0, None)` | capture the fixed input; `Nx.moveaxis` another axis to 0 |
| Per-sample grads | `vmap(grad(f))` | `vmap` of `grad` |
| Custom reverse rule | `@jax.custom_vjp` | `custom_vjp p q (fun x -> (y, pullback))` |
| Custom forward rule | `@jax.custom_jvp` | `custom_jvp p q (fun x -> (y, tangent_map))` |
| Rematerialization | `jax.checkpoint(f)` | `remat s f`, `s` the signature of `f` |
| Jacobian | `jacfwd` / `jacrev` | `jacfwd'` / `jacrev'` |
| Hessian | `jax.hessian(f)(x)` | `jacfwd' (grad' f) x` |
| HVP | `jvp`-of-`grad` recipe | `snd (jvp' (grad' f) x v)` |
| Scan | `jax.lax.scan(f, init, xs)` | `scan' ~f ~init xs`, or `scan c x y ~f ~init xs` over structures |
| Stop gradient | `jax.lax.stop_gradient(x)` | `detach x` |
| Gradient check | `check_grads(f, (x,), 1)` | `check_grads p f params` |
| Debug tracing | `jax.debug.print` | an `Nx.Op.intercept` interpreter that prints each operation |
| JIT | `jax.jit(f)` | `jit Nx.Ptree.(p @-> returns q) f` |
