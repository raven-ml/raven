# Rune vs. JAX -- A Practical Comparison

This guide explains how Rune's functional transformations relate to [JAX](https://jax.readthedocs.io/), focusing on:

* How core concepts map (grad, vjp, jvp, vmap)
* Where the APIs feel similar vs. deliberately different
* How to translate common JAX patterns into Rune

If you already use JAX, this should be enough to become productive in Rune quickly.

---

## 1. Big-Picture Differences

| Aspect            | JAX (Python)                                              | Rune (OCaml)                                          |
| ----------------- | --------------------------------------------------------- | ----------------------------------------------------- |
| Language          | Dynamic, interpreted                                      | Statically typed, compiled                            |
| Array type        | `jax.Array`                                               | `Nx.t` (no separate Rune tensor type)                 |
| Array library     | `jax.numpy`                                               | Nx                                                    |
| AD mechanism      | Tracing + XLA compilation                                 | OCaml 5 effect handlers                               |
| Reverse-mode AD   | `jax.grad`, `jax.value_and_grad`                          | `grad`, `value_and_grad`, `grads`, `value_and_grads`  |
| Forward-mode AD   | `jax.jvp`                                                 | `jvp`, `jvps`                                         |
| VJP               | `jax.vjp`                                                 | `vjp`, `vjps`                                         |
| Vectorising map   | `jax.vmap`                                                | `vmap`, `vmaps`                                       |
| JIT compilation   | `jax.jit`                                                 | Not yet implemented                                   |
| Device placement  | `jax.device_put`, device kwarg                            | Not yet implemented                                   |
| Gradient stopping | `jax.lax.stop_gradient`                                   | `no_grad`, `detach`                                   |
| Gradient checking | `jax.test_util.check_grads`                               | `check_gradient`, `check_gradients`                   |
| Debugging         | `jax.debug.print`                                         | `debug`                                               |
| Control flow      | Restricted inside `jit` (requires `lax.cond`, `lax.scan`) | Full OCaml control flow (if, match, loops, recursion) |
| Mutability        | Immutable arrays; functional updates                      | Immutable Nx tensors; same model                      |

**Key things to know:**
- Rune operates on `Nx.t` directly. There is no separate tensor type, no `rune.numpy`, and no tracing step.
- Because Rune uses effect handlers rather than tracing, ordinary OCaml control flow works inside differentiated functions. No need for `lax.cond` or `lax.scan`.
- JIT compilation and device/GPU placement do not exist yet. All computation runs eagerly on CPU via the Nx C backend.

---

## 2. Reverse-Mode AD (grad)

### 2.1 Basic gradient

**JAX**

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

grad_f = jax.grad(f)
x = jnp.array([1.0, 2.0, 3.0])
print(grad_f(x))  # [2. 4. 6.]
```

**Rune**

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let () =
  let f x = sum (mul x x) in
  let grad_f = grad f in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  print_data (grad_f x)
  (* [2. 4. 6.] *)
```

Both `jax.grad` and `Rune.grad` take a function and return a new function that computes the gradient. The input function must return a scalar.

### 2.2 Value and gradient

**JAX**

```python
loss, grads = jax.value_and_grad(loss_fn)(params)
```

**Rune**

<!-- $MDX skip -->
```ocaml
let loss, gradient = value_and_grad loss_fn params
```

Both avoid computing the forward pass twice.

### 2.3 Multiple inputs

**JAX**

```python
def f(x, y):
    return jnp.sum(x ** 2 + y ** 2)

# argnums selects which arguments to differentiate
dx, dy = jax.grad(f, argnums=(0, 1))(x, y)
```

**Rune**

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let () =
  let f inputs =
    match inputs with
    | [x; y] -> sum (add (mul x x) (mul y y))
    | _ -> assert false
  in
  let gs = grads f [scalar Float32 3.0; scalar Float32 4.0] in
  List.iter (fun g -> Printf.printf "%.1f " (item [] g)) gs
  (* 6.0 8.0 *)
```

JAX uses `argnums` to select which positional arguments to differentiate. Rune takes a function of `Nx.t list` and differentiates with respect to all inputs. `value_and_grads` combines both:

<!-- $MDX skip -->
```ocaml
let loss, gradients = value_and_grads loss_fn [w; b]
```

### 2.4 Auxiliary outputs

**JAX**

```python
def f(x):
    pred = model(x)
    loss = compute_loss(pred)
    return loss, pred  # pred is auxiliary

(loss, pred), grads = jax.value_and_grad(f, has_aux=True)(x)
```

**Rune**

<!-- $MDX skip -->
```ocaml
let f x =
  let pred = forward_pass x in
  let loss = compute_loss pred in
  (loss, pred)  (* pred is auxiliary -- not differentiated *)

let loss, gradient, pred = value_and_grad_aux f x
```

JAX uses a `has_aux=True` flag. Rune has dedicated `_aux` variants: `value_and_grad_aux` and `value_and_grads_aux`.

### 2.5 Higher-order derivatives

**JAX**

```python
f = lambda x: x ** 4
f_prime = jax.grad(f)
f_double_prime = jax.grad(f_prime)
```

**Rune**

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let () =
  let f x = mul x (mul x (mul x x)) in
  let f' = grad f in
  let f'' = grad f' in
  let f''' = grad f'' in
  let x = scalar Float32 2.0 in
  Printf.printf "f(2)    = %.1f\n" (item [] (f x));
  Printf.printf "f'(2)   = %.1f\n" (item [] (f' x));
  Printf.printf "f''(2)  = %.1f\n" (item [] (f'' x));
  Printf.printf "f'''(2) = %.1f\n" (item [] (f''' x))
```

Both compose naturally because `grad` returns an ordinary function.

---

## 3. VJP (Vector-Jacobian Product)

**JAX**

```python
def f(x):
    return x ** 2

primals, vjp_fn = jax.vjp(f, x)
grads = vjp_fn(v)
```

**Rune**

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let () =
  let f x = mul x x in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  let v = ones Float32 [|3|] in
  let y, g = vjp f x v in
  print_data y;  (* [1. 4. 9.] *)
  print_data g   (* [2. 4. 6.] *)
```

In JAX, `jax.vjp` returns a closure `vjp_fn` that you call with the cotangent. In Rune, `vjp f x v` takes the cotangent `v` directly and returns `(y, g)` in one call.

For multiple inputs, JAX still uses positional arguments while Rune uses `vjps` with a list:

<!-- $MDX skip -->
```ocaml
let y, gs = vjps f [x1; x2] v
```

---

## 4. Forward-Mode AD (JVP)

**JAX**

```python
def f(x):
    return x ** 2

primals, tangents = jax.jvp(f, (x,), (v,))
```

**Rune**

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let () =
  let f x = mul x x in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  let v = ones Float32 [|3|] in
  let y, tangent = jvp f x v in
  print_data y;       (* [1. 4. 9.] -- primal *)
  print_data tangent  (* [2. 4. 6.] -- directional derivative *)
```

The API shape is nearly identical. JAX takes tuples of primals and tangents; Rune takes them as separate arguments.

For multiple inputs:

<!-- $MDX skip -->
```ocaml
let y, tangent = jvps f [x1; x2] [v1; v2]
```

`jvp_aux` carries auxiliary outputs:

<!-- $MDX skip -->
```ocaml
let y, tangent, aux = jvp_aux f x v
```

---

## 5. Stopping Gradients

**JAX**

```python
import jax.lax

def f(x):
    baseline = jax.lax.stop_gradient(running_mean)
    return loss(x) - baseline
```

**Rune**

There are two options:

<!-- $MDX skip -->
```ocaml
(* Option 1: detach a single tensor *)
let baseline = detach running_mean

(* Option 2: block an entire computation *)
let baseline = no_grad (fun () ->
  mean predictions
)
```

JAX has a single `stop_gradient` that operates on arrays. Rune offers two mechanisms:
- `detach x` returns a copy of `x` that is treated as a constant during differentiation. Closest to `jax.lax.stop_gradient`.
- `no_grad f` runs `f ()` without recording any operations. Useful when a whole sub-computation should be excluded.

---

## 6. Vectorising Map (vmap)

### 6.1 Basic usage

**JAX**

```python
def f(x):
    return jnp.sum(x ** 2)

f_batched = jax.vmap(f)
batch = jnp.ones((10, 5))
results = f_batched(batch)  # shape (10,)
```

**Rune**

<!-- $MDX skip -->
```ocaml
let f x = sum (mul x x) in
let f_batched = vmap f in
let batch = ones Float32 [|10; 5|] in
let results = f_batched batch
(* results has shape [|10|] *)
```

Both map over axis 0 by default and stack outputs on axis 0.

### 6.2 Axis specifications

**JAX**

```python
# Map over axis 1
jax.vmap(f, in_axes=1)

# Don't map an input (broadcast it)
jax.vmap(f, in_axes=(0, None))
```

**Rune**

<!-- $MDX skip -->
```ocaml
(* Map over axis 1 *)
let f_axis1 = vmap ~in_axes:(Single (Map 1)) f

(* Don't map an input (broadcast it) *)
let f_shared =
  vmaps
    ~in_axes:[Map 0; NoMap]
    f_multi
```

JAX uses `None` to indicate a non-mapped input and integers for mapped axes. Rune uses `Map n` and `NoMap` constructors. For single-input functions, wrap in `Single`; for multi-input, use `vmaps` with a list.

Output axis control:

<!-- $MDX skip -->
```ocaml
(* Stack outputs along axis 1 instead of 0 *)
let f' = vmap ~out_axes:(OutSingle (Some 1)) f

(* Discard the batch dimension (e.g., for reductions) *)
let f' = vmap ~out_axes:(OutSingle None) f
```

### 6.3 Composing vmap with grad

**JAX**

```python
# Per-example gradients
per_example_grad = jax.vmap(jax.grad(loss_fn))
```

**Rune**

<!-- $MDX skip -->
```ocaml
let per_example_grad = vmap (grad loss_fn)
```

Both compose naturally. This gives per-example gradients without writing batch loops.

---

## 7. Gradient Checking

**JAX**

```python
from jax._src import test_util as jtu

jtu.check_grads(f, (x,), order=1)
```

**Rune**

<!-- $MDX skip -->
```ocaml
match check_gradient ~verbose:true f x with
| `Pass result -> Printf.printf "max error: %e\n" result.max_abs_error
| `Fail result ->
  Printf.printf "%d of %d elements failed\n"
    result.num_failed result.num_checked
```

Rune provides more detailed results. The `gradient_check_result` record includes:
- `max_abs_error`, `max_rel_error`, `mean_abs_error`, `mean_rel_error`
- `failed_indices` with per-element `(index, autodiff_value, finite_diff_value, abs_error)`
- `passed`, `num_checked`, `num_failed`

Additional utilities:
- `finite_diff f x` -- approximate gradient via finite differences
- `finite_diff_jacobian f x` -- approximate Jacobian for non-scalar outputs
- `check_gradients f xs` -- check a multi-input function

You can control the finite-difference method:

<!-- $MDX skip -->
```ocaml
let fd = finite_diff ~method_:`Forward ~eps:1e-5 f x
```

Available methods: `` `Central `` (default), `` `Forward ``, `` `Backward ``.

---

## 8. Debugging

**JAX**

```python
def f(x):
    y = x ** 2
    jax.debug.print("y = {}", y)
    return y

f(jnp.array(3.0))
```

**Rune**

<!-- $MDX skip -->
```ocaml
let f x = add (mul x x) (sin x) in
let x = scalar Float32 2.0 in
let _ = debug f x in
()
(* Prints each operation, its inputs, and its output *)
```

JAX's `debug.print` is a targeted print inside traced code. Rune's `debug` wraps an entire function and traces every tensor operation, printing the operation name, inputs, and output. It is more coarse-grained but requires no instrumentation inside the function.

---

## 9. Control Flow

This is a fundamental difference.

**JAX**

Inside `jit`-compiled functions, Python control flow does not work because JAX traces the function:

```python
# Breaks under jit:
@jax.jit
def f(x):
    if x > 0:  # Error: traced value used in Python conditional
        return x
    else:
        return -x

# Must use JAX primitives:
@jax.jit
def f(x):
    return jax.lax.cond(x > 0, lambda: x, lambda: -x)
```

**Rune**

OCaml control flow works everywhere, including inside `grad`, `jvp`, and `vmap`:

<!-- $MDX skip -->
```ocaml
let f x =
  if item [] x > 0.0 then x
  else neg x

(* Works fine *)
let df = grad f
```

Rune does not trace functions into a graph. It intercepts operations as they execute via effect handlers, so any OCaml expression is valid. No special `cond`, `scan`, or `while_loop` primitives are needed.

---

## 10. What Rune Does Not Have (Yet)

| JAX feature                                  | Status in Rune                                      |
| -------------------------------------------- | --------------------------------------------------- |
| `jax.jit`                                    | Not implemented. All operations execute eagerly.    |
| Device placement (`jax.device_put`, GPU/TPU) | Not implemented. All computation runs on CPU.       |
| `jax.pmap` / distributed                     | Not implemented.                                    |
| `jax.lax.scan`, `jax.lax.while_loop`         | Not needed. Use ordinary OCaml loops and recursion. |
| `jax.custom_vjp`, `jax.custom_jvp`           | Not yet exposed.                                    |
| `jax.checkpoint` (gradient checkpointing)    | Not implemented.                                    |
| Pytrees / tree utilities                     | Not needed. Use OCaml data structures directly.     |
| `jax.random` (splittable PRNG)               | Use `Nx.rand`, `Nx.randn` directly.                 |

---

## 11. Quick Cheat Sheet

| Task                  | JAX                                      | Rune                              |
| --------------------- | ---------------------------------------- | --------------------------------- |
| Gradient of scalar fn | `jax.grad(f)(x)`                         | `grad f x`                        |
| Value + gradient      | `jax.value_and_grad(f)(x)`               | `value_and_grad f x`              |
| Multi-input gradient  | `jax.grad(f, argnums=(0,1))(x, y)`       | `grads f [x; y]`                  |
| Auxiliary output      | `jax.value_and_grad(f, has_aux=True)(x)` | `value_and_grad_aux f x`          |
| Higher-order deriv    | `jax.grad(jax.grad(f))`                  | `grad (grad f)`                   |
| VJP                   | `primals, fn = jax.vjp(f, x); fn(v)`     | `vjp f x v`                       |
| JVP                   | `jax.jvp(f, (x,), (v,))`                 | `jvp f x v`                       |
| Stop gradient         | `jax.lax.stop_gradient(x)`               | `detach x`                        |
| Block region from AD  | (no direct equivalent)                   | `no_grad (fun () -> ...)`         |
| Batch map             | `jax.vmap(f)(batch)`                     | `vmap f batch`                    |
| vmap axis control     | `jax.vmap(f, in_axes=(0, None))`         | `vmaps ~in_axes:[Map 0; NoMap] f` |
| Per-example grad      | `jax.vmap(jax.grad(f))`                  | `vmap (grad f)`                   |
| Gradient check        | `jtu.check_grads(f, (x,), 1)`            | `check_gradient f x`              |
| Finite differences    | (manual)                                 | `finite_diff f x`                 |
| Debug tracing         | `jax.debug.print(...)`                   | `debug f x`                       |
| JIT compilation       | `jax.jit(f)`                             | Not yet available                 |
| GPU placement         | `jax.device_put(x, gpu)`                 | Not yet available                 |
