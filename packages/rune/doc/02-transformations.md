# Transformations

Rune provides functional transformations that operate on Nx tensor functions. This guide covers every transformation available.

## Reverse-Mode AD

Reverse-mode AD (backpropagation) is efficient when you have many inputs and a scalar output — the typical case in machine learning.

### grad

`grad f` returns a function that computes the gradient of scalar-valued `f`.

```ocaml
open Nx
open Rune

let () =
  let f x = sum (mul x x) in
  let df = grad f in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  print_data (df x)
  (* gradient: [2. 4. 6.] *)
```

### grads

`grads` differentiates with respect to multiple inputs:

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

### value_and_grad

Computes both the function value and gradient in a single forward-backward pass, avoiding redundant computation:

<!-- $MDX skip -->
```ocaml
let loss, gradient = value_and_grad loss_fn params
```

`value_and_grads` does the same for multiple inputs.

### value_and_grad_aux

When your function returns auxiliary data alongside the loss (e.g., predictions, metrics), use the `_aux` variants to carry it through without differentiating it:

<!-- $MDX skip -->
```ocaml
let f x =
  let pred = forward_pass x in
  let loss = compute_loss pred in
  (loss, pred)  (* pred is auxiliary — not differentiated *)

let loss, gradient, pred = value_and_grad_aux f x
```

`value_and_grads_aux` does the same for multiple inputs.

### vjp

Vector-Jacobian product. Unlike `grad`, the function does not need to return a scalar — you provide a cotangent vector:

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

`vjps` handles multiple inputs.

## Forward-Mode AD

Forward-mode AD propagates tangent vectors alongside primal values. It is efficient when the number of inputs is small relative to the number of outputs.

### jvp

Jacobian-vector product. Provide a tangent vector with the same shape as the input:

```ocaml
open Nx
open Rune

let () =
  let f x = mul x x in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  let v = ones Float32 [|3|] in
  let y, tangent = jvp f x v in
  print_data y;       (* [1. 4. 9.] — primal *)
  print_data tangent  (* [2. 4. 6.] — directional derivative *)
```

`jvps` handles multiple inputs. `jvp_aux` carries auxiliary outputs.

### Choosing Between Forward and Reverse Mode

- **Reverse mode** (`grad`, `vjp`): One backward pass gives gradients for all inputs. Best when outputs << inputs (typical in ML: scalar loss, many parameters).
- **Forward mode** (`jvp`): One forward pass gives one directional derivative. Best when inputs << outputs (e.g., sensitivity analysis with few parameters).

## Stopping Gradients

### no_grad

Evaluate a computation without recording it for differentiation:

<!-- $MDX skip -->
```ocaml
let baseline = no_grad (fun () ->
  mean predictions
)
```

Everything computed inside `no_grad` is treated as a constant by enclosing gradient computations.

### detach

Make a single tensor a constant:

<!-- $MDX skip -->
```ocaml
let target = detach current_value
(* target has the same values but is not differentiated *)
```

## Vectorising Map

### vmap

`vmap` transforms a function that operates on single examples into one that operates on batches:

<!-- $MDX skip -->
```ocaml
(* Function that works on a single vector *)
let f x = sum (mul x x)

(* Automatically batched: maps over axis 0 of the input *)
let f_batched = vmap f

(* Process a batch of 10 vectors at once *)
let batch = rand Float32 [|10; 5|] in
let results = f_batched batch
(* results has shape [|10|] — one scalar per example *)
```

By default, `vmap` maps over axis 0 of inputs and stacks outputs on axis 0. You can customize this:

<!-- $MDX skip -->
```ocaml
(* Map over axis 1 instead *)
let f_axis1 = vmap ~in_axes:(Single (Map 1)) f

(* Don't map an input (broadcast it) *)
let f_shared = vmap ~in_axes:(Single NoMap) f
```

`vmaps` handles functions with multiple inputs, with per-input axis specifications.

### Composing vmap with grad

Since transformations are composable, you can compute per-example gradients:

<!-- $MDX skip -->
```ocaml
(* Per-example gradient (no manual batching needed) *)
let per_example_grad = vmap (grad loss_fn)
```

## Gradient Checking

Rune provides utilities for verifying that autodiff gradients are correct by comparing them against finite-difference approximations.

### finite_diff

Approximate the gradient using finite differences:

```ocaml
open Nx
open Rune

let () =
  let f x = sum (mul x x) in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  let fd_grad = finite_diff f x in
  let ad_grad = grad f x in
  print_data fd_grad;
  print_data ad_grad
  (* both approximately [2. 4. 6.] *)
```

The default method is central differences (`(f(x+h) - f(x-h)) / 2h`). You can choose `Forward` or `Backward` methods and adjust `eps` (default `1e-4`).

### check_gradient

Automated comparison of autodiff vs finite-difference gradients:

<!-- $MDX skip -->
```ocaml
match check_gradient ~verbose:true my_function x with
| `Pass result -> Printf.printf "max error: %e\n" result.max_abs_error
| `Fail result ->
  Printf.printf "%d of %d elements failed\n"
    result.num_failed result.num_checked
```

`check_gradients` handles functions with multiple inputs.

## Debugging

### debug

Print every tensor operation as it executes:

<!-- $MDX skip -->
```ocaml
let () =
  let f x = add (mul x x) (sin x) in
  let x = scalar Float32 2.0 in
  let _ = debug f x in
  ()
(* Prints each operation, its inputs, and its output *)
```

This is useful for understanding what operations a function performs, especially when debugging unexpected gradients.

## Summary

| Transform | Purpose | When to use |
|-----------|---------|-------------|
| `grad` | Gradient of scalar function | Training loss → parameter gradients |
| `value_and_grad` | Value + gradient together | Avoid duplicate forward pass |
| `vjp` | Vector-Jacobian product | Non-scalar outputs |
| `jvp` | Jacobian-vector product | Few inputs, many outputs |
| `vmap` | Vectorise over a batch dimension | Per-example computation |
| `no_grad` / `detach` | Stop gradient propagation | Baselines, targets, constants |
| `check_gradient` | Verify gradient correctness | Testing custom operations |
| `debug` | Trace all operations | Understanding/debugging |
