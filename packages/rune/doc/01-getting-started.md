# Getting Started

This guide shows you how to compute gradients and use Rune's transformations.

## Installation

<!-- $MDX skip -->
```bash
opam install rune
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build rune
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries rune))
```

## Your First Gradient

Rune operates on Nx tensors directly. Write a function using Nx operations, then use `grad` to get its derivative:

```ocaml
open Nx
open Rune

let () =
  (* A simple function: f(x) = x² + sin(x) *)
  let f x = add (mul x x) (sin x) in

  (* grad returns a function that computes the derivative *)
  let f' = grad f in

  let x = scalar Float32 2.0 in
  Printf.printf "f(2)  = %.4f\n" (item [] (f x));
  Printf.printf "f'(2) = %.4f\n" (item [] (f' x))
  (* f'(x) = 2x + cos(x), so f'(2) ≈ 3.5839 *)
```

Key points:
- `grad f` takes a function `f : Nx.t -> Nx.t` and returns a new function that computes the gradient
- The input function must return a scalar tensor
- The gradient has the same shape as the input

## Value and Gradient Together

In practice, you usually want both the function value and its gradient. Use `value_and_grad` to avoid computing the forward pass twice:

```ocaml
open Nx
open Rune

let () =
  let f x = mean (mul x x) in
  let x = create Float32 [|3|] [|1.0; 2.0; 3.0|] in
  let value, gradient = value_and_grad f x in
  Printf.printf "f(x) = %.4f\n" (item [] value);
  print_data gradient
```

## Multiple Inputs

When your function takes multiple inputs, use `grads` or `value_and_grads`:

```ocaml
open Nx
open Rune

let () =
  let f inputs =
    match inputs with
    | [x; y] -> add (mul x x) (mul y y)
    | _ -> failwith "expected 2 inputs"
  in
  let df = grads f in
  match df [scalar Float32 3.0; scalar Float32 4.0] with
  | [dx; dy] ->
    Printf.printf "df/dx = %.1f\n" (item [] dx);
    Printf.printf "df/dy = %.1f\n" (item [] dy)
  | _ -> assert false
```

## Higher-Order Derivatives

Since `grad` returns a regular function, you can differentiate again:

```ocaml
open Nx
open Rune

let () =
  (* f(x) = x⁴ *)
  let f x = mul x (mul x (mul x x)) in
  let f' = grad f in        (* 4x³ *)
  let f'' = grad f' in      (* 12x² *)
  let f''' = grad f'' in    (* 24x *)
  let x = scalar Float32 2.0 in
  Printf.printf "f(2)    = %.1f\n" (item [] (f x));
  Printf.printf "f'(2)   = %.1f\n" (item [] (f' x));
  Printf.printf "f''(2)  = %.1f\n" (item [] (f'' x));
  Printf.printf "f'''(2) = %.1f\n" (item [] (f''' x))
```

## Stopping Gradients

Sometimes you need part of a computation to be treated as a constant:

<!-- $MDX skip -->
```ocaml
open Rune

(* no_grad: nothing inside is recorded *)
let baseline = no_grad (fun () ->
  (* compute a baseline value that should not be differentiated *)
  mean predictions
)

(* detach: make a single tensor a constant *)
let target = detach current_prediction
```

## A Simple Training Loop

Here is a minimal example that trains a linear model with gradient descent:

<!-- $MDX skip -->
```ocaml
open Nx
open Rune

let () =
  (* Data: y = 2x + 1 *)
  let x_data = create Float32 [|4; 1|] [|1.; 2.; 3.; 4.|] in
  let y_data = create Float32 [|4; 1|] [|3.; 5.; 7.; 9.|] in

  (* Parameters *)
  let w = rand Float32 [|1; 1|] in
  let b = zeros Float32 [|1|] in

  let loss_fn params =
    match params with
    | [w; b] ->
      let pred = add (matmul x_data w) b in
      mean (mul (sub pred y_data) (sub pred y_data))
    | _ -> assert false
  in

  let lr = scalar Float32 0.01 in
  for epoch = 1 to 200 do
    let loss, gs = value_and_grads loss_fn [w; b] in
    match gs with
    | [gw; gb] ->
      ignore (sub ~out:w w (mul lr gw));
      ignore (sub ~out:b b (mul lr gb));
      if epoch mod 50 = 0 then
        Printf.printf "epoch %d  loss %.6f\n" epoch (item [] loss)
    | _ -> assert false
  done;
  Printf.printf "w = %.3f  b = %.3f\n" (item [0; 0] w) (item [0] b)
```

For real neural networks, use [Kaun](/docs/kaun/) which provides layers, optimizers, and training loops built on top of Rune.

## Next Steps

- [Transformations](/docs/rune/transformations/) — complete guide to grad, jvp, vmap, and more
- [How It Works](/docs/rune/how-it-works/) — how effects-based autodiff works under the hood
- [Kaun Getting Started](/docs/kaun/getting-started/) — high-level neural network training
