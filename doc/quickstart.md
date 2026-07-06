# Quickstart

This gets you from zero to computing gradients and training a model in five minutes.

## Setup

<!-- $MDX skip -->
```bash
opam install raven
```

Create a `dune-project` and `dune` file:

<!-- $MDX skip -->
```dune
; dune-project
(lang dune 3.20)
```

<!-- $MDX skip -->
```dune
; dune
(executable
 (name main)
 (libraries kaun rune vega nx))
```

Installing `kaun` pulls in `nx` and `rune` automatically; `vega` provides the optimizers.

## Step 1: Arrays with Nx

Nx provides n-dimensional arrays. Every value has a data type and a shape.

```ocaml
open Nx

let () =
  (* Create arrays *)
  let a = create Float32 [|2; 3|] [|1.; 2.; 3.; 4.; 5.; 6.|] in
  let b = ones Float32 [|2; 3|] in

  (* Element-wise operations *)
  let c = add a b in
  print_data c;

  (* Reductions *)
  Printf.printf "sum = %.1f\n" (item [] (sum a));
  Printf.printf "mean = %.1f\n" (item [] (mean a));

  (* Matrix multiplication *)
  let x = rand Float32 [|3; 4|] in
  let y = rand Float32 [|4; 2|] in
  let z = matmul x y in
  Printf.printf "matmul shape: %s\n"
    (Array.to_list (shape z) |> List.map string_of_int |> String.concat "x")
```

## Step 2: Gradients with Rune

Rune computes derivatives of Nx functions automatically. Write a function using Nx operations, then use `grad'` to differentiate it (the primed variants take a single tensor; the unprimed ones work over any parameter structure).

```ocaml
open Nx
open Rune

let () =
  (* f(x) = x² + sin(x) *)
  let f x = add (mul x x) (sin x) in

  (* grad' returns the derivative function *)
  let f' = grad' f in

  let x = scalar Float32 2.0 in
  Printf.printf "f(2)  = %.4f\n" (item [] (f x));
  Printf.printf "f'(2) = %.4f\n" (item [] (f' x));

  (* Higher-order: second derivative *)
  let f'' = grad' f' in
  Printf.printf "f''(2) = %.4f\n" (item [] (f'' x))
```

## Step 3: Training with Kaun

Kaun provides layers, losses, and initializers built on Rune. A model is a plain record with a hand-written traversal (`Nx.Ptree.S`), the training step is `value_and_grad` plus one Vega optimizer update, and the loop is a plain for loop — no trainer, no layer type.

<!-- $MDX skip -->
```ocaml
open Kaun

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Nx.tanh (Linear.apply p.l1 x))
end

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->

  (* XOR dataset *)
  let x = Nx.create Nx.float32 [|4; 2|]
    [|0.; 0.; 0.; 1.; 1.; 0.; 1.; 1.|] in
  let y = Nx.create Nx.float32 [|4; 1|]
    [|0.; 1.; 1.; 0.|] in

  (* Model parameters *)
  let params =
    { Mlp.l1 = Linear.init ~inputs:2 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1 }
  in

  (* Training step: value_and_grad + one Adam update *)
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in
  let step (params, ostate) =
    let l, grads = Rune.value_and_grad (module Mlp) loss params in
    let params, ostate =
      Vega.adam_step (module Mlp) ~lr:0.05 ostate ~params ~grads
    in
    ((params, ostate), Nx.item [] l)
  in

  (* Train *)
  let state = ref (params, Vega.adam_init (module Mlp) params) in
  for i = 1 to 500 do
    let s, l = step !state in
    state := s;
    if i mod 100 = 0 then Printf.printf "step %4d  loss %.6f\n" i l
  done;

  (* Predict *)
  let pred = Fn.sigmoid (Mlp.apply (fst !state) x) in
  Printf.printf "\npredictions (expected 0 1 1 0):\n";
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Nx.item [i; 0] x) (Nx.item [i; 1] x) (Nx.item [i; 0] pred)
  done
```

## Next Steps

- **[Nx](/docs/nx/getting-started/)** — full guide to arrays, slicing, broadcasting, linear algebra
- **[Rune](/docs/rune/getting-started/)** — all transformations: grad, jvp, vmap, and more
- **[Kaun](/docs/kaun/getting-started/)** — layers, losses, data, metrics, pretrained models
- **[Ecosystem Overview](/docs/ecosystem-overview/)** — how all 9 libraries fit together
