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
 (libraries kaun))
```

Installing `kaun` pulls in `nx` and `rune` automatically.

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

Rune computes derivatives of Nx functions automatically. Write a function using Nx operations, then use `grad` to differentiate it.

```ocaml
open Nx
open Rune

let () =
  (* f(x) = x² + sin(x) *)
  let f x = add (mul x x) (sin x) in

  (* grad returns the derivative function *)
  let f' = grad f in

  let x = scalar Float32 2.0 in
  Printf.printf "f(2)  = %.4f\n" (item [] (f x));
  Printf.printf "f'(2) = %.4f\n" (item [] (f' x));

  (* Higher-order: second derivative *)
  let f'' = grad f' in
  Printf.printf "f''(2) = %.4f\n" (item [] (f'' x))
```

## Step 3: Training with Kaun

Kaun provides layers, optimizers, and training loops built on Rune.

<!-- $MDX skip -->
```ocaml
open Kaun

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->

  (* XOR dataset *)
  let x = Nx.create Nx.Float32 [|4; 2|]
    [|0.; 0.; 0.; 1.; 1.; 0.; 1.; 1.|] in
  let y = Nx.create Nx.Float32 [|4; 1|]
    [|0.; 1.; 1.; 0.|] in

  (* Define model *)
  let model = Layer.sequential [
    Layer.linear ~in_features:2 ~out_features:8 ();
    Layer.tanh ();
    Layer.linear ~in_features:8 ~out_features:1 ();
  ] in

  (* Create trainer and initialize *)
  let trainer = Train.make ~model
    ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 0.01) ()) in
  let st = Train.init trainer ~dtype:Nx.Float32 in

  (* Train *)
  let st = Train.fit trainer st
    ~report:(fun ~step ~loss _st ->
      if step mod 250 = 0 then
        Printf.printf "step %4d  loss %.6f\n" step loss)
    (Data.repeat 1000 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in

  (* Predict *)
  let pred = Train.predict trainer st x |> Nx.sigmoid in
  Printf.printf "\npredictions (expected 0 1 1 0):\n";
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Nx.item [i; 0] x) (Nx.item [i; 1] x) (Nx.item [i; 0] pred)
  done
```

## Next Steps

- **[Nx](/docs/nx/getting-started/)** — full guide to arrays, slicing, broadcasting, linear algebra
- **[Rune](/docs/rune/getting-started/)** — all transformations: grad, jvp, vmap, and more
- **[Kaun](/docs/kaun/getting-started/)** — layers, optimizers, training loops, pretrained models
- **[Ecosystem Overview](/docs/ecosystem-overview/)** — how all 9 libraries fit together
