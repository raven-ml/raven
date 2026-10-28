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

Installing `kaun` pulls in `nx`; `rune` differentiates and `vega` provides the optimizers, each installed beside it.

## Step 1: Arrays with Nx

Nx provides n-dimensional arrays. Every value has a data type, a shape, and the device set it lies on. A value made from numbers is a host value, of type `(_, _, Nx.host) Nx.t`, which lies in the process's memory and which the host computes on ([nx.mli, Device sets](../packages/nx/lib/nx.mli)). A random draw takes a key, so a program that fixes its keys draws the same numbers on every run.

```ocaml
let () =
  (* Create arrays *)
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.ones Nx.float32 [| 2; 3 |] in

  (* Element-wise operations *)
  let c = Nx.add a b in
  Nx.print c;

  (* Reductions *)
  Printf.printf "sum = %.1f\n" (Nx.item [] (Nx.sum a));
  Printf.printf "mean = %.1f\n" (Nx.item [] (Nx.mean a));

  (* Random draws: equal keys give equal numbers *)
  let keys = Nx.Rng.split (Nx.Rng.key 0) in
  let x = Nx.Rng.uniform ~key:keys.(0) Nx.float32 [| 3; 4 |] in
  let y = Nx.Rng.uniform ~key:keys.(1) Nx.float32 [| 4; 2 |] in

  (* Matrix multiplication *)
  let z = Nx.matmul x y in
  Format.printf "matmul shape: %a@." Nx.pp_shape (Nx.shape z)
```

## Step 2: Gradients with Rune

Rune computes derivatives of Nx functions automatically. Write a function using Nx operations, then use `grad'` to differentiate it (the primed variants take a single array; the unprimed ones work over any parameter structure).

```ocaml
let () =
  (* f(x) = x² + sin(x) *)
  let f x = Nx.add (Nx.mul x x) (Nx.sin x) in

  (* grad' returns the derivative function *)
  let f' = Rune.grad' f in

  let x = Nx.scalar Nx.float32 2.0 in
  Printf.printf "f(2)  = %.4f\n" (Nx.item [] (f x));
  Printf.printf "f'(2) = %.4f\n" (Nx.item [] (f' x));

  (* Higher-order: second derivative *)
  let f'' = Rune.grad' f' in
  Printf.printf "f''(2) = %.4f\n" (Nx.item [] (f'' x))
```

## Step 3: Training with Kaun

Kaun provides layers, losses, and initializers. A model is a plain record with a hand-written `walk`, one line per field, instantiated by `Nx.Ptree.instantiate`. A layer's constructor draws its parameters from the key it is given. The training step is `value_and_grad` plus one Vega optimizer update, and the loop is a plain for loop, with no trainer and no layer type.

<!-- $MDX skip -->
```ocaml
open Kaun

module Mlp = struct
  type ('a, 'd) t = { l1 : ('a, 'd) Linear.t; l2 : ('a, 'd) Linear.t }

  let walk c { l1; l2 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { l1; l2 }

  let apply p x = Linear.apply p.l2 (Nx.tanh (Linear.apply p.l1 x))
end

let mlp = Nx.Ptree.instantiate (module Mlp)

let () =
  (* XOR dataset *)
  let x = Nx.create Nx.float32 [| 4; 2 |]
    [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |] in
  let y = Nx.create Nx.float32 [| 4; 1 |]
    [| 0.; 1.; 1.; 0. |] in

  (* Model parameters, drawn from two keys of one seed *)
  let keys = Nx.Rng.split (Nx.Rng.key 42) in
  let params =
    { Mlp.l1 = Linear.init ~key:keys.(0) ~inputs:2 ~outputs:8;
      l2 = Linear.init ~key:keys.(1) ~inputs:8 ~outputs:1 }
  in

  (* Training step: value_and_grad + one Adam update *)
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in
  let step (params, ostate) =
    let l, grads = Rune.value_and_grad mlp loss params in
    let params, ostate =
      Vega.adam_step mlp ~lr:(Vega.lr 0.05) ostate ~params ~grads
    in
    ((params, ostate), Nx.item [] l)
  in

  (* Train *)
  let state = ref (params, Vega.adam_init mlp params) in
  for i = 1 to 500 do
    let s, l = step !state in
    state := s;
    if i mod 100 = 0 then Printf.printf "step %4d  loss %.6f\n" i l
  done;

  (* Predict *)
  let pred = Nx.sigmoid (Mlp.apply (fst !state) x) in
  Printf.printf "\npredictions (expected 0 1 1 0):\n";
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Nx.item [i; 0] x) (Nx.item [i; 1] x) (Nx.item [i; 0] pred)
  done
```

## Step 4: Compiling the Step

`Rune.jit` compiles a function, given its signature: one structure per argument, joined by `@->`, and the result's after `returns`. The step's code runs once, when the first call traces it, and later calls run the compiled program. Donating the state at the call, with `Nx.Ptree.donate`, lets the step write the next state into its memory. In Step 3, replace the step and the loop with:

<!-- $MDX skip -->
```ocaml
  let st = Nx.Ptree.pair mlp (Vega.adam_ptree mlp) in
  let step =
    Rune.jit
      Nx.Ptree.(st @-> returns (pair st tensor))
      (fun (params, ostate) ->
        let l, grads = Rune.value_and_grad mlp loss params in
        let params, ostate =
          Vega.adam_step mlp ~lr:(Vega.lr 0.05) ostate ~params ~grads
        in
        ((params, ostate), l))
  in
  let state = ref (params, Vega.adam_init mlp params) in
  for i = 1 to 500 do
    let s, l = step (Nx.Ptree.donate st !state) in
    state := s;
    if i mod 100 = 0 then Printf.printf "step %4d  loss %.6f\n" i (Nx.item [] l)
  done;
```

The Predict block is unchanged. A donated value is dead once the call has read it, and a later use of the old state raises ([nx.mli, Donation](../packages/nx/lib/nx.mli)).

The same step runs on a GPU. `Nx.devices` mints a device set from a GPU that rig opens, `Nx.place` and `Nx.Ptree.place` put the data and the parameters there, and the step compiles for that device. A set minted without `~kernels` over GPUs computes nothing eagerly, and an operation there raises, so work there runs under `Rune.jit`, one device per compiled program. [Compilation](../packages/rune/doc/06-compilation.md) covers devices.

## Next Steps

- **[Nx](../packages/nx/doc/01-getting-started.md)** — full guide to arrays, slicing, broadcasting, linear algebra
- **[Rune](../packages/rune/doc/index.md)** — all transformations: grad, jvp, vmap, and more
- **[Kaun](../packages/kaun/doc/01-getting-started.md)** — layers, losses, data, metrics, pretrained models
- **[Ecosystem Overview](ecosystem-overview.md)** — how the libraries fit together
