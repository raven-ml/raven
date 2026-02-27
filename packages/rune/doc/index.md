# rune

Rune provides functional transformations for Nx tensors: automatic differentiation (forward and reverse mode), vectorising maps, and gradient checking. It operates on `Nx.t` values directly — no special tensor type is needed.

## Features

- **Reverse-mode AD** — `grad`, `value_and_grad`, `vjp` for backpropagation
- **Forward-mode AD** — `jvp` for Jacobian-vector products
- **Vectorising map** — `vmap` to lift per-example functions to batched operations
- **Gradient checking** — `check_gradient` and `finite_diff` for testing
- **Composable** — nest transformations freely (`grad (grad f)`, `vmap (grad f)`)
- **Effect-based** — uses OCaml 5 effects to intercept Nx operations cleanly

## Quick Start

```ocaml
open Nx
open Rune

(* Define a function using Nx operations *)
let f x = add (mul x x) (sin x)

(* Compute its gradient *)
let f' = grad f

let () =
  let x = scalar float32 2.0 in
  Printf.printf "f(2)  = %.4f\n" (item [] (f x));
  Printf.printf "f'(2) = %.4f\n" (item [] (f' x))
```

## Next Steps

- [Getting Started](/docs/rune/getting-started/) — installation and first gradients
- [Transformations](/docs/rune/transformations/) — complete guide to grad, jvp, vmap, and more
- [How It Works](/docs/rune/how-it-works/) — effects-based automatic differentiation explained
