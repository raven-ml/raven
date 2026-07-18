# Getting Started

This guide shows you how to compute gradients — of single tensors first, then of your own typed parameter records.

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
 (libraries rune nx))
```

## Your First Gradient

Rune differentiates ordinary functions over plain Nx tensors. For a function of a single tensor, use `grad'`:

```ocaml
let () =
  (* f(x) = x² + sin(x) *)
  let f x = Nx.add (Nx.mul x x) (Nx.sin x) in

  (* grad' returns a function that computes the derivative *)
  let f' = Rune.grad' f in

  let x = Nx.scalar Nx.float32 2.0 in
  Printf.printf "f(2)  = %.4f\n" (Nx.item [] (f x));
  Printf.printf "f'(2) = %.4f\n" (Nx.item [] (f' x))
  (* f'(x) = 2x + cos(x), so f'(2) ≈ 3.5839 *)
```

Key points:

- `grad' f` takes a function `f : Nx.t -> Nx.t` and returns a function that computes its gradient
- The function must return a scalar tensor (exactly one element)
- The gradient has the same shape and dtype as the input

## Differentiating a Record

Real models have more than one parameter. In rune the parameters are a record you define, made traversable by implementing `Nx.Ptree.S` — three one-line functions that visit the record's tensor leaves:

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

That is the entire registration story: no ppx, no runtime tree, no string keys. Every transformation takes the module as a first-class argument and returns values of your record type:

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
  (* The gradient is a value of type [params]. *)
  let g = Rune.grad (module Params) loss params in
  Printf.printf "dw:\n%s\n" (Nx.to_string g.w);
  Printf.printf "db: %s\n" (Nx.to_string g.b)
```

Leaves of the record that do not contribute to the result get all-zero gradients. Leaves may have different dtypes; each gradient leaf has its parameter leaf's dtype.

## Gradient Descent

In practice you want the loss and its gradient together; `value_and_grad` computes both in a single forward and backward pass. A training step is then a record update, and the loop is ordinary OCaml:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  (* Synthetic data: y = x @ w_true + 0.3. *)
  let w_true = Nx.create Nx.float32 [| 3; 1 |] [| 2.0; -1.0; 0.5 |] in
  let x = Nx.randn Nx.float32 [| 64; 3 |] in
  let y = Nx.add_s (Nx.matmul x w_true) 0.3 in

  let loss p =
    let pred = Nx.add (Nx.matmul x p.w) p.b in
    Nx.mean (Nx.square (Nx.sub pred y))
  in

  let lr = 0.1 in
  let step p =
    let l, g = Rune.value_and_grad (module Params) loss p in
    let p =
      { w = Nx.sub p.w (Nx.mul_s g.w lr); b = Nx.sub p.b (Nx.mul_s g.b lr) }
    in
    (p, Nx.item [] l)
  in

  let p =
    ref { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  for i = 1 to 200 do
    let p', l = step !p in
    p := p';
    if i mod 50 = 0 then Printf.printf "step %3d  loss %.6f\n" i l
  done;
  Printf.printf "w (expected ~[2.0; -1.0; 0.5]):\n%s\n"
    (Nx.to_string !p.w)
```

This is the whole pattern — the full program is [`examples/01-gradient-descent`](https://github.com/raven-ml/raven/tree/main/packages/rune/examples/01-gradient-descent). For neural networks, [kaun](/docs/kaun/) provides layers whose parameter records compose exactly this way.

## Auxiliary Outputs

When the objective returns data alongside the loss — predictions, metrics, updated state — use `value_and_grad_aux`. The auxiliary value rides through undifferentiated:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let f v =
    let pred = Nx.mul v v in
    (Nx.mean pred, pred) (* pred is auxiliary — not differentiated *)
  in
  let module Vec = struct
    type t = Nx.float32_t

    let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) v = f v
    let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
    let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) v = f v
  end in
  let loss, g, pred = Rune.value_and_grad_aux (module Vec) f x in
  Printf.printf "loss = %.2f\n" (Nx.item [] loss);
  Printf.printf "grad = %s\n" (Nx.to_string g);
  Printf.printf "pred = %s\n" (Nx.to_string pred)
```

The `Vec` module above is worth noting: a single tensor is itself a one-leaf `Ptree.S` structure, so the structured API subsumes the single-tensor one.

## Higher-Order Derivatives

`grad'` returns a regular function, so you can differentiate again:

```ocaml
let () =
  (* f(x) = x⁴ *)
  let f x = Nx.mul x (Nx.mul x (Nx.mul x x)) in
  let f' = Rune.grad' f in      (* 4x³ *)
  let f'' = Rune.grad' f' in    (* 12x² *)
  let f''' = Rune.grad' f'' in  (* 24x *)
  let x = Nx.scalar Nx.float32 2.0 in
  Printf.printf "f(2)    = %.1f\n" (Nx.item [] (f x));
  Printf.printf "f'(2)   = %.1f\n" (Nx.item [] (f' x));
  Printf.printf "f''(2)  = %.1f\n" (Nx.item [] (f'' x));
  Printf.printf "f'''(2) = %.1f\n" (Nx.item [] (f''' x))
```

## Stopping Gradients

Two mechanisms hold part of a computation constant during differentiation:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in

  (* detach: gradients do not flow through the copy. *)
  let f v = Nx.mean (Nx.mul v (Rune.detach v)) in
  Printf.printf "with detach:  %s\n"
    (Nx.to_string (Rune.grad' f x));

  (* no_grad: nothing inside is recorded. *)
  let g v =
    let baseline = Rune.no_grad (fun () -> Nx.mean v) in
    Nx.mean (Nx.mul v (Nx.sub v baseline))
  in
  ignore (Rune.grad' g x)
```

`detach` also serves as the escape hatch for operations whose gradient is not implemented (see [Transformations](02-transformations/)): detach their inputs if differentiation should not flow through them.

## Next Steps

- [Transformations](02-transformations/) — vjp, jvp, vmap, Hessians, remat, custom rules, control flow
- [How It Works](03-how-it-works/) — effects, handlers, and the tape
- [Kaun Getting Started](/docs/kaun/getting-started/) — neural networks on top of rune
