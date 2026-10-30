# Callbacks

A callback runs OCaml on the values of a traced or transformed function
(the Callbacks section of [`rune.mli`](../lib/rune.mli)). This page
builds the pattern of a model whose state comes from a solver outside nx.
The solver computes the state, an adjoint solve computes its gradient,
and rune differentiates, loops and compiles around both.

## A state from a solver

The state `u` of a parameter `θ` solves `(θ² + 1) u = 1`. In a real model
the equation is a large system and the solver an OCaml or C library. Here
the solver divides, and counts its runs:

```ocaml
let solves = ref 0
let adjoints = ref 0
let f64 = Nx.float64

let solve theta =
  incr solves;
  let t = Nx.item [] theta in
  Nx.scalar f64 (1. /. ((t *. t) +. 1.))
```

`solve` reads its argument's element with `Nx.item`. A callback's
function receives arrays with elements, so the read works even where the
callback's call is traced, as in a loop's step or under `jit`.

## Its derivative

A callback has no derivative, so the gradient of `u` comes from a custom
rule (the Custom differentiation rules section of
[`rune.mli`](../lib/rune.mli)). Differentiating the
state equation along `θ` gives `(θ² + 1) u̇ = -2θ u θ̇`. The cotangent of
`θ` is therefore `-2θ u λ`, where the *adjoint* `λ` solves the transposed
system `(θ² + 1) λ = ū` for the cotangent `ū` of `u`. The adjoint solve
is a second host function:

```ocaml
let adjoint (theta, (u, ct)) =
  incr adjoints;
  let t = Nx.item [] theta and u = Nx.item [] u in
  let lambda = Nx.item [] ct /. ((t *. t) +. 1.) in
  Nx.scalar f64 (-2. *. t *. u *. lambda)

let state =
  let one = Nx.Ptree.(tensor @-> returns tensor) in
  let three = Nx.Ptree.(pair tensor (pair tensor tensor) @-> returns tensor) in
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun theta ->
      let u = Rune.callback one ~like:theta solve theta in
      let pullback ct =
        Rune.callback three ~like:theta adjoint (theta, (u, ct))
      in
      (u, pullback))
```

`~like:theta` declares each result's dtype, shape and placement. The
state and `θ`'s cotangent have `θ`'s. The rule returns `u` with its
pullback, which closes over `θ` and `u` and calls the adjoint solve on
them and the cotangent. Inside a scan's step, `θ` and `u` are stand-ins,
and the callback hands `adjoint` their values at each execution.

## In a loop, under a gradient

The loss sums the states of four parameters with a scan:

```ocaml
let loss thetas =
  fst
    (Rune.scan' ~init:(Nx.scalar f64 0.)
       (fun acc theta -> (Nx.add acc (state theta), theta))
       thetas)

let thetas = Nx.create f64 [| 4 |] [| -1.; 0.; 0.5; 2. |]
let g = Rune.grad' loss thetas
```

```ocaml
# let () = Printf.printf "%d solves, %d adjoint solves\n" !solves !adjoints;;
4 solves, 4 adjoint solves
```

The scan's step is traced once, and its callbacks run once per row. The
forward pass solves for each row's state, and the backward pass runs the
pullback once per row. The gradient is `du/dθ = -2θ/(θ² + 1)²`
at each parameter:

```ocaml
# let () =
    let exact t = -2. *. t /. (((t *. t) +. 1.) ** 2.) in
    let err =
      List.fold_left2
        (fun e t d -> Float.max e (Float.abs (d -. exact t)))
        0.
        (Array.to_list (Nx.to_array thetas))
        (Array.to_list (Nx.to_array g))
    in
    Printf.printf "error below 1e-12: %b\n" (err < 1e-12);;
error below 1e-12: true
```

## Compiled

Under `jit` the gradient's programs hold the callbacks. A call evaluates
each program around them, and each stretch without one is a compiled call
of its own (the Evaluated programs section of
[`rune.mli`](../lib/rune.mli)). The callbacks still run once per row and
per call, never at the trace:

```ocaml
# let () =
    solves := 0;
    adjoints := 0;
    let step = Rune.jit' (Rune.grad' loss) in
    ignore (step thetas);
    ignore (step thetas);
    Printf.printf "%d solves, %d adjoint solves\n" !solves !adjoints;;
8 solves, 8 adjoint solves
```

Two calls of four rows each make eight solves and eight adjoint solves.
On a set other than the host, `solve` and `adjoint` would receive their
arguments where they lie and place them on the host with `Nx.place`
before reading them, then place their results where `like`'s lie, as
`Rune.callback` requires.
