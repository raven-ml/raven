# rune2

rune2 computes derivatives, maps, loops and compiled versions of OCaml
functions whose arguments and results hold nx arrays. Its entry points are
values of the module `Rune2`, and each takes the function it transforms.

rune2 is a library of the `dev` project, private until it replaces rune. A
program in this repository adds `rune2` and `nx` to its `libraries`:

```dune
(executable
 (name main)
 (libraries nx rune2))
```

The preamble of [`rune2.mli`](../lib/rune2.mli) defines the terms these
pages use: structures, signatures, transformations, constructs, stand-ins
and programs.

## A first gradient

This program computes the gradient of `sum x²` at `x = [1; 2; 3]`:

```ocaml
let x = Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]
let loss x = Nx.sum Nx.Infix.(x * x)
let g = Rune2.grad Nx.Ptree.tensor loss x
```

```ocaml
# let () = Nx.print g;;
float64 [3] [2, 4, 6]
```

The gradient of `Σ xᵢ²` is `2x`, so `g` is `[2; 4; 6]`.

- `Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]` makes a float64 array of
  shape `[3]` holding 1, 2 and 3, on the host. A differentiation tracks
  float and complex arrays; the gradient of an integer or boolean array is
  zero (the Derivatives section of [`rune2.mli`](../lib/rune2.mli)).
- `loss` returns `Nx.sum` of the squares, one float element. `grad`
  requires such a result; any other result is differentiated by `vjp`
  (`grad` in the Reverse mode section of [`rune2.mli`](../lib/rune2.mli)).
  `Nx.Infix.( … )` opens nx's operators locally, so `x * x` is `Nx.mul x x`.
- `Nx.Ptree.tensor` is the *structure* of the parameters: it says that `x` is
  one array. rune2 never inspects an OCaml value by itself, so each
  transformation and construct takes one structure for each value whose arrays
  it reads. A pair of arrays is `Nx.Ptree.(pair tensor tensor)`.
- `Rune2.grad Nx.Ptree.tensor loss x` runs `loss` once, over values that
  carry a tangent, records the operations on the tangents, and runs that
  record backward from the result. The gradient has the structure, dtypes
  and shapes of `x` (the Reverse mode section).

Functions of one array have shorter forms, named with a prime, which take
no structure: `Rune2.grad' loss x` is the same gradient (the Functions of one
array section of [`rune2.mli`](../lib/rune2.mli)).

## Two parameters

A model with a weight and a bias has a pair of arrays as its parameters, so
its structure is a pair. `value_and_grad` returns the loss beside the
gradient, from one forward and one backward pass:

```ocaml
let inputs = Nx.create Nx.float64 [| 3 |] [| 0.; 1.; 2. |]
let targets = Nx.create Nx.float64 [| 3 |] [| 1.; 3.; 5. |]

let mse (w, b) =
  let err = Nx.Infix.((w * inputs) + b - targets) in
  Nx.mean Nx.Infix.(err * err)

let w = Nx.create Nx.float64 [||] [| 1. |]
let b = Nx.create Nx.float64 [||] [| 0. |]

let value, (gw, gb) =
  Rune2.value_and_grad Nx.Ptree.(pair tensor tensor) mse (w, b)
```

```ocaml
# let () = Nx.print value; Nx.print gw; Nx.print gb;;
float64 [] 4.666666666666667
float64 [] -5.333333333333333
float64 [] -4
```

The errors are `w x + b - y = [-1; -2; -3]`, so the loss is `14 / 3`, its
derivative along `b` is the mean of `2 err`, `-4`, and along `w` the mean
of `2 err x`, `-16 / 3`. The gradient comes back as a pair, as the
parameters went in. `mse` uses `inputs` and `targets`
without receiving them, so they are captured, and `grad` does not track
them.

## A compiled function

`jit` makes a compiled function from a function and its *signature*, one
structure per argument and one for the result, joined by `@->` and
`returns`. The compiled function is
bound once and called many times:

```ocaml
let step =
  Rune2.jit
    Nx.Ptree.(tensor @-> returns tensor)
    (fun w -> Nx.Infix.(w - (Rune2.grad' loss w *$ 0.1)))

let w2 = step (step x)
```

```ocaml
# let () = Nx.print w2;;
float64 [3] [0.64, 1.28, 1.92]
```

Each step moves `x` against the gradient `2x`, so `w ↦ 0.8 w`, and two
steps give `0.64 x = [0.64; 1.28; 1.92]`, to the accuracy a compiled call
keeps (the Numerics section of [`rune2.mli`](../lib/rune2.mli)).

The first call traces the function over stand-ins of `x`, values with a
dtype, shape and placement but no elements, into a program, and compiles
it. The second call has the same *key*, the dtypes, shapes and placements
of its arguments among others, so it runs the compiled program without
tracing again (the Keys section of [`rune2.mli`](../lib/rune2.mli)).
Writing `Rune2.jit s f x` at each call makes a new compiled function each
time, which traces every call; the preamble's life cycle says why `step`
is bound once.

## The pages

- [Tracing](01-tracing.md): when the code of a function runs, and what a
  traced function may do.
- [Derivatives](02-derivatives.md): forward and reverse mode over
  structures, and complex arrays.
- [Maps](03-maps.md): `vmap`, lanes and Jacobians.
- [Loops](04-loops.md): `scan`, `iterate`, `root` and `remat`.
- [Totals](05-totals.md): sums collected from inside a function.
- [Compilation](06-compilation.md): keys, donation, checks and accuracy.
- [One call](07-one-call.md): the gradient of a compiled loop, followed
  through rune2.
