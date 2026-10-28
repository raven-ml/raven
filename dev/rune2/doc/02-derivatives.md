# Derivatives

rune2 differentiates in two modes. Forward mode carries a tangent beside
each value and gives a function's change along one direction of its
parameters. Reverse mode records the operations on tangents and runs the
record backward, and gives the change of one result along every parameter.
The Derivatives and Reverse mode sections of
[`rune2.mli`](../lib/rune2.mli) state the rules this page applies.

## Forward and reverse over a structure

Here `f` takes a pair and returns a pair. `jvp` moves the parameters along a
tangent of the same structure, and `vjp` returns a pullback that maps a
cotangent of the result to one of the parameters:

```ocaml
let pair = Nx.Ptree.(pair tensor tensor)

let f (a, b) = (Nx.Infix.(a * b), Nx.sin a)

let a = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |]
let b = Nx.create Nx.float64 [| 2 |] [| 3.; 4. |]

let _, (dy0, dy1) =
  Rune2.jvp pair pair f (a, b) (Nx.ones_like a, Nx.zeros_like b)

let _, pullback = Rune2.vjp pair pair f (a, b)
let ua, ub = pullback (Nx.ones_like a, Nx.zeros_like a)
```

```ocaml
# let () = Nx.print dy0; Nx.print dy1; Nx.print ua; Nx.print ub;;
float64 [2] [3, 4]
float64 [2] [0.5403023058681397, -0.4161468365471424]
float64 [2] [3, 4]
float64 [2] [1, 2]
```

Along `a`, the tangent of `a * b` is `b = [3; 4]` and that of `sin a` is `cos
a`. The cotangent `(1, 0)` selects the first result, whose derivatives are `b`
along `a` and `a = [1; 2]` along `b`. The tangent has the parameters' structure,
and the cotangent the result's.

`linearize` runs `f` once and returns the tangent map at that point, which
evaluates its record for each tangent without running `f` again:

```ocaml
let _, along = Rune2.linearize pair pair f (a, b)
let t1, _ = along (Nx.ones_like a, Nx.zeros_like b)
let t2, _ = along (Nx.zeros_like a, Nx.ones_like b)
```

```ocaml
# let () = Nx.print t1; Nx.print t2;;
float64 [2] [3, 4]
float64 [2] [1, 2]
```

The first tangent equals `jvp`'s `dy0`, and the second is `a`, the
derivative of `a * b` along `b`.

## Second derivatives

Differentiations nest, each tracking the values of its own function. The
Hessian-vector product of `g` is the forward derivative of its gradient:

```ocaml
let g x = Nx.sum Nx.Infix.(x * x * x)
let hvp x v = snd (Rune2.jvp' (Rune2.grad' g) x v)
let h = hvp a (Nx.ones_like a)
```

```ocaml
# let () = Nx.print h;;
float64 [2] [6, 12]
```

The gradient of `Σ xᵢ³` is `3x²`, its Hessian is diagonal with `6x`, so
along ones the product is `6a = [6; 12]`.

`detach` gives a value a zero derivative under every differentiation around
it (the Controlling differentiation section). Here only the first factor
is differentiated:

```ocaml
let held = Rune2.grad' (fun x -> Nx.sum Nx.Infix.(x * Rune2.detach x)) a
```

```ocaml
# let () = Nx.print held;;
float64 [2] [1, 2]
```

The derivative of `x · c` with `c = x` held is `c = a`, where the
undetached product would give `2a`.

## Complex parameters

A complex array is two real parts per element, and its tangent displaces
both. The gradient of a real function of `z = x + iy` is
`∂f/∂x − i ∂f/∂y` (Tangents and cotangents, in the Derivatives section of
[`rune2.mli`](../lib/rune2.mli)). For `|z|² = x² + y²` that is
`2x − 2iy = 2 conj z`:

```ocaml
let z =
  Nx.create Nx.complex128 [| 2 |]
    [| { Complex.re = 1.; im = 2. }; { Complex.re = -3.; im = 0.5 } |]

let norm2 z = Nx.sum (Nx.real Nx.float64 Nx.Infix.(z * Nx.conjugate z))
let gz = Rune2.grad' norm2 z
```

```ocaml
# let () = Nx.print gz;;
complex128 [2] [2-4i, -6-1i]
```

`gz` is `2 conj z = [2 - 4i; -6 - i]`. A descent step moves `z` against the
gradient's conjugate, `z - lr · conj gz`, which here is `z - 2 lr z` and
shrinks `|z|²`.

## Checking a gradient

`check_grads` compares a gradient with central differences along two fixed
directions, and returns `Ok ()` when they agree within its tolerance:

```ocaml
let checked = Rune2.check_grads Nx.Ptree.tensor (fun x -> Nx.sum (Nx.exp x)) a
```

```ocaml
# let () =
    match checked with
    | Ok () -> print_endline "ok"
    | Error msg -> print_endline msg;;
ok
```

The function is smooth and the parameters are float64, whose differences
are reliable, so the two derivatives agree.

## Custom rules

A function may give the derivative a differentiation uses for it: a tangent
map with `custom_jvp`, a pullback with `custom_vjp`, or a linear function's
transpose with `custom_transpose`. Here a square root iterates in OCaml
until its steps stop moving, and gives the pullback of `√a`, `g / (2x)`:

```ocaml
let sqrt_newton =
  Rune2.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun a ->
      let rec go x =
        let next = Nx.Infix.((x + (a / x)) /$ 2.) in
        let moved = Nx.item [] (Nx.max (Nx.abs Nx.Infix.(next - x))) in
        if moved < 1e-12 then next else go next
      in
      let x = go a in
      (x, fun g -> Nx.Infix.(g / (x *$ 2.))))

let r = Nx.create Nx.float64 [| 2 |] [| 4.; 9. |]
let dr = Rune2.grad' (fun a -> Nx.sum (sqrt_newton a)) r
```

```ocaml
# let () = Nx.print (sqrt_newton r); Nx.print dr;;
float64 [2] [2, 3]
float64 [2] [0.25, 0.16666666666666666]
```

Outside every transformation, and under one differentiation, the rule runs
on its argument's values, so `Nx.item` reads them. The gradient is
`1 / (2 √a)`, from the pullback, whatever steps the loop took. Where the
rule is traced, as under `jit`, the read raises:

```ocaml
# let () =
    match Rune2.jit' sqrt_newton r with
    | _ -> print_endline "no error"
    | exception Invalid_argument msg -> print_endline msg;;
Nx.item: the value is traced by Rune.custom_vjp
```

The Custom differentiation rules section of
[`rune2.mli`](../lib/rune2.mli) says where a rule runs on values and where
it is traced.
