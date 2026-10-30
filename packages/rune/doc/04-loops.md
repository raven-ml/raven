# Loops

A loop applies a step to a carry. `scan` passes each step a row of its
input and stacks what the steps return; `iterate` runs until a test holds.
The step is traced once per trace of the loop, whatever the number of
steps, and every
transformation of a loop is a loop of its step's transformation. The Loops,
Roots and Rematerialization sections of [`rune.mli`](../lib/rune.mli)
state the rules.

## A scan and its gradient

This scan computes the running product of its rows:

```ocaml
let xs = Nx.create Nx.float64 [| 4 |] [| 1.; 2.; 3.; 4. |]

let running_product xs =
  Rune.scan' ~init:(Nx.scalar Nx.float64 1.)
    (fun acc row ->
      let next = Nx.Infix.(acc * row) in
      (next, next))
    xs

let total, prefixes = running_product xs
let g = Rune.grad' (fun xs -> fst (running_product xs)) xs
```

```ocaml
# let () = Nx.print total; Nx.print prefixes; Nx.print g;;
float64 [] 24
float64 [4] [1, 2, 6, 24]
float64 [4] [24, 12, 8, 6]
```

The outputs are the prefixes' products `[1; 2; 6; 24]`. The derivative of `Π xⱼ`
along `xᵢ` is the product of the others, `24 / xᵢ`, so the gradient is `[24; 12;
8; 6]`. Reverse mode ran the loop once forward, keeping of each step the values
its tangent's operations read, here the carry, and reading each row from `xs`,
then ran the step's transpose from the last step to the first.

## Iterating to a tolerance

`iterate'` applies Newton's step for `√a` until the residual is below a
tolerance, at most `max` times:

```ocaml
let newton a =
  Rune.iterate' ~max:50
    ~until:(fun x -> Nx.all (Nx.less_s (Nx.abs Nx.Infix.((x * x) - a)) 1e-12))
    (fun x -> Nx.Infix.((x + (a / x)) /$ 2.))
    a

let roots = newton (Nx.create Nx.float64 [| 2 |] [| 4.; 9. |])
```

```ocaml
# let () = Nx.print roots;;
float64 [2] [2, 3]
```

Newton's method converges to `√4 = 2` and `√9 = 3`. Under a map each lane
stops on its own test, and a lane that stopped keeps its carry while the
others run:

```ocaml
let lanes_apart =
  Rune.vmap' newton (Nx.create Nx.float64 [| 3 |] [| 1.; 16.; 1e6 |])
```

```ocaml
# let () = Nx.print lanes_apart;;
float64 [3] [1, 4.000000000000051, 1000]
```

`1` satisfies the test at once and takes no step; `1e6` takes the most.
With too small a `max` the call raises, naming the first lane still
running. `1` passes at once, and `16` still fails the test after three
steps, so lane 1 is the first:

```ocaml
let too_few () =
  Rune.vmap'
    (fun a ->
      Rune.iterate' ~max:3
        ~until:(fun x ->
          Nx.all (Nx.less_s (Nx.abs Nx.Infix.((x * x) - a)) 1e-12))
        (fun x -> Nx.Infix.((x + (a / x)) /$ 2.))
        a)
    (Nx.create Nx.float64 [| 3 |] [| 1.; 16.; 1e6 |])
```

```ocaml
# let () =
    match too_few () with
    | _ -> print_endline "no error"
    | exception Invalid_argument msg -> print_endline msg;;
Rune.iterate': until is still false after max = 3 steps, in lane 1
```

## Roots

Differentiating the Newton loop would differentiate every step it took. A
`root` states instead that the value its solve returns is a zero of a
residual, and takes the derivative from the residual: here
`x² − a = 0` gives `dx/da = 1 / (2x)`. The solve is the same loop:

```ocaml
let sqrt_root a =
  let residual x = Nx.Infix.((x * x) - a) in
  Rune.root Nx.Ptree.tensor ~residual (fun () -> newton a)

let a = Nx.create Nx.float64 [| 2 |] [| 4.; 9. |]
let da = Rune.grad' (fun a -> Nx.sum (sqrt_root a)) a

let checked =
  Rune.check_grads Nx.Ptree.tensor (fun a -> Nx.sum (sqrt_root a)) a
```

```ocaml
# let () =
    Nx.print da;
    print_endline (match checked with Ok () -> "ok" | Error msg -> msg);;
float64 [2] [0.25, 0.16666666666666666]
ok
```

`1 / (2 √4)` and `1 / (2 √9)` are `0.25` and `1/6`, and central
differences agree. The derivative is taken at the returned point; the
solve's own operations have none (the Roots section).

## Rematerialization

Reverse mode keeps the values a step's tangent operations read. `remat` keeps a
function's arguments and captures instead and computes its intermediates again
in the backward pass, so it computes more and keeps less. Every transformation
sees `remat s f` as it sees `f`, so the gradient is unchanged:

```ocaml
let layer =
  Rune.remat
    Nx.Ptree.(tensor @-> returns tensor)
    (fun x -> Nx.tanh (Nx.sin x))

let deep xs =
  fst
    (Rune.scan' ~init:(Nx.scalar Nx.float64 0.5)
       (fun acc row ->
         let next = layer Nx.Infix.(acc + row) in
         (next, next))
       xs)

let plain xs =
  fst
    (Rune.scan' ~init:(Nx.scalar Nx.float64 0.5)
       (fun acc row ->
         let next = Nx.tanh (Nx.sin Nx.Infix.(acc + row)) in
         (next, next))
       xs)

let gap =
  Nx.item [] (Nx.max (Nx.abs Nx.Infix.(Rune.grad' deep xs - Rune.grad' plain xs)))
```

```ocaml
# let () = Printf.printf "agree within 1e-12: %b\n" (gap < 1e-12);;
agree within 1e-12: true
```

The derivatives of `remat s f` are `f`'s, so the two gradients agree to
rounding.
