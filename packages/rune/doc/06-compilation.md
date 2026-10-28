# Compilation

`jit s f` is `f` compiled through tolk, raven's compiler. A call traces `f`
over stand-ins of its arguments into a program, keyed by the arguments'
abstract value, and runs the program compiled. The Compilation section of
[`rune2.mli`](../lib/rune2.mli) and its subsections state the rules.

## Keys

A call retraces exactly when its key is new to the compiled function. The key
holds each array's dtype, shape and placement, so calls whose arrays differ only
in their elements share it:

```ocaml
let traces = ref 0

let affine =
  Rune2.jit'
    (fun x ->
      incr traces;
      Nx.Infix.((x *$ 2.) +$ 1.))

let v n = Nx.full Nx.float64 [| n |] (float_of_int n)

let () =
  ignore (affine (v 3));
  ignore (affine (Nx.create Nx.float64 [| 3 |] [| 7.; 8.; 9. |]));
  ignore (affine (v 5))
```

```ocaml
# let () = Printf.printf "%d traces\n" !traces;;
2 traces
```

The first two calls share the key `float64 [3]` on the host, whatever their
elements; the third has the shape `[5]`, a new key. With the environment
variable `RUNE_JIT_DEBUG` set to a nonzero integer when rune2 is
initialised, each retrace prints on standard error how its key differs from
the latest call's (the Keys section).

## A training step

A compiled training step computes the loss, its gradient and the update in
one program. The parameters are a pair, and the step's signature has one
structure for them and one for its result:

```ocaml
let params = Nx.Ptree.(pair tensor tensor)

let inputs = Nx.create Nx.float64 [| 4 |] [| 0.; 1.; 2.; 3. |]
let targets = Nx.create Nx.float64 [| 4 |] [| 1.; 3.; 5.; 7. |]

let mse (w, b) =
  let err = Nx.Infix.((w * inputs) + b - targets) in
  Nx.mean Nx.Infix.(err * err)

let step =
  Rune2.jit
    Nx.Ptree.(params @-> returns (pair tensor params))
    (fun ((w, b) as p) ->
      let loss, (gw, gb) = Rune2.value_and_grad params mse p in
      (loss, (Nx.Infix.(w - (gw *$ 0.1)), Nx.Infix.(b - (gb *$ 0.1)))))

let rec train n p =
  let loss, p = step p in
  if n = 1 then (loss, p) else train (n - 1) p

let loss, (w, b) =
  train 200 (Nx.scalar Nx.float64 0., Nx.scalar Nx.float64 0.)
```

```ocaml
# let () = Nx.print loss; Nx.print w; Nx.print b;;
float64 [] 2.446119840025779e-14
float64 [] 2.0000001144011805
float64 [] 0.9999997557487191
```

The targets are `2x + 1`, so gradient descent on the mean squared error
moves `(w, b)` towards `(2, 1)` and the loss towards `0`. Every call has the
key of two float64 scalars, so `step` traces once and its later calls run
the compiled program.

## Donation

A donated argument gives up its memory to the call, which may write a result
into it. The donated value dies once the call has checked its arguments, and
a later use raises, naming the entry point, here `Rune.jit'` (the
Donation section of [`rune2.mli`](../lib/rune2.mli)):

```ocaml
let decay = Rune2.jit' (fun w -> Nx.Infix.(w *$ 0.5))

let w = Nx.create Nx.float64 [| 3 |] [| 2.; 4.; 6. |]
let kept = decay w
let w' = decay (Nx.donate w)
```

```ocaml
# let () =
    Nx.print w';
    Printf.printf "equal: %b\n" (Nx.to_array kept = Nx.to_array w');
    match Nx.to_array w with
    | _ -> print_endline "w is alive"
    | exception Invalid_argument msg -> print_endline msg;;
float64 [3] [1, 2, 3]
equal: true
Nx.to_array: float64 [3] was donated to Rune.jit'
```

A result's values are the same whether or not it takes a donated argument's
memory, so `w'` equals `kept`. `w` was consumed by the second call.

## Checks

A check (`Nx.check`) inside a compiled function is computed with the
program, and the first that fails raises its exception when the call
returns (the Checks section):

```ocaml
let checked_log =
  Rune2.jit' (fun x ->
      Nx.check Nx.Ptree.tensor (Nx.greater_s x 0.) x (fun i _ ->
          Failure (Printf.sprintf "log of a nonpositive element at %d" i.(0)));
      Nx.log x)
```

```ocaml
# let () =
    match checked_log (Nx.create Nx.float64 [| 3 |] [| 1.; -1.; 0. |]) with
    | _ -> print_endline "no error"
    | exception Failure msg -> print_endline msg;;
log of a nonpositive element at 1
```

The first failing index in C order is `1`, and the call's results are lost.

## Accuracy

A compiled call computes each nx function to the accuracy the Numerics section
of [`rune2.mli`](../lib/rune2.mli) states. Addition and multiplication are exact
there, in the sense that a compiled call gives eager's bits for them, so a
function of them does too:

```ocaml
let poly x = Nx.Infix.((x * x * x) - (x *$ 3.) + x)
let probe = Nx.create Nx.float64 [| 3 |] [| 0.1; 1.7; -2.3 |]
let bits x = Array.map Int64.bits_of_float (Nx.to_array x)
let exact = bits (Rune2.jit' poly probe) = bits (poly probe)
```

```ocaml
# let () = Printf.printf "bit for bit: %b\n" exact;;
bit for bit: true
```

A function such as `Nx.exp` may differ from eager's result, within the ulps
that section's table gives it.
