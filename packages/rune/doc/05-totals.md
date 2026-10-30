# Totals

A total is a write-only sum. Code anywhere inside a function adds to it,
and the caller reads it when the function returns: `Total.collect t ~init f`
is `f ()` beside `init` plus the additions to `t` that reached it. Nothing
reads a total before its `collect` returns, so an addition never changes a
value the function computes. The Totals section of
[`rune.mli`](../lib/rune.mli) states the rules.

## A penalty from inside a loop

Here each step of a scan adds the square of its carry to a total, as a
regulariser would, and the caller collects it:

```ocaml
let penalty : (float, Nx.float64_elt, Nx.host) Rune.Total.t =
  Rune.Total.make ()

let xs = Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]

let run xs =
  fst
    (Rune.scan' ~init:(Nx.scalar Nx.float64 0.)
       (fun acc row ->
         let next = Nx.Infix.(acc + row) in
         Rune.Total.add penalty Nx.Infix.(next * next);
         (next, next))
       xs)

let final, collected =
  Rune.Total.collect penalty ~init:(Nx.scalar Nx.float64 0.) (fun () -> run xs)
```

```ocaml
# let () = Nx.print final; Nx.print collected;;
float64 [] 6
float64 [] 46
```

The carries are `1`, `3` and `6`, so the total is `1 + 9 + 36 = 46`. The
step was traced once, and its addition was made once per step, as the
steps written out one after another would make it.

## Under jit and vmap

A compiled function adds its additions once per call, as the call returns,
and computes their sums in its program:

```ocaml
let compiled = Rune.jit' run

let _, compiled_total =
  Rune.Total.collect penalty ~init:(Nx.scalar Nx.float64 0.) (fun () ->
      compiled xs)
```

```ocaml
# let () = Nx.print compiled_total;;
float64 [] 46
```

The compiled scan makes the same three additions, so the total is `46`.
Inside a map each lane makes the additions of the code it runs, and a
`collect` outside the map receives each addition's lanes in turn, lane 0
first:

```ocaml
let batch = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 0.; 0.; 1. |]

let _, mapped_total =
  Rune.Total.collect penalty ~init:(Nx.scalar Nx.float64 0.) (fun () ->
      Rune.vmap' run batch)
```

```ocaml
# let () = Nx.print mapped_total;;
float64 [] 47
```

The first lane's additions come to `46`, as above; the second's carries
are `0`, `0` and `1`, so its additions come to `1`, and the total is `47`.

## Under a differentiation

An addition under a differentiation adds its value's primal, once per
execution of the code that makes it:

```ocaml
let _, grad_total =
  Rune.Total.collect penalty ~init:(Nx.scalar Nx.float64 0.) (fun () ->
      Rune.grad' run xs)
```

```ocaml
# let () = Nx.print grad_total;;
float64 [] 46
```

`grad'` runs `run` once, and the scan's step makes its addition once per
step, so the total is `46` again.
