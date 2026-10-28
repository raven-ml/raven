# Tracing

A function given to rune2 is ordinary OCaml code. This page shows when that
code runs under each entry point, and what it may do while it runs. The
rules are in the Tracing section of [`rune2.mli`](../lib/rune2.mli).

## How often the code runs

A transformation runs its function once per call, over values that carry
more than their elements. A construct traces its functions: their code runs
over stand-ins, values with a dtype, shape and placement but no elements,
and what they compute is recorded as a program. A loop's step is traced once
per trace of the loop, whatever its number of steps. A `jit` traces once per
key.

This program counts the runs of three functions with a side effect:

```ocaml
let x = Nx.create Nx.float64 [| 4 |] [| 1.; 2.; 3.; 4. |]

let runs = ref 0

let loss x =
  incr runs;
  Nx.sum Nx.Infix.(x * x)

let counted f =
  runs := 0;
  f ();
  !runs

let under_grad =
  counted (fun () ->
      ignore (Rune2.grad' loss x);
      ignore (Rune2.grad' loss x))

let under_scan =
  counted (fun () ->
      ignore
        (Rune2.scan' ~init:(Nx.scalar Nx.float64 0.)
           ~f:(fun acc row ->
             incr runs;
             (Nx.Infix.(acc + row), acc))
           x))

let compiled = Rune2.jit' loss

let under_jit =
  counted (fun () ->
      for _ = 1 to 3 do
        ignore (compiled x)
      done;
      ignore (compiled (Nx.ones Nx.float64 [| 2 |])))
```

```ocaml
# let () =
    Printf.printf "grad %d, scan %d, jit %d\n" under_grad under_scan under_jit;;
grad 2, scan 1, jit 2
```

Two gradients run `loss` twice, once per call. The scan's step runs once
for four rows, because the scan traces it once. The compiled `loss` runs
once for the three calls of shape `[4]` and once more for the new shape
`[2]`, once per key (the Keys section of [`rune2.mli`](../lib/rune2.mli)).

## Reading elements

A stand-in has no elements, so `Nx.item` of one, or of a value computed from
one, raises, naming the construct:

```ocaml
let read_in_step () =
  Rune2.scan' ~init:(Nx.scalar Nx.float64 0.)
    ~f:(fun acc row ->
      let next = if Nx.item [] row > 2. then acc else Nx.Infix.(acc + row) in
      (next, next))
    x
```

```ocaml
# let () =
    match read_in_step () with
    | _ -> print_endline "no error"
    | exception Invalid_argument msg -> print_endline msg;;
Nx.item: the value is traced by Rune.scan'
```

A branch on a value is written with `Nx.where`, which selects per element,
or with `switch` or `cond`, which compute one branch (the Branches section of
[`rune2.mli`](../lib/rune2.mli)). This step adds the rows below `2.5`:

```ocaml
let summed =
  Rune2.scan' ~init:(Nx.scalar Nx.float64 0.)
    ~f:(fun acc row ->
      let kept = Nx.where (Nx.less_s row 2.5) row (Nx.zeros_like row) in
      let next = Nx.Infix.(acc + kept) in
      (next, next))
    x
```

```ocaml
# let () = Nx.print (fst summed);;
float64 [] 3
```

The rows `1` and `2` are below `2.5`, so the final carry is `3`.

Under `grad`, outside every construct, `Nx.item` reads the value's primal,
its elements without the tangent, and an OCaml `if` differentiates the path
taken (Reads, in the Derivatives section of [`rune2.mli`](../lib/rune2.mli)).

## Captures

A traced function reads what it captures when it is traced. Under `jit` that
is the first call of each key, so a `ref` read inside the function keeps the
value it had then:

```ocaml
let scale = ref 2.

let scaled = Rune2.jit' (fun x -> Nx.Infix.(x *$ !scale))

let before = scaled x

let () = scale := 10.

let after = scaled x
```

```ocaml
# let () = Nx.print before; Nx.print after;;
float64 [4] [2, 4, 6, 8]
float64 [4] [2, 4, 6, 8]
```

Both calls have the key of `x`, so the second runs the program traced at
the first, where `!scale` was `2`. A value that changes between calls is
passed as an argument.

## Reading a program

`Rune2.program` traces a function as `jit` would, without compiling or
computing it, and `pp_program` prints it (the Inspecting a trace section of
[`rune2.mli`](../lib/rune2.mli)). Here a function holds a scan:

```ocaml
let cumulative x =
  snd
    (Rune2.scan' ~init:(Nx.scalar Nx.float64 0.)
       ~f:(fun acc row ->
         let next = Nx.Infix.(acc + (row * row)) in
         (next, next))
       x)

let traced = Rune2.program Nx.Ptree.tensor Nx.Ptree.tensor cumulative x
```

```ocaml
# let () = Format.printf "%a@." Rune2.pp_program traced;;
program (%0)
  %1 : float64 [] at CPU = Map [float64:0x0000000000000000] at CPU
  %2 : float64 [] at CPU, %3 : float64 [4] at CPU = scan 4 rows, init (%1), xs (%0), captures ()
    step
      program (%4, %5)
        %6 : float64 [] at CPU = Map [mul(x0, x1)] (x0: float64 [] at CPU) (x1: float64 [] at CPU)
          with x0 = %5, x1 = %5
        %7 : float64 [] at CPU = Map [add(x0, x1)] (x0: float64 [] at CPU) (x1: float64 [] at CPU)
          with x0 = %4, x1 = %6
        return %7, %7
  return %3
```

The scan is one equation of the program, with its step's program indented
beneath it. The step was traced once, for all four rows.
