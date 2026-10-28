# One call

This page follows one call through rune2: the gradient of a compiled
function that holds a scan. Each step names the module that takes it and
the section of [`rune2.mli`](../lib/rune2.mli) that states its rule.

## The program

`f` sums the squares of a running sum. It is compiled once, and its body
counts its own runs:

```ocaml
let runs = ref 0

let f =
  Rune2.jit' (fun xs ->
      incr runs;
      let _, sums =
        Rune2.scan' ~init:(Nx.scalar Nx.float64 0.)
          ~f:(fun acc row ->
            let next = Nx.Infix.(acc + row) in
            (next, next))
          xs
      in
      Nx.sum Nx.Infix.(sums * sums))

let xs = Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]
let g1 = Rune2.grad' f xs
let g2 = Rune2.grad' f (Nx.create Nx.float64 [| 3 |] [| 0.; 0.; 1. |])
```

```ocaml
# let () = Nx.print g1; Nx.print g2; Printf.printf "%d run\n" !runs;;
float64 [3] [20, 18, 12]
float64 [3] [2, 2, 2]
1 run
```

The running sums of `[1; 2; 3]` are `s = [1; 3; 6]`, and `xᵢ` enters every
`sⱼ` with `j ≥ i`, so the gradient of `Σ sⱼ²` along `xᵢ` is `Σ_{j≥i} 2 sⱼ`:
`[20; 18; 12]`. For `[0; 0; 1]`, `s = [0; 0; 1]` and the gradient is
`[2; 2; 2]`. The function's body ran once for both gradients.

## Binding

`Rune2.jit' body` makes a compiled function with no program. It keeps one
program per key, and each program compiles once per *variant*, the layout
of its arguments in memory among others (the preamble's life cycle, and the
Keys and Compiled programs sections). The module `Jit` keeps the programs.

## Reverse mode starts

`grad' f xs` runs `f` once in forward mode, whose values carry tangents,
and records the operations on tangents (the Reverse mode section). `xs`
enters with a tangent. The module `Jvp` computes the tangents, the module
`Trace` records them, and `Linear` drives both.

## The trace

The call of `f` reaches `Jit` with the key of `xs`, `float64 [3]` on the
host, a key it has not seen. It traces the body over a stand-in of `xs`:
the body's code runs, `runs` becomes `1`, and every operation it applies is
recorded, the scan as one construct carrying its step's program. The step
is traced once, for all three rows (the Tracing section). This is the
program, as `Rune2.program` shows it, the host printed as `CPU`:

```ocaml
let body xs =
  let _, sums =
    Rune2.scan' ~init:(Nx.scalar Nx.float64 0.)
      ~f:(fun acc row ->
        let next = Nx.Infix.(acc + row) in
        (next, next))
      xs
  in
  Nx.sum Nx.Infix.(sums * sums)
```

```ocaml
# let () =
    Format.printf "%a@." Rune2.pp_program
      (Rune2.program Nx.Ptree.tensor Nx.Ptree.tensor body xs);;
program (%0)
  %1 : float64 [] at CPU = Map [float64:0x0000000000000000] at CPU
  %2 : float64 [] at CPU, %3 : float64 [3] at CPU = scan 3 rows, init (%1), xs (%0), captures ()
    step
      program (%6, %7)
        %8 : float64 [] at CPU = Map [add(x0, x1)] (x0: float64 [] at CPU) (x1: float64 [] at CPU)
          with x0 = %6, x1 = %7
        return %8, %8
  %4 : float64 [3] at CPU = Map [mul(x0, x1)] (x0: float64 [3] at CPU) (x1: float64 [3] at CPU)
    with x0 = %3, x1 = %3
  %5 : float64 [] at CPU = Reduce [Sum x0] over [0] (x0: float64 [3] at CPU)
    with x0 = %4
  return %5
```

`%1` is the scan's zero carry. The step adds its row to the carry and
returns the sum as both carry and output, and `%4` and `%5` square and sum
the outputs.

## The derivation

The call reads a value forward mode tracks, so forward mode handles it by
deriving a program from the key's, its derivative (`Program.derive`). A
program is derived once per *derivation*: the kind of program, and which
arguments have a tangent, with their dtypes, shapes and placements. Under
reverse mode that derivative is split by a second derivation into the part
that reads no tangent, which computes at the call and returns the values
the rest reads, and the rest, which the record holds as one call of a
program linear in the tangents. The derivative of a scan is a scan of its
step's derivative, so each part holds a scan (the Transformations of
compiled calls section, and the preamble).

## Compiling and running

The part that reads no tangent compiles as the key's program would. It is
lowered through tolk into one loop whose step compiles once, then compiled
for the variant of its arguments (the Compiled programs and Loops and rules
sections). It runs, and returns `f xs` and the values the backward pass
reads: the running sums, which the tangent of `sums * sums` multiplies. The
step's tangent, `dacc + drow`, reads no value, so the loop keeps nothing of
each step (the Loops section). Neither the key's own program nor its
derivative runs here, so neither compiles.

## The backward pass

`grad'` then transposes the record from the cotangent `1` of the result.
The transpose of the linear part is one more derived program, which
compiles as the others do. Its scan runs the step's transpose from the last
row to the first, and returns the cotangent of `xs`, which becomes `g1`.

## The second call

The second gradient's argument has the same key, so the call finds the
key's program without tracing, and the body does not run again. It finds
the derived programs too, because the derivations are the same: reverse
mode with a tangent on `xs`, of `float64 [3]` on the host. Its argument is
fresh, as the first was, so it has the same layout, the compiled variants
are found as well, and the call runs them.
