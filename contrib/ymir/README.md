# Ymir

Exact physical units for OCaml, built on [Nx](../../packages/nx/).

`ymir.units` gives a unit as an exact value: a product of primes, π and named
symbols with rational exponents, kept in one canonical form whose text is its
identity. `km` and `1e3 m` are one unit, and a conversion is one correctly
rounded multiply, or it raises.

## Quick start

```ocaml
open Ymir_units

let speed = Unit.(kilo metre / second)
let () = print_endline (Unit.to_string speed)          (* 1e3 m s^-1 *)

(* The factor from km/s to m/s, rounded once to float32. *)
let f = Unit.ratio Nx.float32 speed Unit.(metre / second)   (* 1000. *)

(* Exact SI constants are units: h/k_B stays exact until it is rounded. *)
let h_over_k = Unit.(planck / boltzmann)
```

## Features

- **Exact units**: `Unit.int`, `Unit.decimal`, `Unit.pi`, `Unit.symbol`,
  `Unit.scoped` and the algebra `*`, `/`, `**`, `root`
- **Canonical text**: `Unit.to_string` and a strict `Unit.of_string`, a stable
  format for files and table metadata
- **Conversion**: `Unit.ratio` rounds the exact factor once to any nx dtype,
  identically on every platform, and raises on a factor that would be 0,
  subnormal, overflow or lose integers
- **The SI**: base and derived units, prefixes from `quecto` to `quetta`, and
  the exact defining constants, with the radian as a dimension
