# Ymir

Astronomy for OCaml, built on [Nx](../../packages/nx/): exact physical
units, and the background cosmology.

`ymir.units` gives a unit as an exact value: a product of primes, π and named
symbols with rational exponents, kept in one canonical form whose text is its
identity. `km` and `1e3 m` are one unit, and a conversion is one correctly
rounded multiply, or it raises. A quantity is a tensor in a unit, a structure
that `jit`, `vmap`, `scan` and `jvp` carry with no rule of their own.

## Quick start

```ocaml
open Ymir_units

let speed = Unit.(kilo metre / second)
let () = print_endline (Unit.to_string speed)          (* 1e3 m s^-1 *)

(* The factor from km/s to m/s, rounded once to float32. *)
let f = Unit.ratio Nx.float32 speed Unit.(metre / second)   (* 1000. *)

(* Exact SI constants are units: h/k_B stays exact until it is rounded. *)
let h_over_k = Unit.(planck / boltzmann)

(* A tensor in km/s, read in m/s: one multiply by 1000. *)
let v = Quantity.v speed (Nx.create Nx.float32 [| 2 |] [| 1.; 2.5 |])
let v_si = Quantity.value Unit.(metre / second) v
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
- **Quantities**: `Quantity.v`, `value` and `convert`, the algebra `add`,
  `sub`, `mul`, `div`, `pow`, `root`, `times` and `per`, and payloads of every
  float, complex and integer dtype; an integer conversion raises rather than
  wrap
- **Measured constants**: `Constant.v` reads the published notation
  `6.67430(15)e-11`, and `Codata.v2018` and `Codata.v2022` hold the CODATA
  releases; a constant rounds once to the dtype a program asks for
- **Names**: `Vocabulary.lookup` reads a symbol with an SI prefix (`MJy`),
  `Vocabulary.spell` and `pp` write a unit with a vocabulary's symbols, and
  `Vocabulary.si` holds the SI's
- **Cosmology**: `Cosmology.t`, one record of tensors for flat and curved
  ΛCDM, wCDM and w0waCDM with radiation and massive neutrinos; the
  expansion rate, density parameters, distances, volumes and times as fixed
  Gauss–Legendre sums with a stated error bound, batched, compiled and
  differentiable in every parameter; and the Planck and WMAP realisations
