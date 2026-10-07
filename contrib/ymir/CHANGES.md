# Changelog

All notable changes to Ymir are documented in this file.

## Unreleased

- `Ymir.Frame` names the fixed celestial frames, ICRS, FK5 at J2000, Galactic,
  the J2000 ecliptic and supergalactic, with one value per frame type.
  `Frame.matrix` converts between any two from their standards' orientations,
  each rounded once, so a conversion never depends on a path.
- `Ymir.Direction` holds float64 directions in a frame: `lonlat`, `of_xyz`,
  `lon`, `lat`, `rotate`, `separation` and `position_angle`. Readers accept any
  finite vector as its direction, keep relative accuracy at any separation and
  give derivative 0, never NaN, where no derivative exists.
- New library `ymir`, astronomy over nx: `open Ymir` brings `ymir.units`'
  modules and `Ymir.Units`, astronomy's units, with `Units.parsec`
  (648000/π au) and `Units.julian_year` (31557600 s), both exact.
- New package: exact physical units. `Ymir_units.Unit` is a unit as an exact
  monomial over primes, π and named symbols with rational exponents, with one
  canonical text (`to_string`, strict `of_string`), the SI's units, prefixes
  and exact defining constants, and `Unit.ratio`, a conversion factor
  correctly rounded to any nx dtype. `Ymir_units.Quantity` is a tensor in a
  unit, an `Nx.Ptree.S` that reports its unit's canonical text, with `value`
  and `convert` multiplying by that factor and raising on integer overflow.
- `Ymir_units.Constant` is a measured constant, a published decimal with its
  uncertainty, rounded once to a payload's dtype. `Ymir_units.Codata` holds
  the CODATA 2018 and 2022 releases, transcribed from NIST's tables.
- `Ymir_units.Vocabulary` names units: `lookup` reads a symbol with an SI
  prefix, `spell` and `pp` write a unit with a vocabulary's symbols, and
  `Vocabulary.si` holds the SI's.
