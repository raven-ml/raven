# Changelog

All notable changes to Ymir are documented in this file.

## Unreleased

- New package: exact physical units. `Ymir_units.Unit` is a unit as an exact
  monomial over primes, π and named symbols with rational exponents, with one
  canonical text (`to_string`, strict `of_string`), the SI's units, prefixes
  and exact defining constants, and `Unit.ratio`, a conversion factor
  correctly rounded to any nx dtype.
