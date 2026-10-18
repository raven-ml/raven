# Reviewing ymir

## Design

- A conversion or rule that rune could derive from the formula.
- A special case for one frame, projection or unit that the general
  transform should hold.

## Semantics

References: the FITS 4.0 standard, NIST SP 811 and CODATA, and the papers
each `.mli` cites (the cosmology's parameters).

Failure here is a number off by a unit, a prefix or a frame: a quantity
converted twice, a direction in the wrong frame, a pixel mapped to the wrong
sky position.

- Units: exact rationals stay exact; prefixes on composite units.
- Projections at the poles, at the edges of their domain and across 0°/360°.
- FITS: what a header and data unit read back as written.
