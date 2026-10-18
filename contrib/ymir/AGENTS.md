# ymir

Astronomy over nx: exact units and quantities, celestial frames, FITS
files, maps from pixels to the sky, aperture photometry and the background
cosmology. Every function is a formula of nx operations, so rune batches,
compiles and differentiates it with no rule of ymir's.

- Changes go in `contrib/ymir/CHANGES.md`.
- `ymir.units` keeps canonical units. `spell` puts one prefix on the first
  word with a positive exponent (NIST SP 811); callers keep their own text.
- Constants come from NIST's CODATA tables, named as NIST names them.
