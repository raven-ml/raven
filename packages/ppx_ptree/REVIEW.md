# Reviewing ppx_ptree

## Design

- A special case for one type form: what does the type's structure say that
  the general rule misses?

## Semantics

Failure here is a derived `walk` that differs from the hand-written one, or
an error that points somewhere other than the user's type.

- Each type form the deriver accepts against the hand-written walk: records,
  variants, parameters, nested structures.
- A form it refuses: refused at the type, with a location.

## Skip

`test/golden/`.
