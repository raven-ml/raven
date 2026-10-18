# Reviewing rig

## Design

- Does the promise hold by construction, or by a check, a lock or a flag
  added beside it? A new state that guards an ordering is a sign the design
  could make that ordering impossible.
- Does correctness now depend on the caller keeping values alive, ordering
  calls or holding a lock?
- Core or driver: does the change put a driver's fact in core, or core's
  question in a driver?
- Code one driver gains that another already has stays duplicated. Judge
  each copy on its own.
- A removal: does it remove a capability (`AGENTS.md`) or a promise?

## Semantics

References: the vendor's interface (its headers and documentation), Linux
and POSIX for system calls, signals and `fork`, the OCaml 5 and C11 memory
models for what two domains see. tinygrad's runtime is a datapoint.

Failure here is work that runs out of the order its memory implies; a device
left inconsistent after an error, a loss or an interrupt; a resource leaked
on an error path; a hang; memory reached past its bounds; a hot path that
allocates.

- Each fallible driver call's error path: does rig's state come back, and is
  the outcome one the `.mli` states? `test_walk.ml` covers the operations it
  lists; a new operation joins it.
- C that two threads reach reads and writes atomically.
- Hand-written ABI values against the vendor's header, field by field.

## Do not report

- Duplication between drivers.
- A capability as unused, or a path as untested on hardware.

## Skip

`bench/*.thumper`, `LICENSE-*`, `test/*/fixtures/` and other `gen.sh`
output.
