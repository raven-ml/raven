# rune

Function transformations over plain nx functions: `grad`, `jvp`, `vjp`,
`vmap`, `jit`, staged loops. Structures are `Nx.Ptree.S` modules; rune adds
no tensor type.

## Naming

- A plain name takes one structure module: a structure in, a tensor out.
- Suffix `2` adds a module for the output structure.
- Suffix `'` takes single tensors and no module (`grad'`, `vmap'`, `jit'`).
  Never use `'` for another meaning.
- When a transformation gains a role, the common case gets a `'` form and
  the general form takes one module per structure role.

## Semantics

- Complex gradients: `grad` returns dL/dre + i·dL/dim, so `z - lr * g`
  descends. The tape holds the conjugate, so reverse rules are
  conjugation-free; `grad`, `vjp`, `vjp_fun` and `custom_vjp`'s backward
  conjugate at the boundary.
- A compiled call agrees with the eager one. Two exceptions, both stated in
  the `.mli`: `max` and `min` are exact up to the sign of a zero result, and
  a reduction's accumulator may fuse its multiply and add, written as
  `Mulacc` in the IR and never through compiler flags.
- rune lowers each nx operation to tolk's low-level uops. tolk has no
  matmul, contraction, attention or scan node, and none is planned: never
  design a feature that needs one.
- Where the lowering builds something other than tinygrad's decomposition,
  `DIVERGENCES.md` gets an entry with its reason and the test that pins it.

## Who calls rune

Programs apply transformations; libraries never call `jit`. A library
depends on rune only for what it alone can state about its own results:
`detach`, `root`, `custom_jvp`, remat, a loop that must stage. Everything
else takes nx functions.

## Checks

A change to loop staging or to what rune hands tolk's scheduler also runs
`dune build @packages/jera/runtest`. jera's compiled marches have caught
what rune's suites missed.
