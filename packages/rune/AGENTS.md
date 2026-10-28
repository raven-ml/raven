# rune

Function transformations of plain nx functions: `jvp`, `grad`, `vjp`,
`linearize`, `vmap` over named axes, `jit`, traced loops (`scan`,
`iterate`), `remat`, custom rules and totals. Structures are `Nx.Ptree`
descriptions; rune adds no tensor type.

## Naming

- A plain name takes one description per structure role: a structure in,
  a tensor out, or a signature (`Nx.Ptree.fn`) for `vmap`, `jit` and the
  loops.
- Suffix `'` takes single tensors and no description (`grad'`, `vmap'`,
  `jit'`). Never use `'` for another meaning.

## Semantics

- A transformation is an nx interpretation with one rule per `Nx.Prim`
  operation. A construct traces its functions at the call into a program,
  eagerly too, so a step's code runs once and a read or a draw without a
  key inside it raises. No transformation re-installs itself around user
  code.
- Complex gradients: the gradient of a real `f` of `z = x + iy` is
  `∂f/∂x - i ∂f/∂y`, against whose conjugate a step descends; a pullback
  applies `f'(z)` to its cotangent unconjugated, and every transpose rule is
  the plain transpose.
- Reverse mode is linearize then transpose; the tape is a linear program.
- A compiled call agrees with the eager one within the classes rune.mli's
  jit section states, one place for them.
- rune lowers each nx operation to tolk's low-level uops. tolk has no
  matmul, contraction, attention or scan node, and none is planned: never
  design a feature that needs one.
- Where the lowering builds something other than tinygrad's decomposition,
  `DIVERGENCES.md` gets an entry with its reason and the test that pins it.

## Who calls rune

Programs apply transformations; libraries never call `jit`. A library
depends on rune only for what it alone can state about its own results:
`detach`, custom rules, `remat`, a loop that must be traced. Everything
else takes nx functions.

## Checks

A change to loops, to programs or to what rune hands tolk's scheduler also
runs `dune build @packages/jera/runtest`. jera's compiled marches have
caught what rune's suites missed.
