# rune

Function transformations of plain nx functions: `jvp`, `grad`, `vjp`,
`linearize`, `linear_transpose`, `vmap` over named axes, the constructs
(`scan`, `iterate`, `switch`, `root`, `remat`, `jit`), custom rules, totals
and probes. Structures are `Nx.Ptree` descriptions; rune adds no tensor type.

## Structure

- `rune.ml` holds the entry points. Every other module is private but
  `loops` and the lowering's (`lib/dune`).
- `trace.ml` records a construct's functions as a `Program.t`; `program.ml`
  holds programs and their constructs and evaluates them.
- `jvp.ml`, `linear.ml` and `vmap.ml` are `Nx.Prim` interpretations, with one
  rule per operation and one per construct. A construct's rule transforms its
  program: the derivative of a scan is a scan of its step's derivative.
  `scalar.ml` derives fused scalar programs, `factor.ml` the factorisations,
  `custom.ml` runs the custom rules, `loops.ml` takes loops' programs apart.
- Reverse mode linearizes under `jvp` into a tape, a linear program, and
  transposes it (`linear.ml`).
- `jit.ml` keeps one program per key, in `cache.ml`'s tables; `compile.ml`
  compiles a program once per variant through tolk, its loops through
  `loop.ml`; `lower*.ml` lower each nx kernel operation to tolk UOps.

## Naming

- A plain name takes one description per structure role: a structure in,
  a tensor out, or a signature (`Nx.Ptree.fn`) for `vmap`, `jit` and
  `remat`. The loops take a description per role.
- Suffix `'` takes single tensors and no description (`grad'`, `vmap'`,
  `jit'`). Never use `'` for another meaning.

## Semantics

- A construct traces its functions at the call, eagerly too, so a step's code
  runs once per trace, and `Nx.item` of a traced value or a draw without a
  key inside it raises. No transformation re-installs itself around user
  code.
- Outside every transformation a construct other than `jit` agrees bit for
  bit with its functions applied eagerly. A compiled call agrees with eager
  within the classes rune.mli's Numerics section states, the one place for
  them.
- Complex gradients: the gradient of a real `f` of `z = x + iy` is
  `∂f/∂x - i ∂f/∂y`, against whose conjugate a step descends; a pullback
  applies `f'(z)` unconjugated, and every transpose rule is the plain
  transpose.
- tolk has no matmul, contraction, attention or scan node, and none is
  planned: never design a feature that needs one.
- A departure from JAX's meaning of a transformation, or from tinygrad's
  decomposition in the lowering, gets a `DIVERGENCES.md` entry naming its
  function, its reason and the test that pins it.

## Who calls rune

Programs apply transformations; libraries never call `jit`. A library
depends on rune only for what it alone can state about its own results:
`detach`, custom rules, `remat`, a loop that must be traced. Everything
else takes nx functions.

## Tests

- Laws, by `prop`: a derivative rule against finite differences, at real
  and complex dtypes, composed with `vmap` and `jit`; `jit` equal to eager
  within its class; a construct equal to its loop or branch written out.
- The lowering's suites (`test_lower_*.ml`) compile each lowered graph on
  the host and compare it with nx's eager result.
- The special functions' derivative goldens come from
  `uv run packages/rune/test/gen/special_grad.py`; `--check` compares.
- A change to loops, to programs or to what rune hands tolk's scheduler also
  runs `dune build @packages/jera/runtest`. jera's compiled marches have
  caught what rune's suites missed.

## Benches

`bench/bench_rune2.ml` times what a user calls (traces, loops, compiled
steps, gradients), `bench/compile/` a process's cold first call. Baselines
are `rune2.thumper` and `compile/compile.thumper`, recorded on fudge.
