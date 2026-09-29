# tolk.next tests

## Layout

- `<module path>/` holds the suite of `lib/<module path>.ml`, one executable
  named after the module: `dtype/test_dtype.ml`, `uop/op/test_op.ml`. A suite
  tests its module through its interface, and its goldens sit beside it.
- `support/` is the `tolk_next_test` library that every suite links: its
  `Golden` module turns goldens into tests. Generators, witnesses and law
  combinators over tolk.next types join it as modules of their own, so that a
  law test is one line too. Its `.golden` files are fixtures of its own suite, not tinygrad
  output.
- `gen/` generates the goldens from tinygrad.
- `REGRESSIONS.md` maps each old tolk and tinygrad test to the test that
  replaces it. `../DIVERGENCES.md` is the divergence ledger.

A suite's stanza:

```lisp
(test
 (name test_dtype)
 (package tolk)
 (libraries windtrap tolk.next tolk_next_test)
 (deps
  (glob_files *.golden))
 (action
  (run %{test} --exclude-tag slow)))
```

## Slow tests

`dune build @packages/tolk/next/runtest` is the default run, and it skips
every test tagged `slow`: each stanza runs its suite with
`--exclude-tag slow`. Declare with `slow` (or `~tags:["slow"]`) whatever is
heavy: compiling more than a few kernels, GPUs, end-to-end runs, the
optimisation fuzzer. Each module's default suite runs in 2 s or less, and the
whole default run in 60 s or less.

A suite that has slow tests adds a rule to the `slow` alias:

```lisp
(rule
 (alias slow)
 (package tolk)
 (deps
  (glob_files *.golden))
 (action
  (run %{exe:test_dtype.exe} --tag slow)))
```

`dune build @packages/tolk/next/slow` runs them all. A suite without slow
tests has no such rule, since `--tag slow` then selects nothing, which windtrap
reports as a failure.

## Coverage and mutation

The library carries windtrap's coverage and mutation backends, which do
nothing until a build asks for them. For one module, here `Uop.Op`:

```sh
# Coverage: run the suite instrumented, then report each file's uncovered code.
dune build @packages/tolk/next/test/uop/op/runtest --instrument-with ppx_windtrap.coverage
dune exec windtrap -- coverage -u

# Mutation: each SURVIVED block is a change to the module that no test noticed.
dune exec --instrument-with ppx_windtrap.mutate \
  packages/tolk/next/test/uop/op/test_op.exe -- --mutate=packages/tolk/next/lib/uop/op.ml
```

A suite reads its goldens beside its executable, where dune copies them when
the suite runs, so run it once with `runtest` before running it with
`dune exec`.

## Goldens

A golden is output recorded from tinygrad, in a file `<name>.golden`. Its
first line is `# tinygrad <commit>`, the commit it was recorded from, and the
rest is its body. A golden check is one line:

- **Text**, such as a listing of UOps or a rendered source: `Golden.text` is
  the test, named after the golden, that a text equals its body. Its failure
  prints the diff.

  ```ocaml
  Golden.text "elementwise_add.golden" (fun () -> listing kernel)
  ```

- **Table**: cells separated by tabs, the column names first, then one line
  per row. A cell is the text tinygrad prints for its value (`dtypes.half`,
  `True`, `inf`). `Golden.cases` makes one test per row, named by its key
  cells (the first column by default), and gives the check the row's `cell`
  function. A failure names the row, and one bad row hides no other.

  ```ocaml
  Golden.cases "least_upper.golden" ~key:[ "a"; "b" ] (fun cell ->
      equal dtype (dtype (cell "least_upper"))
        (Dtype.least_upper (dtype (cell "a")) (dtype (cell "b"))))
  ```

  `Golden.columns` and `Golden.rows` give the table itself, for a claim about
  the whole table, such as that it has no column the suite does not check.

A golden is tinygrad's output and is never edited by hand. When a ledger entry
changes an output, the test that compares it states the difference in code and
names the entry, and it is that entry's pinning test.

## Generating goldens

`gen/<module path>.py` generates the goldens of `<module path>/`. It declares
each golden as a function, named after the golden, with the decorators of
`gen/golden.py`:

```python
from golden import listing, table, text
from tinygrad.dtype import dtypes, least_upper_dtype

@table
def least_upper():
    return ["a", "b", "least_upper"], [(a, b, least_upper_dtype(a, b)) for a in ... for b in ...]
```

`@table` returns the columns and the rows, whose values it prints with
`repr` (strings stay as they are). `@text` returns the text, and
`listing(uops)` is the listing tinygrad prints for a list of UOps.

`gen/generate.py` runs each generator in a fresh interpreter, with a scrubbed
environment, against the tinygrad checkout `_tinygrad_next` of the main
working tree, which must be clean and at the commit the script pins. No
Python runs at test time. To regenerate:

1. `uv run packages/tolk/next/test/gen/generate.py --check` writes nothing,
   prints the diff of every golden that would change, and fails if one would.
2. `uv run packages/tolk/next/test/gen/generate.py` writes the goldens and
   `gen/manifest`, the list of every golden, and removes goldens no generator
   declares any more. `MODULE` arguments, such as `dtype` or `uop/op`, limit
   the run to those generators.
3. Review every changed golden with `git diff` before committing it, and
   explain each change from tinygrad's source.

To move to another tinygrad commit, check it out in `_tinygrad_next`, change
`TINYGRAD` in `gen/generate.py`, and regenerate: every golden changes its
header, and the diff shows what else moved.
