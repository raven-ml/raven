# tolk.next tests

## Layout

- `<module path>/` holds the suite of `lib/<module path>.ml`, one executable
  named after the module: `dtype/test_dtype.ml`, `uop/op/test_op.ml`. A suite
  tests its module through its interface, and its goldens sit beside it.
- `golden/` is the `Golden` library, which every suite uses to read goldens.
- `gen/` generates the goldens from tinygrad.
- `REGRESSIONS.md` maps each old tolk and tinygrad test to the test that
  replaces it. `../DIVERGENCES.md` is the divergence ledger.

A suite's stanza:

```lisp
(test
 (name test_dtype)
 (package tolk)
 (libraries windtrap tolk.next tolk_next_golden)
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

## Goldens

A golden is output recorded from tinygrad, in a file `<name>.golden`. Its
first line is `# tinygrad <commit>`, the commit it was recorded from, and the
rest is its body. There are two kinds:

- **Text**: a listing of UOps, a rendered source, any text. `Golden.text`
  reads it, and `equal text` compares it, printing a diff on failure:

  ```ocaml
  test "the linearized kernel matches tinygrad" (fun () ->
      equal text (Golden.text "elementwise_add.golden") (listing kernel))
  ```

- **Table**: cells separated by tabs, the column names first, then one line
  per row. Each cell is the text tinygrad prints for its value
  (`dtypes.half`, `True`, `inf`). `Golden.table` reads the rows, and
  `Windtrap.cases` makes each row a test named by its key columns, so a
  failure names the row and one bad row does not hide the others:

  ```ocaml
  cases "least_upper_dtype" (Golden.table "least_upper.golden")
    ~name:(Golden.key [ "a"; "b" ])
    (fun row ->
      let dtype column = dtype_of_string (Golden.cell row column) in
      equal dtype (dtype "least_upper") (Dtype.least_upper (dtype "a") (dtype "b")))
  ```

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
