# tolk tests

## Layout

- `<path>/` holds the suite of `lib/<path>.ml`, one executable named after
  the file: `dtype/test_dtype.ml`, `uop/op/test_op.ml`. A suite
  tests its module through its interface, and its goldens sit beside it.
- `support/` is the `tolk_test` library that every suite links: its
  `Golden` module turns goldens into tests. Generators, witnesses and law
  combinators over tolk types join it as modules of their own, so that a
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
 (libraries windtrap tolk tolk_test)
 (deps
  (glob_files *.golden))
 (action
  (run %{test} --exclude-tag slow)))
```

## Slow tests

`dune build @packages/tolk/runtest` is the default run, and it skips
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

`dune build @packages/tolk/slow` runs them all. A suite without slow
tests has no such rule, since `--tag slow` then selects nothing, which windtrap
reports as a failure.

## Hardware checks

What no machine of the default run could check, to verify on the hardware
named:

- **NVIDIA:** a CUDA kernel stores an infinity converted to e5m2 as an
  infinity and to e4m3 as a NaN of its sign (D16), and `__half` and
  `nv_bfloat16` operations compute in their own type, as `cuda_fp16.hpp` and
  `cuda_bf16.hpp` declare (D17).
- **Contraction (D25):** a CUDA kernel compiled with `--fmad=false` and a HIP
  kernel compiled with `-ffp-contract=off` compute `a*b + c` with two
  roundings: `(1 + 2^-12)^2 - (1 + 2^-11)` is 0, not `2^-24`.
- **Widened tensor-core operands (D29):** on CUDA and HIP, a `float32`
  matrix product of `half` and `bfloat16` values widened first, run on the
  narrow-in, `float32`-out core, equals the same product computed in
  `float32`.
- **Division (D50):** a CUDA kernel compiled by NVRTC and a HIP kernel
  compiled by comgr, at their default flags, divide floats with `/` as IEEE
  does, rounding once: `3 / 7` is `0x3edb6db7`, where the product by the
  reciprocal is `0x3edb6db8` (`division (D50)`'s operands).
- **AMD gfx950:** `f32_to_fp8`'s non-saturating `cvt_pk_{fp8,bf8}_f32` makes
  an infinity a NaN in fp8 and keeps it in bf8 (D16).
- **CUDA (nx.cuda.device):** a device waits with the `Sleep` completion: the
  runtime polls the signal word, and after 200 ms of stillness the sleep
  queries both streams each millisecond, so a fault loses the device with the
  driver's error and a kernel that never signals hangs after the timeout
  (`test_cuda.ml`'s fault test, `test_cuda_hang.ml`).
- **AMD and NV (nx.amd.device, nx.nv.device):** the queue and channel words
  are buffers at the address the host and the GPU share, and the device's
  own submissions (NV's engine binding and local memory) run through
  `Nx_device.submit` (`test_amd_hw.ml`, `test_nv_hw.ml`).
- **The 32-bit carry on AMD and NV:** a device's timeline crossing 2^32 and
  2^33. AMD's SDMA work waits for the low half of the signal word to equal
  v - 1 and writes the low half, then the high half when the low one is 0;
  NV's copy channel releases the low word, then the high word, while its
  compute channel releases all 64 bits. No waiter may pass early mid-write,
  and a late high word may never take the word back below another
  channel's value.
- **NV batches (Ops_nv, D38, D40, D42, D51):** `test_ops_nv_exec` (the `slow` alias) on
  an NVIDIA GPU: chained launches, copies on the copy engine inside a batch,
  runs without synchronizing, profiled spans and each trip of a range on its
  own descriptors. The copy channel's second release lands at the signal
  word's high half or at the device's sink word, as its first release's low
  word says (D40), launches read descriptors, constant buffers and
  arguments from pinned memory (D42), the program the engine loads through
  nx.nv.device (D38), and the local memory size from the device's word
  (D51).

## Coverage and mutation

The library carries windtrap's coverage and mutation backends, which do
nothing until a build asks for them. For one module, here `Op`:

```sh
# Coverage: run the suite instrumented, then report each file's uncovered code.
dune build @packages/tolk/test/uop/op/runtest --instrument-with ppx_windtrap.coverage
dune exec windtrap -- coverage -u

# Mutation: each SURVIVED block is a change to the module that no test noticed.
dune exec --instrument-with ppx_windtrap.mutate \
  packages/tolk/test/uop/op/test_op.exe -- --mutate=packages/tolk/lib/uop/op.ml
```

A suite reads its goldens beside its executable, where dune copies them when
the suite runs, so run it once with `runtest` before running it with
`dune exec`.

Each mutant runs in a child forked after a dry run of the suite, so a module
that memoises its results (Divandmod's `fold_divmod_general`) answers the
mutant from the dry run's cache: its survivors and unevaluated mutants are not
verdicts. Rerun each with `--arm <site>`, which starts with an empty cache.

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
      equal dtype (dtype_of_cell (cell "least_upper"))
        (Dtype.least_upper [ dtype_of_cell (cell "a"); dtype_of_cell (cell "b") ]))
  ```

  `Golden.columns` and `Golden.rows` give the table itself, for a claim about
  the whole table, such as that it has no column the suite does not check.

A golden is tinygrad's output and is never edited by hand. When a ledger entry
changes an output, the test that compares it states the difference in code and
names the entry, and it is that entry's pinning test.

## Graphs

A test's input graph is built with the UOp constructors, or read from an
**input golden**: a generator runs a tinygrad `Tensor` program and records the
graph that the program hands to the compiler (`graph.boundary`), or any graph
it builds. An output golden of a UOp graph (G-list) uses the same format, so
tolk has one text form of a graph.

In a suite, `Golden.sink "matmul.golden"` is the graph of an input golden, and
`Golden.graph "kernel.golden" (fun () -> sink)` is the test that a graph is an
output golden; its failure is a diff of the two texts. The library's
`Graph.of_string` and `Graph.to_string` read and write the text itself, which
is also the form of the programs and schedules the disk cache keeps. Reading a node checks that
its written dtype is the one tolk derives from its op, sources and arg.

### The graph format

The body of a graph golden lists the nodes under a sink, one per line, in
topological order:

```
<index> <op> <dtype> [<sources>] <arg> tag=<tag>
```

```
4 Ops.PARAM dtypes.half [] ParamArg(slot=1, dtype=dtypes.half, size=16, device="CPU")
5 Ops.CONST dtypes.weakint [] 16
6 Ops.RANGE dtypes.weakint [5] (0, AxisType.REDUCE)
7 Ops.INDEX dtypes.half [4, 6]
8 Ops.CAST dtypes.float [7] dtypes.float
9 Ops.REDUCE dtypes.float [8, 6] (Ops.ADD, 0)
```

- **index** is the line's position, from 0. A node comes after its sources
  and after the nodes that its arg holds. Sources are visited first, in order,
  so a graph whose args hold no node is in `UOp.toposort` order.
- **op** prints as `Op.pp` does, and **dtype** as `Dtype.pp` does. A node's
  dtype follows from its op, sources and arg; it is written so that a reader
  can check that it derives the same one.
- **sources** are indices of earlier lines.
- **arg** is left out when it is `None`, and so is ` tag=<tag>` when the tag
  is `None`. Both are values:

| Value | Written as |
|---|---|
| none, booleans | `None`, `True`, `False` |
| integer | decimal, of any size: `-3`, `340282366920938463463374607431768211456` |
| float | the shortest decimal that reads back as the same double, as Python's `repr` writes it: `1.0`, `-0.0`, `1e-05`, `1e+16`, `inf`, `-inf`; `nan` for the positive quiet NaN, and any other NaN as its bits in hex: `nan(0xfffffc0000000000)` |
| the invalid constant | `Invalid` |
| string | in double quotes; `"`, `\`, newline and tab escaped as `\"`, `\\`, `\n` and `\t`, and every other byte outside printable ASCII as `\xHH` |
| bytes | as a string, prefixed with `b`: `b"\x7fELF"` |
| data type | as `Dtype.pp` prints it: `dtypes.half` |
| enumeration | `<Enum>.<NAME>`: `Ops.ADD`, `AxisType.REDUCE`, `AddrSpace.GLOBAL` |
| node | `%` and its index: `%12` |
| tuple | `()`, `(v,)`, `(v, w)` |
| record | `<Name>(<field>=<value>, ...)`, fields in declaration order, each field left out when it holds its default |

A record's fields that are runtime state are never written: `ParamArg.buffer`
holds a device buffer, which `tolk.engine` owns. A graph whose `CallInfo`
holds a gradient function cannot be written, since tolk does not
differentiate.
Node metadata, a side table of debugging names outside the node, is not
written either.

Hash-consing makes each node unique, so a graph has one text: two equal
graphs are two equal goldens, and a diff starts at the first node that
differs.

### Arg kinds

These are the args tinygrad's operations take. The list is the union of a
reading of `uop/ops.py`, `schedule/*`, `codegen/*` and
`runtime/support/hcq2.py` with a count of every arg built while tinygrad's
tests run: all of `test/null` but `test_disk_cache`, and `test_ops`,
`test_tensor`, `test_getitem_ops` and `test_symbolic_ops` of `test/runtime` on
the CPU. The writer wrote every sink, linear and program those runs built,
about 330,000 graphs, and refused none.

| Op | Arg |
|---|---|
| the ALU ops, `INDEX`, `SHRINK`, `STACK`, `RESHAPE`, `EXPAND`, `PAD`, `AFTER`, `GROUP`, `END`, `BARRIER`, `IF`, `ENDIF`, `NOOP`, `BACKEDGE`, `MSTACK`, `DETACH`, `CONTIGUOUS_BACKWARD` | none |
| `LOAD`, `STORE` | none, or a string: a flag of the access, such as `"nontemporal"` |
| `CONST` | an integer, a boolean, a float or `Invalid`; its dtype follows from the kind: `dtypes.weakint`, `dtypes.bool`, `dtypes.weakfloat` and `dtypes.bool` |
| `CAST`, `BITCAST` | the target data type |
| `PERMUTE` | a tuple of integers, the order of the axes |
| `FLIP` | a tuple of booleans, one per axis |
| `REDUCE` | `(<op>, <count>)`: the reducing operation and how many leading axes it reduces, as `(Ops.ADD, 1)` |
| `ALLREDUCE` | `(<op>, <device>)` |
| `RANGE` | the axis id, one or more integers, then its `AxisType`: `(0, AxisType.LOOP)`, `(1, 0, AxisType.REDUCE)` |
| `SPECIAL` | a string, the name of the launch dimension: `"gidx0"` |
| `UNSHARD` | a tuple of one integer, the axis |
| `MSELECT` | an integer, the device's position |
| `COPY`, `GETADDR` | a device |
| `LINEAR` | none, a string, or `(<devices>, <queue>)` |
| `CUSTOM`, `CUSTOMI` | `(<code>, <data type>)`, the code a string |
| `INS` | `(<instruction>, <data type>)`, the instruction a string |
| `CUSTOM_FUNCTION`, `SOURCE` | a string |
| `BINARY` | bytes |
| `WMMA` | `((<N>, <M>, <K>), <input data type>, <threads>, <upcast axes>)`, the upcast axes a tuple of tuples of `(<axis>, <size>)` pairs |
| `PARAM`, `BUFFER`, `ALLOC` | `ParamArg` |
| `CALL` | `CallInfo` |
| `SINK` | none, or `KernelInfo` |
| `STAGE` | none, or `BufferizeOpts` |
| `PROGRAM` | `ProgramInfo` |

The records, with their fields in order:

| Record | Fields |
|---|---|
| `ParamArg` | `slot` integer, `dtype`, `size` integer or none, `vmin_vmax` a pair of constants, `multiple_of` integer, `name` string, `addrspace` `AddrSpace`, `device` a device, `volatile` boolean, `image` (always none: images are excluded), `bind_on_realize` boolean, `val` integer, `phase` integer (tolk's own field, D54, never in tinygrad's goldens); `buffer` is runtime state and never written |
| `KernelInfo` | `name` string, `applied_opts` and `opts_to_apply` tuples of `Opt` (the second may be none), `estimates` `Estimates` or none, `beam` integer |
| `Opt` | `op` `OptOps`, `axis` integer or none, `arg` an integer, a tuple, or none |
| `Estimates` | `ops`, `lds`, `mem`: each an integer or a node |
| `ProgramInfo` | `global_size` and `local_size` tuples of integers, `vars` a tuple of nodes, `globals`, `outs` and `ins` tuples of integers, `target` `Target` |
| `Target` | `device`, `renderer`, `arch`, `interface`, `indices`: strings |
| `CallInfo` | `grad_fxn` (never written), `name` string or none, `precompile` and `precompile_backward` booleans, `aux` none or `HCQInfo`, `dtype` data type |
| `BufferizeOpts` | `device` a device or none, `addrspace` `AddrSpace`, `removable` and `inlinable` booleans |
| `HCQInfo` | `device` a tuple of strings, `kernels` a tuple of `(<devices>, <name>, <estimates>, <stamp slots>, <profile key>, <input slots>, (<outs>, <ins>))`, `estimates` `Estimates`, `nargs` and `table` integers, `inputs` a tuple of `(<node>, <integer>, <string>)`, `slots` a tuple of `(<string>, <integer>)`, `written_bufs` a tuple of nodes |

A device is a string, or a tuple of strings for a sharded value. The
enumerations are `Ops`, `AxisType`, `AddrSpace` and `OptOps`. A tag is a
boolean, an integer, a string, bytes (Metal's indirect command buffers tag
with them), a data type, or a tuple of these, such as `(0, dtypes.int)`; it
never holds a node. `PYLITERAL` and `REWRITE_ERROR` are
excluded operations, and the arg kinds of the excluded ISA path (register
records, integer instructions) are not part of the format.

The format serves the G-list goldens too, in place of the listing
`uop/render.py` prints. That listing drops information: it inlines constants
into their users, leaves out tags, and prints the nodes inside an arg as
expressions. A golden in the graph format pins the whole graph, and a
mismatch reads as a diff of nodes. The listing is `Render`'s own output, and
its goldens test `Render`.

## Generating goldens

`gen/<path>.py` generates the goldens of `<path>/`. It declares
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
`listing(uops)` is the listing tinygrad prints for a list of UOps. `@graph`
returns a sink, which `gen/graph.py` writes in the graph format;
`graph.boundary(*tensors)` is the sink a `Tensor` program hands to the
compiler:

```python
from golden import graph
from graph import boundary

@graph
def matmul():
    a, b = Tensor.empty(4, 8, device="CPU"), Tensor.empty(8, 3, device="CPU")
    return boundary(a @ b)
```

A codegen pass's input is recorded where the pipeline hands it over:
`graph.kernels(*tensors)` is the kernels a `Tensor` program compiles, and
`graph.stage(name, kernel, renderer)` is the sink that compiling `kernel` for
`renderer` hands to the pass `name`: a `graph_rewrite` by its name,
`"apply_opts"` (the kernel that kernel optimization receives) or
`"linearize"`.

`gen/generate.py` runs each generator in a fresh interpreter, with a scrubbed
environment, and makes each golden in a process forked from it, so that no
golden sees the state another left in tinygrad, such as its buffer numbering.
It runs against the tinygrad checkout `_tinygrad_next` of the main
working tree, which must be clean and at the commit the script pins, through
a copy of it with `gen/tinygrad.patch` applied: the patch gives tinygrad's
rewrites the restrictions that keep IEEE and modular values, its constants
a NaN's bits (DIVERGENCES D13, D24 and D27), a selection of `-0.` beside
a load, which stays a selection (D52), its tensor-core matcher the
widened operands of D29, and its constant arithmetic the corrections of the
README's CPython rows, so that the goldens hold the graphs and values
tolk makes. No Python runs at test time. To regenerate:

1. `uv run packages/tolk/test/gen/generate.py --check` writes nothing,
   prints the diff of every golden that would change, and fails if one would.
2. `uv run packages/tolk/test/gen/generate.py` writes the goldens and
   `gen/manifest`, the list of every golden, and removes goldens no generator
   declares any more. `MODULE` arguments, such as `dtype` or `uop/op`, limit
   the run to those generators.
3. Review every changed golden with `git diff` before committing it, and
   explain each change from tinygrad's source.

To move to another tinygrad commit, check it out in `_tinygrad_next`, change
`TINYGRAD` in `gen/generate.py`, re-apply `gen/tinygrad.patch` to the new
commit and save it again (`git diff` in a patched copy), and regenerate: every
golden changes its header, and the diff shows what else moved.
