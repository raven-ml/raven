# The benchmark suite

This directory holds talon's release gate:
the H2O db-benchmark questions and the 22 TPC-H queries, timed on DuckDB and
Polars, with the answers that CI checks.

A workload passes when **talon's median time is at most the faster
baseline's**, with every engine using every core of the machine that recorded
the baselines. The gate sizes are H2O at 10⁷ and 10⁸ rows and TPC-H at scale
factors 1 and 10. CI runs the same queries at 10⁶ rows and scale factor 0.1
and compares their answers with the committed DuckDB answers.

## Files

| File | Role |
|---|---|
| `h2o_data.py`, `tpch_data.py` | Generate the data |
| `baseline.py` | Time DuckDB and Polars into a thumper file |
| `answers.py` | Check both engines against the committed answers, or record them |
| `h2o.py`, `tpch.py` | The questions, in DuckDB SQL and as Polars queries |
| `h2o.ml`, `tpch.ml` | The questions in talon |
| `workload.py` | Tables, questions, and the two engines |
| `answer.py` | The canonical form of an answer, its files, and comparison |
| `thumper.py` | Thumper's baseline format and machine key |
| `h2o-baselines.thumper`, `tpch-baselines.thumper` | The baselines, once recorded |
| `answers/` | DuckDB's answers at the CI size |
| `bench_h2o.ml`, `bench_tpch.ml`, `cases.ml` | Time talon into `h2o.thumper` and `tpch.thumper` |
| `runner.ml` | Write talon's answer to a question, check talon's answers |
| `answer.ml`, `workload.ml` | The OCaml twins of `answer.py` and `workload.py` |

Every script pins the same environment in its inline metadata: Python 3.13,
DuckDB 1.5.6, Polars 1.44.2, numpy 2.5.3 and pyarrow 25.0.1, resolved with
`exclude-newer = 2026-10-01`. `uv run` builds it on first use.

## Data

```sh
uv run h2o_data.py 1e7 DATA     # DATA/h2o-1e7/, about 1.4 GB
uv run h2o_data.py 1e8 DATA     # DATA/h2o-1e8/, about 14 GB
uv run tpch_data.py 1 DATA      # DATA/tpch-sf1/, about 0.3 GB
uv run tpch_data.py 10 DATA     # DATA/tpch-sf10/, about 3 GB
```

DATA is any directory outside the repository. The CI sizes are `1e6` and
`0.1`.

**H2O** follows db-benchmark's `groupby-datagen.R` (K = 100, no missing
values, unsorted) and `join-datagen.R`, written as CSV under db-benchmark's
file names. The values come from numpy's PCG64 seeded with 108. They differ
from R's but follow the same distributions, and they are identical on every
machine for the pinned numpy.

**TPC-H** comes from DuckDB's tpch extension, which runs the reference dbgen.
The tables are written as snappy Parquet, with every `DECIMAL(15,2)` column
(prices, costs, balances, quantities, discounts, taxes) as float64.

## Protocol

- **Loading.** H2O's tables load before timing, from CSV into memory, with the
  same declared schema in every engine: identifiers as strings, other integers
  as int32, measures as float64. TPC-H's queries scan the Parquet files inside
  the timed region, and DuckDB's external file cache is off so that it reads
  them on every run, as Polars does.
- **Timed region.** One question, from its query to its answer materialized in
  memory: an Arrow table from DuckDB, or a collected Polars frame. The previous
  answer is freed and the garbage collector runs before the clock starts.
- **Samples.** Each question runs once to warm up, then five times (`--runs`).
  The runner prints each case's median and its spread, (max − min) / median.
- **Threads.** DuckDB is set to one thread per core, and Polars must run its
  default pool of one thread per core.
- **Engines.** Polars runs its default engine, `auto`, which in 1.44.2 is the
  streaming engine; the runner refuses to start when `POLARS_ENGINE_AFFINITY`
  would select another.
- **Memory.** DuckDB spills to `DATA/duckdb-tmp`. A case during which the
  machine swapped pages in or out (`vm_stat` on macOS, `/proc/vmstat` on
  Linux) is not recorded, since its time measures the disk.

## Baselines

```sh
uv run baseline.py h2o 1e7 DATA
uv run baseline.py h2o 1e8 DATA
uv run baseline.py tpch 1 DATA
uv run baseline.py tpch 10 DATA
```

Each run writes its cases into this machine's section of
`h2o-baselines.thumper` or `tpch-baselines.thumper`, keeping the cases it did
not run. A case id is `WORKLOAD/SIZE/QUESTION/ENGINE`, such as
`groupby/1e8/q03/polars`, `join/1e7/q05/duckdb` or `tpch/sf10/q21/duckdb`.

The files follow thumper's baseline format, and the section's machine key is
the one thumper computes for the same host. The `# ocaml:` line, which the
format requires, reads `none`. The section's annotations pin the engines'
versions, Polars' engine, the thread count, the memory, the Python version and
the data's generator (numpy's version and the seed for H2O, DuckDB's dbgen for
TPC-H). A run under other pins is refused before it measures, so a section
never mixes versions; to move to new engines, delete the section and re-run
every size.

A run refuses to start when the 1-minute load average is at least half the
cores, as thumper's gate does. `--allow-load` measures anyway, and only into an
`--output` other than the committed files, so loaded numbers never gate.

A run exits nonzero when a case swapped, which it does not record, or when a
case's spread is over 5%, which it records and lists. Neither case is fit to
gate until it is re-run on a quieter machine.

## Answers

```sh
uv run answers.py DATA           # check both engines
uv run answers.py DATA --record  # check that Polars agrees with DuckDB, then record DuckDB
```

Recording keeps every committed answer that still agrees with DuckDB's, so a
float that DuckDB sums in another order does not rewrite its file.

An answer's canonical form has integers as int64 and floats as float64. H2O's
answers are unordered, so their rows are sorted by every column in turn,
ascending, with nulls last; TPC-H's keep their query's order.

A committed answer keeps the rows at positions 0, s, 2s, … of the canonical
form, with the stride s chosen to keep at most 200 rows. Answers of at most
200 rows (every TPC-H answer but query 16's) are committed whole.
`answers/index.csv` records each answer's full row count and stride. Because
rows are taken by position, a row missing or added anywhere shifts every later
sample. Floats agree within a relative 1e-9 (or an absolute 1e-12), nulls with
nulls and NaN with NaN; every other value agrees exactly.

## Talon's queries

`bench_h2o.exe` and `bench_tpch.exe` are thumper suites over the data under
`$TALON_BENCH_DATA`, one case per question of each size whose data exists, with
case ids that end in `/talon`, such as `groupby/1e8/q03/talon`:

```sh
export TALON_BENCH_DATA=DATA
dune exec packages/talon/bench/d7/bench_h2o.exe -- list
dune exec packages/talon/bench/d7/bench_h2o.exe -- check -f groupby/1e7
dune exec packages/talon/bench/d7/bench_tpch.exe -- bless -f tpch/sf1
```

Each case loads its workload's tables before thumper measures it, and each
measured call runs the question to its answer, materialized as one table.
Thumper writes `h2o.thumper` and `tpch.thumper` under the machine key the
baselines use, with its own protocol of calibrated batches and a collection
between samples. Talon loads H2O's tables with the same declared schema as the
baselines, identifiers as `String`, never as `Categorical`.

`runner.exe` writes and checks talon's answers:

```sh
R=_build/default/packages/talon/bench/d7/runner.exe
$R questions groupby/1e7                      # the ids of a workload's questions
$R answer DATA tpch/sf1/q21 q21.csv           # write one canonical answer
$R check packages/talon/bench/d7/answers DATA   # check every CI answer
```

TPC-H's talon queries state their joins in a committed order, since talon does
not reorder joins; the baselines use each engine's own planning.

`check` runs every question at the CI size and compares its answer with the
committed one, as `answers.py` does. CI runs it on Linux after generating the
data, so it needs uv, Python, and network access: uv fetches the pinned
packages, and DuckDB's `INSTALL tpch` downloads the extension.

The comparison reads talon's section and the baselines' section with the same
machine key. A question passes when the median of `…/talon` is at most the
smaller of the medians of `…/duckdb` and `…/polars`.

## Differences from db-benchmark and dbgen

- Every engine loads the same schema. db-benchmark's solutions tune their own:
  its DuckDB stores `v3` as float32 and `id1`, `id2` as enums, and its Polars
  casts identifiers to categoricals and join measures to float32.
- Polars runs its default engine (streaming, in 1.44.2) on frames read into
  memory. db-benchmark's Polars solution streams from memory-mapped IPC files.
- DuckDB's answer is fetched as an Arrow table. db-benchmark writes it into a
  table (`CREATE TABLE ans AS …`), which costs DuckDB work Polars does not do.
- `join-datagen.R` refuses fewer than 10⁷ rows. The port splits keys with
  integer counts, so at 10⁶ rows the small table holds one key and every left
  row matches it.
- TPC-H's SQL is the text of DuckDB's tpch extension, except query 11, whose
  fraction is the specification's 0.0001 / SF instead of the scale-factor-1
  constant. The Polars queries are written for this suite.
