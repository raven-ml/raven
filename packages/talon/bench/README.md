# Talon Benchmarks

This directory hosts lightweight benchmarks for Talon alongside a reference
implementation in pandas: filter, group, join and sort over a 40k-row fixture.
Both load the same text columns as categoricals.

## Fixtures

`fixtures.ml` draws two synthetic CSV texts from Nx's generator, keyed by
20240127, so they are the same on every machine:

- `customers.csv`: 1.5k customer records with segments, regions, and loyalty
  metadata.
- `transactions.csv`: 40k point-of-sale transactions linked to customers with
  category, channel, discount, and amount columns.

`bench_talon.exe` draws them when it starts. `write_fixtures.exe DIR` writes
them for pandas:

```bash
dune build packages/talon/bench/write_fixtures.exe
_build/default/packages/talon/bench/write_fixtures.exe DIR
```

`data/nulls.parquet` holds 2^20 rows of a nullable float64 and a nullable bool
column, 10% nulls each, for the `Talon_parquet/read-nullable` row. talon has no
Parquet writer, so pyarrow writes it, and the file is committed. Its SHA-256
is `dd79afb8ddec9ff50870181e2b9a01979ca2edacee17c1b3fbfa6de251579b14`; the
check writes it again and compares both files' digests with that one:

```bash
uv run packages/talon/bench/queries/nulls_data.py packages/talon/bench/data/nulls.parquet
uv run packages/talon/bench/queries/nulls_data.py --check
```

The `Talon/Nulls/*` rows run over a table of 2^20 rows that the bench draws
from a seeded key, with nulls in five of its seven columns.

`Talon_csv/read-mixed` reads a CSV text of 2^16 records from memory, drawn from
a seeded key: an int64 column, a float64 column with nulls, and a string column
with quoted fields.

## Running the benchmarks

### Talon (OCaml)

```bash
dune build @packages/talon/bench/bench
```

The executable first checks each query's rows and amount against what pandas
computes, then measures. `talon.thumper` holds the per-machine baselines.

### Pandas (Python)

```bash
uv run packages/talon/bench/bench_talon.py DIR
```

Compare the two by alternating runs of each under the same load.
