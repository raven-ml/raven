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
column, 10% nulls each, for the `Talon_parquet/read-nullable` row. pyarrow
writes it, so that the read row does not depend on talon's writer, and the
file is committed. Its SHA-256
is `dd79afb8ddec9ff50870181e2b9a01979ca2edacee17c1b3fbfa6de251579b14`; the
check writes it again and compares both files' digests with that one:

```bash
uv run packages/talon/bench/queries/nulls_data.py packages/talon/bench/data/nulls.parquet
uv run packages/talon/bench/queries/nulls_data.py --check
```

`data/text.parquet` holds 2^18 rows of a text column without nulls, words of 8
to 27 bytes drawn from 100, dictionary-encoded, for the
`Talon_parquet/read-text` row. pyarrow writes it too, with the SHA-256
`0f6c1c95ce0ee1052c02a7fe1e0a154a5db877c9869a7c36b53e33deb023cadf`:

```bash
uv run packages/talon/bench/queries/text_data.py packages/talon/bench/data/text.parquet
uv run packages/talon/bench/queries/text_data.py --check
```

`data/keys.parquet` holds 2^17 rows of two int64 columns without nulls, the
same ascending keys each repeated 1 to 7 times, one dictionary-encoded and
one PLAIN, for the `Talon_parquet/read-keys` row. pyarrow writes it too, with
the SHA-256 `afab11e078f1bf5142c689d176785baec0fbb85fcb074f7a794b9f7a8bade385`:

```bash
uv run packages/talon/bench/queries/keys_data.py packages/talon/bench/data/keys.parquet
uv run packages/talon/bench/queries/keys_data.py --check
```

`Talon_parquet/filter-text` reads `data/text.parquet` filtered by one of its
words. `Talon_parquet/write-nullable` and `Talon_parquet/write-text` write the
tables that `read-nullable` and `read-text` read to a sink, in the default
format.

The `Talon/Nulls/*` rows run over a table of 2^20 rows that the bench draws
from a seeded key, with nulls in five of its seven columns.

`Talon/Join/spikes_in_trials` joins 1000 trial windows of one second with
2^20 spike times in order over an hour, the trials on the left; a recording
holds about 10^7 spikes. `Talon/Join/keyed_inequality` joins 2^16 rows with
2^10 on a key of 100 values and `ge`, neither side in order.
`Talon/Join/asof-1M-64k` takes, for each of 2^20 rows, the last of 2^16 rows
at or before it within a key of 100 values, neither side in order, with
`Kit.join_asof`. `Talon/Join/asof-as-range-1M-64k` finds the same rows with a
range join, each right row holding up to the next one of its key.
`Talon/List/per-list-sum-64k-32` reduces with `Expr.per_list`, over 2^16 rows
of 0 to 64 float64 elements, each element times the row's factor.

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
