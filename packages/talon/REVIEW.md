# Reviewing talon

## Design

- A branch for one column type or one format in the query layer: is a fact
  missing from the kinds or from the format's description?
- A reader and writer that drift apart: does one description drive both?

## Semantics

References: RFC 4180 for CSV, the Apache Parquet format specification,
ISO 8601 and RFC 3339 for datetimes, RFC 9636 for time-zone files,
IEEE 754-2019 for floats.

Failure here is a value read or written wrong: a float rounded twice, a
datetime that does not read back as printed, a null that becomes a value
through a join, a span added in the wrong unit.

- Print then parse returns the same value, for every column type.
- Nulls through joins, group-bys and layouts.
- Datetimes at their ticks' bounds, signed years, time zones.
- Decimals at the meeting scale of their operands.

## Skip

`test/golden/`, `bench/data/`, `bench/talon.thumper`.
