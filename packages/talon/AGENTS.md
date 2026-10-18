# talon

Tables of named, typed columns over nx, queried with typed expressions,
read from and written to CSV and Parquet.

- A file format follows Jsont's shape: one printable description drives
  both decoding and encoding, sniffing returns a description, errors carry
  locations. An invalid description raises; bad data is an `Error`.
- Sequential decoding reads a bytesrw reader and encoding writes a bytesrw
  writer. Random-access formats (Parquet) read a mapped nx buffer.
- CSV is written in raven. A sub-library that only bridges an outside format
  may depend on that format's definitive OCaml library.
- No Excel support.
- talon never depends on hugin, nor hugin on talon. They exchange nx values.
- A malformed plan raises `Invalid_argument`.
