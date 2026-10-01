# compress

Compression codecs with no system library, one library per codec:

| Library | Module | Formats |
|---|---|---|
| `compress.deflate` | `Compress_deflate` | deflate, zlib and gzip streams, read and written; CRC-32 |
| `compress.snappy` | `Compress_snappy` | Snappy blocks, read and written |
| `compress.lz4` | `Compress_lz4` | LZ4 blocks and frames, read |
| `compress.zstd` | `Compress_zstd` | Zstandard frames, read |

Data held in memory with a known decompressed length, such as a Parquet page
of a mapped file, decompresses between byte arrays with `decompress`. Deflate,
zlib and gzip streams are also [bytesrw](https://erratique.ch/software/bytesrw)
filters: `decompress_reads` reads a stream and `compress_writes` writes one.

```ocaml
open Bytesrw

(* A string, compressed and back. *)
let z = Bytes.Writer.filter_string [ Compress_deflate.Zlib.compress_writes () ] s
let s' = Bytes.Reader.filter_string [ Compress_deflate.Zlib.decompress_reads () ] z

(* A Snappy page of [file], into [scratch], which holds [size] bytes. *)
let page = Bigarray.Array1.sub scratch 0 size
let result = Compress_snappy.decompress (Bigarray.Array1.sub file first length) page
```

## Normative sources

- [RFC 1950: zlib](https://www.rfc-editor.org/rfc/rfc1950),
  [RFC 1951: deflate](https://www.rfc-editor.org/rfc/rfc1951) and
  [RFC 1952: gzip](https://www.rfc-editor.org/rfc/rfc1952)
- [The Snappy format](https://github.com/google/snappy/blob/main/format_description.txt)
- [The LZ4 block format](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md)
  and [frame format](https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md)
- [RFC 8878: Zstandard](https://www.rfc-editor.org/rfc/rfc8878)
- [XXH32 and XXH64](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md)

The code is written from these specifications and incorporates no source from
the reference libraries.

## C invariants

- The OCaml side checks every span before entering C. Exported C symbols start
  with `compress_` or `caml_compress_`.
- Entry points over bigarrays release the runtime lock for large spans, and
  touch only bigarray and C memory while it is released. Entry points over
  `bytes`, which the filters use, never release it.
- Decoders check every element against both ends of their input and output
  before copying it. Wide copies write at most 16 bytes past what they copy,
  inside the room they were given and before data not yet produced.
- No process-global mutable state: the static tables are filled once, by the
  OCaml module initializer.

## Tests and benchmarks

Each library has a suite under `test/`, run with
`dune build @otherlibs/compress/runtest`. The reference vectors in
`test/fixtures` come from Python's zlib, python-snappy, lz4 and zstandard
through `test/fixtures/generate.py`, which also checks that the references
decode the streams this library pins as its encoders' output. CI runs the
suites with the C built under AddressSanitizer and UndefinedBehaviorSanitizer
(the `sanitize` profile).

`bench/` compares the codecs with the references; see `bench/README.md`.
