# lz Examples

Each example is a program that teaches one idea of lz. Start with
`01-roundtrip`, which encodes and decodes whole buffers with every format.
The examples after it decode and encode in steps, meet errors, and use
lz's bytesrw filters.

| Example | Shows | Key functions |
|---|---|---|
| [`01-roundtrip`](./01-roundtrip/) | Whole buffers: sizing a destination, levels, where a deflate stream ends | `bound`, `encode`, `decode`, `encode_string` |
| [`02-gunzip-file`](./02-gunzip-file/) | A gzip file of two members, decoded 4 KiB at a time | `Gzip.decoder`, `Decoder.step`, `Crc32.bigbytes` |
| [`03-stream-encode`](./03-stream-encode/) | A log written to a file through an encoder, and the same bytes whatever the steps | `Zstd.encoder`, `Encoder.step`, `Zstd.decode` |
| [`04-errors`](./04-errors/) | Damaged input, a bounded Zstandard window, dead and busy decoders | `pp_error`, `Zstd.decoder ~window`, `Invalid_argument` |
| [`05-bytesrw`](./05-bytesrw/) | Gzip through bytesrw readers and writers | `Lz_bytesrw.Gzip.compress_writes`, `decompress_reads`, `Lz_bytesrw.Error` |

The examples run in the test suite and print the same on every machine,
which `main.expected` holds. `05-bytesrw` builds and runs only where
bytesrw is installed.

Run an example from its directory, since `02-gunzip-file` and
`03-stream-encode` write a file there and remove it:

```bash
cd packages/lz/examples/02-gunzip-file
dune exec ./main.exe
```
