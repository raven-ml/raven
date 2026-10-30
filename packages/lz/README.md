# lz

lz compresses and decompresses bytes from OCaml: deflate with its zlib and
gzip wrappers, Zstandard, LZ4 and Snappy, and CRC-32. Its codecs are
written in C inside the library and depend on nothing outside the OCaml
distribution, so a program needs no system zlib or libzstd. They work on
byte bigarrays, decode a whole buffer into a destination the program
sizes, or decode and encode in steps that lend their output from the
codec's own memory. A failed decode returns an error with the offset of
the byte where the problem begins.

## A first program

This program compresses a thousand rows with Zstandard and decompresses
them into a destination of the rows' length:

```ocaml
# let () =
    let text = String.concat "" (List.init 1000 (Printf.sprintf "row %d\n")) in
    let zst = Lz.Zstd.encode_string text in
    let src =
      Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
        (String.length zst) (fun i -> Char.code zst.[i])
    in
    let dst =
      Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
        (String.length text)
    in
    match Lz.Zstd.decode src dst with
    | Ok { produced; _ } ->
        Printf.printf "%d bytes, %d of zstd, %d decoded\n"
          (String.length text) (String.length zst) produced
    | Error e -> Format.printf "%a@." Lz.pp_error e;;
7890 bytes, 736 of zstd, 7890 decoded
```

The destination holds exactly the decoded size, which a real program
learns from the container that holds the stream. A stream that decodes to
more is an error, `Too_long`, and the decode allocates nothing of its own
(the Decoding section of [`lz.mli`](lib/lz.mli)).

## Formats

| Format | Module | For |
|---|---|---|
| Raw deflate | `Lz.Deflate` | ZIP entries, and containers that frame deflate themselves |
| Zlib | `Lz.Zlib` | PNG images, PDF streams |
| Gzip | `Lz.Gzip` | `.gz` files and HTTP bodies |
| Zstandard | `Lz.Zstd` | `.zst` files, Parquet pages, and data of your own |
| LZ4 frames and blocks | `Lz.Lz4.Frame`, `Lz.Lz4.Block` | `.lz4` files, and blocks inside a container |
| Snappy blocks | `Lz.Snappy` | blocks inside a container, such as Parquet pages |
| CRC-32 | `Lz.Crc32` | the checksum of gzip, PNG and ZIP |

Deflate, zlib, gzip and Zstandard also have decoders and encoders that
take their input in steps. `lz.bytesrw`, built where bytesrw is
installed, wraps them as bytesrw filters.

lz declines preset dictionaries, Zstandard and LZ4 dictionaries, LZ4's
legacy frame and Snappy's framing format. It has no xz, bzip2 or Brotli.

## Where next

- [The guide](doc/index.md) installs lz and explains each call of a first
  program. Its pages work through whole buffers, streams, errors and
  untrusted input, and the choice of a format.
- The interfaces state every rule: [`lz.mli`](lib/lz.mli), then
  [`lz_bytesrw.mli`](lib/bytesrw/lz_bytesrw.mli).
- [The examples](examples/README.md) teach one idea each.
- [The benches](bench/README.md) time lz on three corpora of their own, and
  [lz against the vendors](bench/README.md#lz-against-the-vendors) compares
  each codec with libdeflate, zstd, lz4 and snappy.
