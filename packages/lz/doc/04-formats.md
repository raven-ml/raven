# Choosing a format

Most programs do not choose. The data they read, or the program that reads
what they write, fixes the format, and lz decodes the streams those
programs write. This page says what tells the formats apart for a program
that does choose. Each format's rules are in its module in
[`lz.mli`](../lib/lz.mli).

| Format | Module | Steps | Levels | Checked by the format | Ends |
|---|---|---|---|---|---|
| Raw deflate (RFC 1951) | `Lz.Deflate` | yes | 1 to 9, default 6 | nothing | at an exact byte |
| Zlib (RFC 1950) | `Lz.Zlib` | yes | 1 to 9, default 6 | Adler-32 | at an exact byte |
| Gzip (RFC 1952) | `Lz.Gzip` | yes | 1 to 9, default 6 | CRC-32 and length of each member | at the input's end |
| Zstandard (RFC 8878) | `Lz.Zstd` | yes | 1 to 12, default 3 | XXH64 and content size, which lz writes | at the input's end |
| LZ4 frames | `Lz.Lz4.Frame` | no | none | content checksum and size, which lz writes | at the input's end |
| LZ4 blocks | `Lz.Lz4.Block` | no | none | nothing | at the input's end |
| Snappy blocks | `Lz.Snappy` | no | none | the content's length | at the input's end |

## Where each is found

Raw deflate is the stream inside a ZIP entry, an NPZ file's included.
Zlib wraps it in PNG images and PDF streams. Gzip is the `.gz` file, and
the `Content-Encoding: gzip` of HTTP. Zstandard is the `.zst` file, and a
codec of Parquet and Arrow. LZ4 frames are the `.lz4` file. LZ4 blocks
and Snappy blocks carry no checksum and no end mark, and live inside
containers that state each block's length, such as Parquet pages.

## For data of your own

On the bench's corpus, Zstandard's default level writes fewer bytes than
deflate's, and its higher levels fewer still. Every frame lz writes
carries a checksum of its content. A decoder needs a bound on its window
for untrusted input ([Errors and untrusted
input](03-errors.md#a-zstandard-decoders-window)).

Gzip and zlib are read by every system, at some cost in size. Gzip
checks each member's CRC-32, and a gzip file may be the concatenation of
several, which a decoder reads as one stream.

LZ4 and Snappy take no level, and write more bytes than the others. lz
has whole-buffer calls for them and no decoders or encoders, so a program
holds the whole block or frame. [The bench
README](../bench/README.md#lz-against-the-vendors) times each format and
level, and gives the bytes each encoder wrote.

## Checksums

A format's checksum is verified before a decode returns `Ok`, and before
a step returns `Done`. A program that keeps its own checksum, as PNG's
chunks and ZIP's entries do, computes CRC-32 with `Lz.Crc32`. This
program checks the CRC-32 of a PNG `IEND` chunk, which the PNG
specification gives as `ae426082`, and shows how a running value
continues across pieces:

```ocaml
# let () =
    Printf.printf "IEND: %08x\n" (Lz.Crc32.string "IEND");
    let a = "first half, " and b = "second half" in
    let whole = Lz.Crc32.string (a ^ b) in
    let pieces = Lz.Crc32.string ~crc:(Lz.Crc32.string a) b in
    Printf.printf "whole %08x, in pieces %08x\n" whole pieces;;
IEND: ae426082
whole 3b98f369, in pieces 3b98f369
```

`Lz.Crc32.bigbytes` computes the same over a bigarray, and releases the
runtime lock on a long one (the preamble of [`lz.mli`](../lib/lz.mli)).
