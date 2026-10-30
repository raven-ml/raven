# lz

lz compresses and decompresses bytes from OCaml: deflate with its zlib and
gzip wrappers, Zstandard, LZ4 and Snappy, and CRC-32. Its formats are
written in C inside the library, so a program needs no system zlib or
libzstd. Its calls read and write byte bigarrays, whose bytes the
collector never moves. A call that has worked through a few KiB
therefore releases the domain lock, so that the domain's other threads
run during the rest of it (the Buffers and the domain lock section of
[`lz.mli`](../lib/lz.mli)).

A program uses lz in one of two ways. A *whole-buffer* call is given all
of its input and writes into a *destination* the program allocated.
A *decoder* or an *encoder* takes its input in pieces, one *step* at a
time, and lends the output of each step, its *run*, to a function the
program passes. The preamble of [`lz.mli`](../lib/lz.mli) defines these
terms.

## Install

lz needs OCaml 5.5 or later. Install the `lz` package and add `lz` to your
executable's `libraries`:

<!-- $MDX skip -->
```sh
$ opam install lz
```

```dune
(executable
 (name main)
 (libraries lz))
```

The library `lz.bytesrw` holds filters for bytesrw's readers and writers.
It is built where bytesrw is installed, and a program adds it beside `lz`
([Decoding and encoding in steps](02-steps.md#through-bytesrw)).

## A first program

This program compresses a short text with gzip and decompresses it again:

```ocaml
# let () =
    let text = String.concat "" (List.init 100 (Printf.sprintf "line %d\n")) in
    let gz = Lz.Gzip.encode_string ~level:9 text in
    let src =
      Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
        (String.length gz) (fun i -> Char.code gz.[i])
    in
    let dst =
      Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
        (String.length text)
    in
    match Lz.Gzip.decode src dst with
    | Ok { consumed; produced } ->
        Printf.printf "%d bytes, %d of gzip; read %d, wrote %d\n"
          (String.length text) (String.length gz) consumed produced
    | Error e -> Format.printf "%a@." Lz.pp_error e;;
790 bytes, 173 of gzip; read 173, wrote 790
```

Each call, in order:

- `Lz.Gzip.encode_string ~level:9 text` is the gzip stream of `text`, as a
  string: one member, whose header holds no file name and a modification
  time of 0. `~level` runs from 1 to 9, and defaults to 6. A higher level
  searches longer for matches (the `Deflate` module in
  [`lz.mli`](../lib/lz.mli)).
- `Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout n f` makes
  a byte bigarray of `n` bytes, the type `Lz.bigbytes`, whose byte `i` is
  `f i`. Here it holds a copy of the gzip stream, since every decode reads
  a bigarray. A program that reads a file can read it straight into a
  bigarray with `In_channel.input_bigarray`.
- `Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
  (String.length text)` is the destination, with room for the decoded
  bytes and no more. The program knows their number here. A file format
  that holds compressed pages usually states each page's decoded size in
  a header, and a program that does not know it decodes in steps instead.
- `Lz.Gzip.decode src dst` decodes every member of `src` into `dst` from
  its start. A gzip input is read to its end, so `consumed` is the
  length of `src` (the type `progress` in [`lz.mli`](../lib/lz.mli)).
- `Ok { consumed; produced }` holds the number of bytes read from `src`
  and written to `dst`. The bytes of `dst` past `produced` are
  unspecified.
- `Error e` holds the offset of the input byte where the problem begins,
  `e.at`, and the reason, `e.why`. `Lz.pp_error` prints both, as in
  `truncated at byte 173` (the Errors section of
  [`lz.mli`](../lib/lz.mli)).

Every other format's module has the same `decode`, `bound`, `encode` and
`encode_string` (the Formats section of [`lz.mli`](../lib/lz.mli)). LZ4's
are in `Lz.Lz4.Block` and `Lz.Lz4.Frame`.

## Reading order

The interfaces state every rule, and read top to bottom:

1. [Lz](../lib/lz.mli): buffers and errors, whole-buffer calls, decoders
   and encoders, each format's refusals, and CRC-32.
2. [Lz_bytesrw](../lib/bytesrw/lz_bytesrw.mli): bytesrw filters over lz's
   decoders and encoders.

These pages show the rules at work, with programs:

1. [Whole buffers](01-whole-buffers.md): sizing a destination, `bound`,
   `consumed` and `produced`, bytes after a stream, and levels.
2. [Decoding and encoding in steps](02-steps.md): steps and `~last`, runs
   and how long they live, a decoder fed from a channel, an encoder
   writing to one, and `lz.bytesrw`.
3. [Errors and untrusted input](03-errors.md): what `at` and `why` mean,
   sizes an input declares, and a Zstandard decoder's `~window`.
4. [Choosing a format](04-formats.md): what each format is for.

The [examples](../examples/README.md) teach one idea each, and run in the
test suite.
