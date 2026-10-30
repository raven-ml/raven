# Whole buffers

A whole-buffer call is given all of its input and writes into a
destination the program allocated. It is the simpler of lz's two ways
in, for a program that knows how large the output can be. The rules are
in the Decoding and Encoding sections of [`lz.mli`](../lib/lz.mli).

The programs on this page share two helpers, which copy between strings
and byte bigarrays:

```ocaml
let of_string s =
  Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
    (String.length s) (fun i -> Char.code s.[i])

let to_string ?len b =
  let n = Option.value len ~default:(Bigarray.Array1.dim b) in
  String.init n (fun i -> Char.chr b.{i})

let bytes n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n
```

## Sizing a destination

`decode src dst` writes the decoded bytes into `dst` from its start, and
reads earlier bytes of `dst` back when a match copies them. The decode
allocates no memory but its result, so `dst` must be large enough for all
of the output. A
program learns the decoded size from the container that holds the
stream. A Parquet page header and a ZIP entry each state it.

This program writes such a container, an 8-byte little-endian size
followed by a Zstandard frame, and reads it back. It then reads two
containers whose headers lie.

```ocaml
let pack text =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int (String.length text));
  Bytes.to_string b ^ Lz.Zstd.encode_string text

let unpack s =
  let n = Int64.to_int (String.get_int64_le s 0) in
  let src = of_string (String.sub s 8 (String.length s - 8)) in
  let dst = bytes n in
  match Lz.Zstd.decode src dst with
  | Ok { produced; _ } when produced = n -> Ok (to_string dst)
  | Ok { produced; _ } -> Error (Printf.sprintf "short: %d of %d" produced n)
  | Error e -> Error (Format.asprintf "%a" Lz.pp_error e)

let with_size n s =
  let b = Bytes.of_string s in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b
```

```ocaml
# let () =
    let page = pack (String.make 1000 'a' ^ String.make 1000 'b') in
    let show = function
      | Ok s -> Printf.printf "ok, %d bytes\n" (String.length s)
      | Error m -> print_endline m
    in
    show (unpack page);
    show (unpack (with_size 1500 page));
    show (unpack (with_size 3000 page));;
ok, 2000 bytes
output too long at byte 7
short: 2000 of 3000
```

The program is correct because the destination holds exactly the size the
header states. A stream that decodes to more is `Too_long` at the header
of the Zstandard block that asks for the extra output, byte 7 here, and
`produced` below the size means the stream held less than the header
said. A Zstandard frame from `encode`
also states its own content size, which the decode checks against what
the frame produced (the `Zstd` module in [`lz.mli`](../lib/lz.mli)).

## `bound`

`encode src dst` writes the stream of `src` into `dst` and returns
`Some len`, or `None` when the stream does not fit. `bound n` is a
destination size large enough for the stream of any `n` bytes, at every
level. Bytes that do not
compress, such as these from a generator, come out a little longer than
they went in, and `bound` leaves room for that:

```ocaml
let noise n =
  let x = ref 12345 in
  String.init n (fun _ ->
      x := (!x * 1103515245 + 12345) land 0x7fffffff;
      Char.chr (!x lsr 16 land 0xff))
```

```ocaml
# let () =
    let src = of_string (noise 1000) in
    let try_ size =
      match Lz.Zstd.encode src (bytes size) with
      | Some len -> Printf.printf "into %d bytes: %d written\n" size len
      | None -> Printf.printf "into %d bytes: does not fit\n" size
    in
    try_ 1000;
    try_ (Lz.Zstd.bound 1000);;
into 1000 bytes: does not fit
into 1021 bytes: 1014 written
```

`encode` into `bound n` bytes never returns `None`, so a program that
allocates such a destination can take the result with `Option.get`. A
program that stores a stream only when it is shorter than its input can
pass a destination of `n` bytes, and keep the input when `encode`
returns `None`.

`encode_string ?level s` is the stream of `s` as a string, for a program
that holds strings. It needs no destination.

## `consumed` and bytes after a stream

`Ok { consumed; produced }` counts the bytes the decode read and wrote.
Most formats are read to the input's end, so `consumed` is the input's
length. Each member begins where the last one ended, and a byte after
the last member where no member begins is an error. A raw deflate or
zlib stream ends at an exact byte instead.
`consumed` stops there, and the program finds whatever follows the stream
at `consumed`.

A container may hold zlib streams back to back, with no lengths between
them. This program decodes two from one input:

```ocaml
# let () =
    let first = "the first stream" and second = "and then the second" in
    let z = Lz.Zlib.encode_string in
    let src = of_string (z first ^ z second) in
    let dst = bytes 64 in
    let rec go off =
      if off < Bigarray.Array1.dim src then
        let n = Bigarray.Array1.dim src - off in
        let rest = Bigarray.Array1.sub src off n in
        match Lz.Zlib.decode rest dst with
        | Ok { consumed; produced } ->
            Printf.printf "at %d: %S\n" off (to_string ~len:produced dst);
            go (off + consumed)
        | Error e -> Format.printf "%a@." Lz.pp_error e
    in
    go 0;;
at 0: "the first stream"
at 24: "and then the second"
```

Each decode starts at the first byte the last one did not read, so each
sees one whole stream at its start. An LZ4 or Snappy block has no end
mark, and bytes after it are read as more of the block. A container of
blocks must state each block's length.

## Levels

Deflate, zlib and gzip take a level from 1 to 9, default 6, and Zstandard
from 1 to 12, default 3. LZ4 and Snappy have none. A higher level
searches longer for matches. It takes more time, and usually writes fewer
bytes:

```ocaml
# let () =
    let words = [| "lz"; "decodes"; "the"; "stream"; "into"; "a"; "page" |] in
    let x = ref 7 in
    let word _ =
      x := (!x * 1103515245 + 12345) land 0x7fffffff;
      words.(!x lsr 16 mod Array.length words)
    in
    let text = String.concat " " (List.init 20000 word) in
    List.iter
      (fun level ->
        Printf.printf "zstd level %2d: %d bytes\n" level
          (String.length (Lz.Zstd.encode_string ~level text)))
      [ 1; 3; 12 ];
    List.iter
      (fun level ->
        Printf.printf "gzip level %2d: %d bytes\n" level
          (String.length (Lz.Gzip.encode_string ~level text)))
      [ 1; 6; 9 ];;
zstd level  1: 17059 bytes
zstd level  3: 16101 bytes
zstd level 12: 13069 bytes
gzip level  1: 17748 bytes
gzip level  6: 13644 bytes
gzip level  9: 12304 bytes
```

An `encode`'s bytes depend only on the input and the level, and never on
the machine, so a program that writes the same data at the same level
writes the same file. A later version of lz may write different bytes
(the Formats section of [`lz.mli`](../lib/lz.mli)).
[The bench README](../bench/README.md#lz-against-the-vendors) times each
level against the reference implementations.

## Domains

Calls may run in parallel from any domains. A program must not run two
in parallel when one writes a buffer the other reads or writes, and must
not touch a buffer a call writes, or write one a call reads, from another
domain (the preamble of [`lz.mli`](../lib/lz.mli)). A whole-buffer call
on an input of 16 KiB or more also releases the domain lock while it
works, so that its domain's other threads run and a stop-the-world
collection need not wait for it (the Buffers and the domain lock section
of [`lz.mli`](../lib/lz.mli)). This program decodes four pages on four
domains:

```ocaml
# let () =
    let page i =
      String.concat "" (List.init 5000 (fun j -> string_of_int (i * j)))
    in
    let pages =
      List.init 4 (fun i ->
          (String.length (page i), Lz.Gzip.encode_string (page i)))
    in
    let decode (n, gz) () =
      let dst = bytes n in
      match Lz.Gzip.decode (of_string gz) dst with
      | Ok { produced; _ } -> produced
      | Error _ -> -1
    in
    let domains = List.map (fun p -> Domain.spawn (decode p)) pages in
    List.iteri
      (fun i d -> Printf.printf "page %d: %d bytes\n" i (Domain.join d))
      domains;;
page 0: 5000 bytes
page 1: 18890 bytes
page 2: 19445 bytes
page 3: 21294 bytes
```

Each domain owns its source and destination, so no buffer is shared and
the four decodes need no lock of the program's own.
