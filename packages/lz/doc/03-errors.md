# Errors and untrusted input

A decode often reads bytes the program did not write, such as a
downloaded file. This page shows what a failed decode returns and how a
program reads untrusted input with bounded memory. The rules are in the
Errors section of [`lz.mli`](../lib/lz.mli), in each format module's
list of refusals, and in its `Decoder` module.

```ocaml
let of_string s =
  Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
    (String.length s) (fun i -> Char.code s.[i])

let bytes n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let show = function
  | Ok { Lz.consumed; produced } ->
      Printf.printf "ok, consumed %d, produced %d\n" consumed produced
  | Error e -> Format.printf "%a@." Lz.pp_error e

let flip s i =
  let b = Bytes.of_string s in
  Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor 0x80);
  Bytes.to_string b
```

## `at` and `why`

A decode that fails returns `Error { at; why }`. `why` is one of four
reasons. `Truncated` says the input ends inside a stream, so more bytes
may complete it. `Malformed` says the bytes break the format, and no
bytes appended would make them valid. `Too_long` says the output would
pass the destination's end. `Unsupported` says the stream is well formed
and uses a feature lz declines, such as a preset dictionary. `at` is the
offset of the first byte of what is wrong. A decode reads its input in
order and returns the first error it meets.

This program damages a zlib stream that holds 1,000 bytes in four ways:

```ocaml
# let () =
    let text = String.concat "" (List.init 200 (Printf.sprintf "%04d\n")) in
    let z = Lz.Zlib.encode_string text in
    let n = String.length z in
    let decode s = show (Lz.Zlib.decode (of_string s) (bytes 1000)) in
    decode z;
    decode (flip z 0);
    decode (flip z (n - 2));
    decode (String.sub z 0 (n - 1));
    show (Lz.Zlib.decode (of_string z) (bytes 999));;
ok, consumed 397, produced 1000
malformed at byte 0: incorrect header check
malformed at byte 393: incorrect data check
truncated at byte 396
output too long at byte 391
```

A flipped bit in the header fails its check bits, at the header's first
byte. One in the Adler-32, the stream's last 4 bytes, is a checksum that
does not match, at the checksum's first byte. A cut stream is `Truncated`
at the input's length. A destination one byte short is `Too_long` at the
code that asks for the byte. The `Zlib` and `Deflate` modules of
[`lz.mli`](../lib/lz.mli) list every refusal and the byte it names.

A step's `at` counts from the first byte its decoder took after it was
made or last returned `Done`, so it names the same byte as a whole-buffer
decode of the same input, however the input was cut into steps:

```ocaml
# let () =
    let z = flip (Lz.Zlib.encode_string (String.make 1000 'x')) 8 in
    let d = Lz.Zlib.decoder () in
    let src = of_string z in
    let step first len ~last =
      Lz.Decoder.step d ~last (Bigarray.Array1.sub src first len) ignore
    in
    (match step 0 5 ~last:false with
     | Ok (More k) -> Printf.printf "took %d\n" k
     | _ -> ());
    match step 5 (String.length z - 5) ~last:true with
    | Error e -> Format.printf "step: %a@." Lz.pp_error e
    | Ok _ -> ();;
took 5
step: malformed at byte 8: invalid literal/length code
```

The error lies in the second piece, and its offset counts from the
input's first byte. The decoder that returned it is dead, and a further
step raises `Invalid_argument`. A program that wants to decode another
stream makes a new decoder.

## Messages

`Malformed` and `Unsupported` carry a message from a fixed set for each
format, such as zlib's messages for deflate, zlib and gzip, and for
Zstandard phrases that name the RFC 8878 field at fault (the type `why` in
[`lz.mli`](../lib/lz.mli)). A message is a constant string and holds no
byte of the input, so a program can log the error of a hostile input as
it is. A program decides what to do from the constructor and `at`, since
the messages are for printing and may change.

## Sizes an input declares

Formats carry sizes, such as gzip's `ISIZE`, Zstandard's
`Frame_Content_Size`, an LZ4 frame's content size and a Snappy block's
preamble. A hostile input can declare any of them. lz checks each against
the decoded bytes, and allocates nothing by one, except a Zstandard
frame's window size and content size, which size a decoder's buffer (the
last section). A whole-buffer decode allocates no memory but its result,
so the program bounds its memory by the destination it chooses.

This Snappy block declares 2<sup>32</sup> - 1 decoded bytes in its
preamble and holds none:

```ocaml
# let () =
    let block = "\xff\xff\xff\xff\x0f" in
    show (Lz.Snappy.decode (of_string block) (bytes 1024));;
output too long at byte 0
```

The decode compares the preamble with the destination's 1,024 bytes
before it writes any output, and refuses at the preamble's first byte. A
program that takes a destination size from an untrusted header caps it
with a limit of its own first.

## Bounding the output of a decoder

A decoder bounds its memory, and nothing bounds the bytes it decodes. A
megabyte of one repeated byte compresses to about a kilobyte, so a small
input can decode to far more than a program wants to hold. A program
that steps a decoder over a stream it does not trust counts the bytes `f`
receives and stops past a limit of its own (the `Decoder` module of
[`lz.mli`](../lib/lz.mli)). `gunzip_capped` raises from `f` once the
output would pass `limit`, which leaves the decoder dead, and drops it:

```ocaml
exception Over_limit

let gunzip_capped ~limit s =
  let d = Lz.Gzip.decoder () in
  let out = Buffer.create 4096 in
  let keep run =
    let n = Bigarray.Array1.dim run in
    if Buffer.length out + n > limit then raise Over_limit;
    for i = 0 to n - 1 do
      Buffer.add_char out (Char.chr run.{i})
    done
  in
  let rec go src =
    let rest k = Bigarray.Array1.sub src k (Bigarray.Array1.dim src - k) in
    match Lz.Decoder.step d ~last:true src keep with
    | Ok (Lz.Done _) -> Ok (Buffer.contents out)
    | Ok (More k) -> go (rest k)
    | Error e -> Error (Format.asprintf "%a" Lz.pp_error e)
  in
  match go (of_string s) with
  | r -> r
  | exception Over_limit -> Error "over the limit"
```

```ocaml
# let () =
    let gz = Lz.Gzip.encode_string (String.make 1_000_000 'a') in
    let try_ limit =
      Printf.printf "limit %d: " limit;
      match gunzip_capped ~limit gz with
      | Ok s -> Printf.printf "ok, %d bytes\n" (String.length s)
      | Error m -> print_endline m
    in
    try_ 65536;
    try_ 2_000_000;;
limit 65536: over the limit
limit 2000000: ok, 1000000 bytes
```

`gunzip_capped` is correct because every byte the decoder hands over
passes through `keep`, which counts it before it keeps it.

## A Zstandard decoder's window

A Zstandard frame declares its window, the decoded bytes that its matches
may reach back into. A whole-buffer decode uses the destination as the
window and accepts any declared size. A decoder keeps the window in
memory of its own, and allocates it when the frame's header arrives,
before any output. `Lz.Zstd.decoder ~window` bounds it, by default at
128 MiB, and refuses a frame that declares more as `Unsupported`.

This header declares a 2 GiB window, and holds no block:

```ocaml
# let () =
    let header = "\x28\xb5\x2f\xfd\x00\xa8" in
    let step d =
      match Lz.Decoder.step d ~last:true (of_string header) ignore with
      | Ok _ -> print_endline "ok"
      | Error e -> Format.printf "%a@." Lz.pp_error e
    in
    step (Lz.Zstd.decoder ());
    step (Lz.Zstd.decoder ~window:(1 lsl 20) ());
    show (Lz.Zstd.decode (of_string header) (bytes 1024));;
unsupported at byte 5: Window_Size above the decoder's limit
unsupported at byte 5: Window_Size above the decoder's limit
truncated at byte 6
```

Both decoders refuse the header at its Window_Descriptor, byte 5, and
allocate nothing for it. The whole-buffer decode reads past the header,
since its window is the destination, and finds the input cut short. A
program that steps through Zstandard streams it does not trust sets
`~window` to the most memory it will give one decoder. The frames lz
writes, from `Lz.Zstd.encode` or an encoder, declare at most 4 MiB.
