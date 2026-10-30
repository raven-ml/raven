# Decoding and encoding in steps

A decoder or an encoder takes its input in pieces, for input that arrives
over time or whose size the program does not know. Each call is a step.
This page follows steps through a decoder and an encoder, then feeds them
from channels and through bytesrw. The rules are in the preamble and the
Decoders and encoders section of [`lz.mli`](../lib/lz.mli), and in its
`Decoder` and `Encoder` modules.

The programs on this page share these helpers:

```ocaml
let of_string s =
  Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
    (String.length s) (fun i -> Char.code s.[i])

let dim = Bigarray.Array1.dim
let to_string b = String.init (dim b) (fun i -> Char.chr b.{i})
let sub b first = Bigarray.Array1.sub b first (dim b - first)

let bytes n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let show = function
  | Ok (Lz.More k) -> Printf.sprintf "More %d" k
  | Ok (Done k) -> Printf.sprintf "Done %d" k
  | Error e -> Format.asprintf "Error: %a" Lz.pp_error e
```

## A step

`Lz.Decoder.step d ~last src f` decodes from the start of `src`, applies
`f` once to the bytes it produced, its run, and returns how many bytes of
`src` it took. This program gives a gzip decoder one member in three
pieces:

```ocaml
# let () =
    let gz = Lz.Gzip.encode_string "a short note, decoded in steps\n" in
    let n = String.length gz in
    let pieces =
      [ String.sub gz 0 5; String.sub gz 5 20; String.sub gz 25 (n - 25) ]
    in
    let d = Lz.Gzip.decoder () in
    List.iteri
      (fun i piece ->
        let last = i = 2 in
        let r =
          Lz.Decoder.step d ~last (of_string piece) (fun run ->
              Printf.printf "step %d, %2d bytes, run %S\n" i
                (String.length piece) (to_string run))
        in
        print_endline ("  " ^ show r))
      pieces;;
step 0,  5 bytes, run ""
  More 5
step 1, 20 bytes, run "a short note, "
  More 20
step 2, 26 bytes, run "decoded in steps\n"
  Done 26
```

Each step took every byte it was given, since no run filled. A decoder
step that returns `More k` with `k` short of its input's length has
filled its run, 256 KiB, and the next step begins at byte `k`. A program
that steps from the first byte the last step did not take never loses or
repeats a byte, whatever the runs' sizes.

## The end of the input

`~last:true` says that the input ends with the bytes this step is given.
A gzip or Zstandard decoder needs it. Members follow one another to the
input's end, so without it the decoder cannot tell the last member from
one that more input would follow. A raw deflate or zlib
stream ends at an exact byte, and its decoder returns `Done` there with or
without `~last`. A program that does not know where its input ends steps
with `~last:false`, and with `~last:true` on an empty input once it does:

```ocaml
# let () =
    let gz = of_string (Lz.Gzip.encode_string "two calls") in
    let d = Lz.Gzip.decoder () in
    print_endline (show (Lz.Decoder.step d ~last:false gz ignore));
    print_endline (show (Lz.Decoder.step d ~last:true (of_string "") ignore));
    let cut = of_string (String.sub (Lz.Gzip.encode_string "cut") 0 12) in
    let d = Lz.Gzip.decoder () in
    print_endline (show (Lz.Decoder.step d ~last:false cut ignore));
    print_endline (show (Lz.Decoder.step d ~last:true (of_string "") ignore));;
More 29
Done 0
More 12
Error: truncated at byte 12
```

The first decoder took all 29 bytes and held a member that ended on the
last one. The empty step with `~last:true` ended the stream. The second
input stops inside its member. A step returns `Truncated` only with
`~last:true`, so the cut shows on the empty step, at the input's length,
12, counted from the first byte the decoder took. A step with
`~last:true` may also return `More k` with `k` the whole input, when
output is still due. The program then steps again with an empty input
until `Done`.

## A run's lifetime

The run is a view of the decoder's own buffer. It is valid while `f`
runs, and the next step writes over it, so `f` must copy the bytes the
program keeps, or write them out, before it returns. The run is also part
of the decoder's window, which later matches copy from, so `f` must not
write it:

```ocaml
let decode_string d s =
  let out = Buffer.create 4096 in
  let keep run = Buffer.add_string out (to_string run) in
  let rec go src =
    match Lz.Decoder.step d ~last:true src keep with
    | Ok (Done _) -> Ok (Buffer.contents out)
    | Ok (More k) -> go (sub src k)
    | Error e -> Error e
  in
  go (of_string s)
```

`decode_string` is correct because `keep` copies each run into `out`
before the step returns and leaves the run as it was, and each step
begins at the first byte the last one did not take. With `~last:true` on
every step, `go` ends at `Done` or at an error.

## A decoder fed from a channel

A program that reads a file decodes it with one buffer of its own, however
long the file. `decode_channel` reads 64 KiB at a time with
`In_channel.input_bigarray`, which returns 0 only at the end of the
channel:

```ocaml
let decode_channel d ic f =
  let buf = bytes 65536 in
  (* [buf]'s [len] bytes from [start] are read and not yet taken. *)
  let rec go start len last =
    match Lz.Decoder.step d ~last (Bigarray.Array1.sub buf start len) f with
    | Error e -> Error e
    | Ok (Done k) -> Ok (Bigarray.Array1.sub buf (start + k) (len - k))
    | Ok (More k) when k < len || last -> go (start + k) (len - k) last
    | Ok (More _) ->
        let n = In_channel.input_bigarray ic buf 0 (dim buf) in
        go 0 n (n = 0)
  in
  go 0 0 false
```

`decode_channel` is correct for every format. When a step takes less than
it was given, or the input has ended, it steps again on the rest. When a
step has taken every byte, it reads more. At the channel's end it steps
with `~last:true` until `Done`. A gzip or Zstandard decoder reads to the
channel's end. A deflate or zlib decoder returns `Done` at its stream's
end, and `decode_channel` then returns the bytes it read past the stream,
which come before the rest of the channel.

## An encoder writing to a channel

Each encoder step takes all of its input up to a quota, 64 KiB for
deflate, zlib and gzip and 128 KiB for Zstandard, and its run always has
room for the output the step owes. The encoder may keep input until more
arrives, so a run may be empty. A step with `~last:true` that takes all
of its input writes the input kept and the stream's end, and returns
`Done`.
`encode_channel` reads from one channel and writes the stream to
another:

```ocaml
let encode_channel e ic oc =
  let buf = bytes 65536 in
  let write run = Out_channel.output_bigarray oc run 0 (dim run) in
  let rec go start len last =
    match Lz.Encoder.step e ~last (Bigarray.Array1.sub buf start len) write with
    | Lz.Done _ -> ()
    | More k when k < len -> go (start + k) (len - k) last
    | More _ ->
        let n = In_channel.input_bigarray ic buf 0 (dim buf) in
        go 0 n (n = 0)
  in
  go 0 0 false
```

This program compresses a file with `encode_channel` and decompresses it
with `decode_channel`:

```ocaml
# let () =
    let text =
      String.concat "" (List.init 50_000 (Printf.sprintf "row %d\n"))
    in
    Out_channel.with_open_bin "rows.txt" (fun oc ->
        Out_channel.output_string oc text);
    In_channel.with_open_bin "rows.txt" (fun ic ->
        Out_channel.with_open_bin "rows.txt.zst" (fun oc ->
            encode_channel (Lz.Zstd.encoder ()) ic oc));
    let out = Buffer.create (String.length text) in
    let keep run = Buffer.add_string out (to_string run) in
    let r =
      In_channel.with_open_bin "rows.txt.zst" (fun ic ->
          decode_channel (Lz.Zstd.decoder ()) ic keep)
    in
    (match r with
     | Ok _ -> ()
     | Error e -> Format.printf "%a@." Lz.pp_error e);
    Printf.printf "%d bytes, same: %b\n" (String.length text)
      (Buffer.contents out = text);
    Sys.remove "rows.txt";
    Sys.remove "rows.txt.zst";;
488890 bytes, same: true
```

An encoder's bytes depend only on the input and the level. They never
depend on how the input was cut into steps, so a program that reads
4 KiB at a time writes the same file as one that reads 64 KiB.
A Zstandard frame written by steps differs from `Lz.Zstd.encode`'s only
in its header, which lacks the content size that `encode` knows.

## Many streams on one decoder

After `Done`, the next step begins a new stream of the format. A deflate
or zlib decoder returns `Done k` at its stream's exact end, and the bytes
from `k` on are the next stream's. This program decodes three zlib streams
that follow one another in one input:

```ocaml
# let () =
    let z = Lz.Zlib.encode_string in
    let src = of_string (z "one" ^ z "two" ^ z "three") in
    let d = Lz.Zlib.decoder () in
    let rec go off =
      if off < Bigarray.Array1.dim src then
        match Lz.Decoder.step d ~last:true (sub src off) (fun run ->
            Printf.printf "at %2d: %S\n" off (to_string run)) with
        | Ok (Done k) -> go (off + k)
        | r -> print_endline (show r)
    in
    go 0;;
at  0: "one"
at 11: "two"
at 22: "three"
```

A gzip or Zstandard decoder instead decodes every member up to the input's
end as one stream, and returns `Done` once, on the step with `~last:true`.

## Through bytesrw

`lz.bytesrw` wraps decoders and encoders as filters of bytesrw's readers
and writers ([`lz_bytesrw.mli`](../lib/bytesrw/lz_bytesrw.mli)). A program
that already reads and writes through bytesrw gets compression from one
more function, and the filter does the staging and copying of the
programs above:

```ocaml
# let () =
    let text = String.concat "" (List.init 1000 (Printf.sprintf "line %d\n")) in
    let compress = Lz_bytesrw.Zstd.compress_reads () in
    let zst = Bytesrw.Bytes.Reader.filter_string [ compress ] text in
    let decompress = Lz_bytesrw.Zstd.decompress_reads () in
    let r = decompress (Bytesrw.Bytes.Reader.of_string zst) in
    let back = Bytesrw.Bytes.Reader.to_string r in
    Printf.printf "%d bytes, %d of zstd, same: %b\n" (String.length text)
      (String.length zst) (back = text);;
8890 bytes, 735 of zstd, same: true
```

A decompressing filter raises `Bytesrw.Bytes.Stream.Error` with
`Lz_bytesrw.Error e` where a decoder would return `Error e`, at the
error's position in the compressed stream (the Errors section of
[`lz_bytesrw.mli`](../lib/bytesrw/lz_bytesrw.mli)).
