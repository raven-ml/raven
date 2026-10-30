(* lz's codecs on three 8 MiB corpora, built as the tests build them
   (Lz_support.corpus): the decoders on lz's own encodings, the encoders
   and checksums on the raw bytes. Every decode row first checks its output
   against the raw corpus, so a row never times a wrong answer. *)

open Bigarray

let size = 8 lsl 20

let corpora =
  [ ("text", Lz_support.Text); ("columns", Columns); ("random", Random) ]

let create n = Array1.create int8_unsigned c_layout n

(* [encoded bound encode raw] is [raw] encoded by [encode], into a
   destination of [bound]'s size cut to the output. *)
let encoded bound encode raw =
  let dst = create (bound (Array1.dim raw)) in
  match encode raw dst with
  | Some n -> Array1.sub dst 0 n
  | None -> failwith "the output exceeds the bound"

let zstd3 = encoded Lz.Zstd.bound (Lz.Zstd.encode ~level:3)

(* Decoders: a suffix naming the format and level, lz's encoder of the
   input and the decode that reads it. *)
let decoders =
  [
    ("zlib6", encoded Lz.Zlib.bound (Lz.Zlib.encode ~level:6), Lz.Zlib.decode);
    ("gz9", encoded Lz.Gzip.bound (Lz.Gzip.encode ~level:9), Lz.Gzip.decode);
    ( "deflate6",
      encoded Lz.Deflate.bound (Lz.Deflate.encode ~level:6),
      Lz.Deflate.decode );
    ("zst1", encoded Lz.Zstd.bound (Lz.Zstd.encode ~level:1), Lz.Zstd.decode);
    ("zst3", zstd3, Lz.Zstd.decode);
    ("zst12", encoded Lz.Zstd.bound (Lz.Zstd.encode ~level:12), Lz.Zstd.decode);
    ("lz4b", encoded Lz.Lz4.Block.bound Lz.Lz4.Block.encode, Lz.Lz4.Block.decode);
    ("lz4f", encoded Lz.Lz4.Frame.bound Lz.Lz4.Frame.encode, Lz.Lz4.Frame.decode);
    ("snappy", encoded Lz.Snappy.bound Lz.Snappy.encode, Lz.Snappy.decode);
  ]

let check name raw dst = function
  | Ok { Lz.consumed = _; produced } when produced = Array1.dim raw ->
      if Array1.sub dst 0 produced <> raw then
        failwith (name ^ ": the output differs from the raw corpus")
  | Ok { produced; _ } ->
      failwith (Printf.sprintf "%s: %d bytes, not %d" name produced (Array1.dim raw))
  | Error { Lz.at; _ } -> failwith (Printf.sprintf "%s: an error at %d" name at)

(* [src] is the corpus encoded, built once for every row that reads it. *)
let decode corpus raw src (suffix, _, decode) =
  let dst = lazy (create (Array1.dim (Lazy.force raw))) in
  Thumper.bench_with_setup ("decode " ^ suffix)
    ~setup:(fun () ->
      let src = Lazy.force src and dst = Lazy.force dst in
      check (corpus ^ "." ^ suffix) (Lazy.force raw) dst (decode src dst);
      (src, dst))
    (fun (src, dst) -> decode src dst)

(* A frame of the corpus's first 64 KiB decoded into a destination of
   exactly that size, as a container that states the size passes it: the
   output's end meets the block's literals there. *)
let small corpus raw =
  let n = 64 lsl 10 in
  let raw = lazy (Array1.sub (Lazy.force raw) 0 n) in
  let src = lazy (zstd3 (Lazy.force raw)) in
  let dst = lazy (create n) in
  Thumper.bench_with_setup "decode zst3 of 64 KiB, exact destination"
    ~setup:(fun () ->
      let src = Lazy.force src and dst = Lazy.force dst in
      check (corpus ^ " 64 KiB zst3") (Lazy.force raw) dst (Lz.Zstd.decode src dst);
      (src, dst))
    (fun (src, dst) -> Lz.Zstd.decode src dst)

(* Stepped decode: a new decoder per stream, the input in slices of
   [slice] bytes, each run handed to a function that reads none of it, as a
   reader that writes its output elsewhere. *)
let stepped decoder src slice =
  let d = decoder () in
  let n = Array1.dim src in
  let rec go at =
    let len = min slice (n - at) in
    let last = at + len = n in
    match Lz.Decoder.step d ~last (Array1.sub src at len) ignore with
    | Error _ -> failwith "stepped decode failed"
    | Ok (Lz.Done _) -> ()
    | Ok (More k) -> go (at + k)
  in
  go 0

let steps srcs =
  List.concat_map
    (fun (suffix, decoder) ->
      List.map
        (fun (label, slice) ->
          let src = List.assoc suffix srcs in
          Thumper.bench_with_setup
            (Printf.sprintf "decode %s in %s steps" suffix label)
            ~setup:(fun () -> Lazy.force src)
            (fun src -> stepped decoder src slice))
        [ ("64 KiB", 64 lsl 10); ("1 MiB", 1 lsl 20) ])
    [ ("zlib6", Lz.Zlib.decoder); ("zst3", fun () -> Lz.Zstd.decoder ()) ]

(* A Zstandard frame of [raw] as a stream flushed every 100 bytes writes
   it: a 1 MiB window (RFC 8878, 3.1.1.1.2), then compressed blocks of 100
   raw literals (a 2-byte literals header) and no sequence. Stepped, a run
   holds hundreds of its blocks. *)
let flushed raw =
  let n = Array1.dim raw in
  let b = Buffer.create (n + (n / 25) + 16) in
  Buffer.add_string b "\x28\xb5\x2f\xfd\x00\x50";
  let rec blocks i =
    let k = min 100 (n - i) in
    let size = k + 3 and last = if i + k = n then 1 else 0 in
    let h = last lor (2 lsl 1) lor (size lsl 3) in
    Buffer.add_char b (Char.chr (h land 0xff));
    Buffer.add_char b (Char.chr ((h lsr 8) land 0xff));
    Buffer.add_char b (Char.chr (h lsr 16));
    Buffer.add_char b (Char.chr (4 lor ((k land 0xf) lsl 4)));
    Buffer.add_char b (Char.chr (k lsr 4));
    for j = i to i + k - 1 do
      Buffer.add_char b (Char.chr (Array1.unsafe_get raw j))
    done;
    Buffer.add_char b '\x00';
    if i + k < n then blocks (i + k)
  in
  blocks 0;
  let s = Buffer.contents b in
  let src = create (String.length s) in
  String.iteri (fun i c -> Array1.unsafe_set src i (Char.code c)) s;
  src

let flushed_steps corpus raw =
  Thumper.bench_with_setup "decode zst of 100-byte blocks in 1 MiB steps"
    ~setup:(fun () ->
      let raw = Lazy.force raw in
      let src = flushed raw and dst = create (Array1.dim raw) in
      check (corpus ^ " in 100-byte blocks") raw dst (Lz.Zstd.decode src dst);
      src)
    (fun src -> stepped (fun () -> Lz.Zstd.decoder ()) src (1 lsl 20))

(* Encoders: a label, the bytes of the corpus they encode ([None] for all
   of it) and the encode, each into a destination of its format's bound.
   The deepest levels encode the first 2 MiB, so that a call of each fits
   the deadline's calibration, warmup and samples. Level 12 sizes its
   tables to an input below its 4 MiB window, 20 MiB for 2 MiB and 10 MiB
   for 1 MiB; at the level's 40 MiB, fresh on each call, 1 MiB of random
   takes ten times as long. Level 1 sizes its window to an input of
   256 KiB, and its fast finder has a faster loop for windows below its
   512 KiB. *)
let deep = Some (2 lsl 20)

let encoders =
  [
    ("deflate 1", None, Lz.Deflate.bound, Lz.Deflate.encode ~level:1);
    ("deflate 6", None, Lz.Deflate.bound, Lz.Deflate.encode ~level:6);
    ("deflate 9 of 2 MiB", deep, Lz.Deflate.bound, Lz.Deflate.encode ~level:9);
    ("zstd 1", None, Lz.Zstd.bound, Lz.Zstd.encode ~level:1);
    ( "zstd 1 of 256 KiB",
      Some (256 lsl 10),
      Lz.Zstd.bound,
      Lz.Zstd.encode ~level:1 );
    ("zstd 3", None, Lz.Zstd.bound, Lz.Zstd.encode ~level:3);
    ("zstd 12 of 2 MiB", deep, Lz.Zstd.bound, Lz.Zstd.encode ~level:12);
    ( "zstd 12 of 1 MiB",
      Some (1 lsl 20),
      Lz.Zstd.bound,
      Lz.Zstd.encode ~level:12 );
    ("lz4", None, Lz.Lz4.Block.bound, Lz.Lz4.Block.encode);
    ("snappy", None, Lz.Snappy.bound, Lz.Snappy.encode);
  ]

(* The ratio each encoder reaches, printed once beside the rows. *)
let encode raw (label, prefix, bound, encode) =
  Thumper.bench_with_setup ("encode " ^ label)
    ~setup:(fun () ->
      let raw = Lazy.force raw in
      let raw =
        match prefix with None -> raw | Some n -> Array1.sub raw 0 n
      in
      let dst = create (bound (Array1.dim raw)) in
      (match encode raw dst with
      | Some n ->
          Printf.eprintf "ratio %s: %.3f\n%!" label
            (float (Array1.dim raw) /. float n)
      | None -> failwith (label ^ ": the output exceeds the bound"));
      (raw, dst))
    (fun (raw, dst) -> encode raw dst)

(* The Zstandard encoder stepped over the corpus in 1 MiB slices, as a
   writer of a stream does, each run handed to a function that reads none
   of it. *)
let encode_steps raw =
  Thumper.bench_with_setup "encode zstd 3 in 1 MiB steps"
    ~setup:(fun () -> Lazy.force raw)
    (fun raw ->
      let e = Lz.Zstd.encoder ~level:3 () in
      let n = Array1.dim raw in
      let rec go at =
        let len = min (1 lsl 20) (n - at) in
        let last = at + len = n in
        match Lz.Encoder.step e ~last (Array1.sub raw at len) ignore with
        | Lz.Done _ -> ()
        | More k -> go (at + k)
      in
      go 0)

let corpus (name, kind) =
  let raw = lazy (Lz_support.of_string (Lz_support.corpus kind size)) in
  let srcs =
    List.map
      (fun (suffix, encode, _) -> (suffix, lazy (encode (Lazy.force raw))))
      decoders
  in
  Thumper.group name
    (List.map
       (fun ((suffix, _, _) as d) -> decode name raw (List.assoc suffix srcs) d)
       decoders
    @ [ small name raw ]
    @ steps srcs
    @ [ flushed_steps name raw ]
    @ List.map (encode raw) encoders
    @ [ encode_steps raw ])

let checksums =
  let sizes = [ ("4 KiB", 4 lsl 10); ("64 KiB", 64 lsl 10); ("8 MiB", 8 lsl 20) ] in
  let raw = lazy (Lz_support.of_string (Lz_support.corpus Random size)) in
  let row (label, size) (name, f) =
    Thumper.bench_with_setup
      (Printf.sprintf "%s of %s" name label)
      ~setup:(fun () -> Array1.sub (Lazy.force raw) 0 size)
      f
  in
  Thumper.group "checksums"
    (List.concat_map
       (fun size ->
         List.map (row size)
           [
             ("crc32", fun b -> ignore (Lz.Crc32.bigbytes b));
             ("adler32", fun b -> ignore (Lz.Adler32.bigbytes b));
             ("xxh64", fun b -> ignore (Lz.Xxh64.bigbytes b));
           ])
       sizes)

(* lz's LZ4 block of 8 MiB of zeros, one match at offset 1, decoded into
   a destination of exactly its size, where the match ends too near the
   destination's end for the fast loop and the exact loop copies it, and
   into one with room past its output, where the fast loop does. *)
let zeros =
  let raw =
    lazy
      (let b = create size in
       Array1.fill b 0;
       b)
  in
  let src =
    lazy (encoded Lz.Lz4.Block.bound Lz.Lz4.Block.encode (Lazy.force raw))
  in
  let row name room =
    let dst = lazy (create (size + room)) in
    Thumper.bench_with_setup name
      ~setup:(fun () ->
        let src = Lazy.force src and dst = Lazy.force dst in
        check "zeros.lz4b" (Lazy.force raw) dst (Lz.Lz4.Block.decode src dst);
        (src, dst))
      (fun (src, dst) -> Lz.Lz4.Block.decode src dst)
  in
  Thumper.group "zeros"
    [ row "decode lz4b, exact destination" 0; row "decode lz4b" 64 ]

let () =
  exit
    (Thumper.run "lz"
       ~budgets:[ Thumper.Budget.no_slower_than 0.10 ]
       (List.map corpus corpora @ [ zeros; checksums ]))
