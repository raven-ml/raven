(* lz's codecs on the corpora reference.py writes: the decoders on the
   vendors' encodings, the encoders and checksums on the raw bytes. Every
   decode row first checks its output against the raw corpus, so a row
   never times a wrong answer. *)

open Bigarray

let corpora = [ "text"; "columns"; "random" ]

let read name =
  let path = Filename.concat "data" name in
  if not (Sys.file_exists path) then begin
    prerr_endline ("missing " ^ path ^ ": run reference.py first, see README.md");
    exit 2
  end;
  let s = In_channel.with_open_bin path In_channel.input_all in
  let b = Array1.create int8_unsigned c_layout (String.length s) in
  String.iteri (fun i c -> Array1.unsafe_set b i (Char.code c)) s;
  b

let create n = Array1.create int8_unsigned c_layout n

(* Decoders: the file suffix reference.py writes and the decode that reads
   it. *)
let decoders =
  [
    ("zlib6", Lz.Zlib.decode);
    ("gz9", Lz.Gzip.decode);
    ("deflate6", Lz.Deflate.decode);
    ("zst1", Lz.Zstd.decode);
    ("zst3", Lz.Zstd.decode);
    ("zst12", Lz.Zstd.decode);
    ("zst19", Lz.Zstd.decode);
    ("lz4b", Lz.Lz4.Block.decode);
    ("lz4f", Lz.Lz4.Frame.decode);
    ("snappy", Lz.Snappy.decode);
  ]

let check name raw dst = function
  | Ok { Lz.consumed = _; produced } when produced = Array1.dim raw ->
      if Array1.sub dst 0 produced <> raw then
        failwith (name ^ ": the output differs from the raw corpus")
  | Ok { produced; _ } ->
      failwith (Printf.sprintf "%s: %d bytes, not %d" name produced (Array1.dim raw))
  | Error { Lz.at; _ } -> failwith (Printf.sprintf "%s: an error at %d" name at)

let decode corpus raw (suffix, decode) =
  let src = lazy (read (corpus ^ "." ^ suffix)) in
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
  let src = lazy (read (corpus ^ ".64k.zst3")) in
  let dst = lazy (create n) in
  Thumper.bench_with_setup "decode zst3 of 64 KiB, exact destination"
    ~setup:(fun () ->
      let src = Lazy.force src and dst = Lazy.force dst in
      let raw = Array1.sub (Lazy.force raw) 0 n in
      check (corpus ^ ".64k.zst3") raw dst (Lz.Zstd.decode src dst);
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

let steps corpus =
  List.concat_map
    (fun (suffix, decoder) ->
      List.map
        (fun (label, slice) ->
          let src = lazy (read (corpus ^ "." ^ suffix)) in
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
   the deadline's calibration, warmup and samples: their speed per byte
   there is within a tenth of the whole corpus's, and level 12's 40 MiB of
   fresh tables take at most a fifth of the call. *)
let deep = Some (2 lsl 20)

let encoders =
  [
    ("deflate 1", None, Lz.Deflate.bound, Lz.Deflate.encode ~level:1);
    ("deflate 6", None, Lz.Deflate.bound, Lz.Deflate.encode ~level:6);
    ("deflate 9 of 2 MiB", deep, Lz.Deflate.bound, Lz.Deflate.encode ~level:9);
    ("zstd 1", None, Lz.Zstd.bound, Lz.Zstd.encode ~level:1);
    ("zstd 3", None, Lz.Zstd.bound, Lz.Zstd.encode ~level:3);
    ("zstd 12 of 2 MiB", deep, Lz.Zstd.bound, Lz.Zstd.encode ~level:12);
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

let corpus name =
  let raw = lazy (read (name ^ ".raw")) in
  Thumper.group name
    (List.map (decode name raw) decoders
    @ [ small name raw ]
    @ steps name
    @ [ flushed_steps name raw ]
    @ List.map (encode raw) encoders
    @ [ encode_steps raw ])

let checksums =
  let sizes = [ ("4 KiB", 4 lsl 10); ("64 KiB", 64 lsl 10); ("8 MiB", 8 lsl 20) ] in
  let raw = lazy (read "random.raw") in
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

let () =
  exit
    (Thumper.run "lz"
       ~budgets:[ Thumper.Budget.no_slower_than 0.10 ]
       (List.map corpus corpora @ [ checksums ]))
