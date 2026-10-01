(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decoders on the corpora reference.py writes compressed by the reference
   encoders, and the encoders on the raw corpora. *)

open Bigarray

let data name =
  let path = Filename.concat "data" name in
  if not (Sys.file_exists path) then begin
    prerr_endline ("missing " ^ path ^ ": run reference.py first, see README.md");
    exit 2
  end;
  let s = In_channel.with_open_bin path In_channel.input_all in
  let b = Array1.create int8_unsigned c_layout (String.length s) in
  String.iteri (fun i c -> Array1.unsafe_set b i (Char.code c)) s;
  b

let decoders =
  [
    ("zlib6", Compress_deflate.Zlib.decompress_into);
    ("deflate6", Compress_deflate.Deflate.decompress_into);
    ("gz", Compress_deflate.Gzip.decompress_into);
    ("snappy", Compress_snappy.decompress_into);
    ("lz4b", Compress_lz4.Block.decompress_into);
    ("lz4f", Compress_lz4.Frame.decompress_into);
    ("zst3", Compress_zstd.decompress_into);
    ("zst19", Compress_zstd.decompress_into);
  ]

let corpus name =
  let raw = data (name ^ ".raw") in
  let n = Array1.dim raw in
  let dst = Array1.create int8_unsigned c_layout n in
  let decode (codec, decompress) =
    let src = data (name ^ "." ^ codec) in
    Thumper.bench ("Decompress " ^ codec) (fun () ->
        match decompress src dst with Ok () -> () | Error e -> failwith e)
  in
  let snappy =
    Array1.create int8_unsigned c_layout
      (Compress_snappy.max_compressed_length n)
  in
  let s = String.init n (fun i -> Char.unsafe_chr (Array1.unsafe_get raw i)) in
  let deflate level () = ignore (Compress_deflate.Zlib.compress ~level s) in
  Thumper.group name
    (List.map decode decoders
    @ [
        Thumper.bench "Compress zlib level 1" (deflate 1);
        Thumper.bench "Compress zlib level 6" (deflate 6);
        Thumper.bench "Compress snappy" (fun () ->
            Compress_snappy.compress_into raw snappy);
      ])

let () =
  Thumper.run "compress"
    ~budgets:
      [ Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.10 ]
    (List.map corpus [ "text"; "columns"; "random" ])
