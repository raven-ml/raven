let encode codec level src =
  let n = Bigarray.Array1.dim src in
  let with_bound bound f =
    let dst = Lz_support.create (bound n) in
    match f src dst with
    | Some len -> Lz_support.to_string ~length:len dst
    | None -> failwith "the output exceeds the bound"
  in
  match codec with
  | "deflate" -> with_bound Lz.Deflate.bound (Lz.Deflate.encode ~level)
  | "zlib" -> with_bound Lz.Zlib.bound (Lz.Zlib.encode ~level)
  | "gzip" -> with_bound Lz.Gzip.bound (Lz.Gzip.encode ~level)
  | "zstd" -> with_bound Lz.Zstd.bound (Lz.Zstd.encode ~level)
  | "lz4b" -> with_bound Lz.Lz4.Block.bound Lz.Lz4.Block.encode
  | "lz4f" -> with_bound Lz.Lz4.Frame.bound Lz.Lz4.Frame.encode
  | "snappy" -> with_bound Lz.Snappy.bound Lz.Snappy.encode
  | c -> invalid_arg ("unknown codec " ^ c)

let () =
  match Sys.argv with
  | [| _; codec; level; spec |] ->
      let src = Lz_support.of_string (Lz_support.raw_of_spec spec) in
      set_binary_mode_out stdout true;
      print_string (encode codec (int_of_string level) src)
  | _ ->
      prerr_endline "usage: encode.exe CODEC LEVEL RAW";
      exit 2
