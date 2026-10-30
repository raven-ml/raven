(* lz's side of bench/README.md's table: whole-buffer [encode] and
   [decode] of every format and level, and CRC-32, timed from OCaml as a
   user calls them. vendor_table.c times the vendors with the same rows,
   buffers and protocol; vendor_table.sh runs both.

     vendor_table.exe [-r ROW] DATA DECODE_RUNS ENCODE_RUNS CORPUS...
     vendor_table.exe raw DATA

   Prints one line per row: corpus, row, implementation, MB/s of raw
   bytes, encoded size (0 for a decoder), runs. A row's time is the
   fastest of its runs, after one untimed run that faults the
   destination in and checks the output: a decode must give the raw
   corpus back, an encode must decode to it. [-r ROW] keeps the row named
   ROW.

   [raw] writes DATA/text.raw, DATA/columns.raw and DATA/random.raw, the
   bench's 8 MiB corpora (Lz_support.corpus). *)

open Bigarray

let create n = Array1.create int8_unsigned c_layout n

let read path =
  let s = In_channel.with_open_bin path In_channel.input_all in
  let b = create (String.length s) in
  String.iteri (fun i c -> Array1.unsafe_set b i (Char.code c)) s;
  b

let best runs f =
  let t = ref infinity in
  for _ = 1 to runs do
    let t0 = Unix.gettimeofday () in
    f ();
    let dt = Unix.gettimeofday () -. t0 in
    if dt < !t then t := dt
  done;
  !t

let report corpus row n size runs t =
  Printf.printf "%s\t%s\tlz\t%.0f\t%d\t%d\n%!" corpus row
    (float n /. t /. 1e6)
    size runs

let decoded name raw dst = function
  | Ok { Lz.produced; _ } when produced = Array1.dim raw ->
      if Array1.sub dst 0 produced <> raw then
        failwith (name ^ ": the output differs from the raw corpus")
  | Ok { produced; _ } -> failwith (Printf.sprintf "%s: %d bytes" name produced)
  | Error { Lz.at; _ } -> failwith (Printf.sprintf "%s: an error at %d" name at)

(* Decoders: the file suffix vendor_table.c's [encode] writes, the row and
   the decode. *)
let decoders =
  [
    ("deflate6", "decode deflate", Lz.Deflate.decode);
    ("zlib6", "decode zlib", Lz.Zlib.decode);
    ("gz9", "decode gzip", Lz.Gzip.decode);
    ("zst1", "decode zstd 1", Lz.Zstd.decode);
    ("zst3", "decode zstd 3", Lz.Zstd.decode);
    ("zst12", "decode zstd 12", Lz.Zstd.decode);
    ("zst19", "decode zstd 19", Lz.Zstd.decode);
    ("lz4b", "decode lz4 block", Lz.Lz4.Block.decode);
    ("lz4f", "decode lz4 frame", Lz.Lz4.Frame.decode);
    ("snappy", "decode snappy", Lz.Snappy.decode);
  ]

(* Encoders: the row, the bound, the encode and the decode that checks
   it. *)
let encoders =
  let levels name top bound encode decode =
    List.init top (fun i ->
        let row = Printf.sprintf "encode %s %d" name (i + 1) in
        (row, bound, encode (i + 1), decode))
  in
  List.concat
    [
      levels "deflate" 9 Lz.Deflate.bound
        (fun level -> Lz.Deflate.encode ~level)
        Lz.Deflate.decode;
      levels "zlib" 9 Lz.Zlib.bound
        (fun level -> Lz.Zlib.encode ~level)
        Lz.Zlib.decode;
      levels "gzip" 9 Lz.Gzip.bound
        (fun level -> Lz.Gzip.encode ~level)
        Lz.Gzip.decode;
      levels "zstd" 12 Lz.Zstd.bound
        (fun level -> Lz.Zstd.encode ~level)
        Lz.Zstd.decode;
      [
        ( "encode lz4 block",
          Lz.Lz4.Block.bound,
          Lz.Lz4.Block.encode,
          Lz.Lz4.Block.decode );
        ( "encode lz4 frame",
          Lz.Lz4.Frame.bound,
          Lz.Lz4.Frame.encode,
          Lz.Lz4.Frame.decode );
        ("encode snappy", Lz.Snappy.bound, Lz.Snappy.encode, Lz.Snappy.decode);
      ];
    ]

let crc_runs = 50

let corpus ~keep data drun erun name =
  let file suffix = Filename.concat data (name ^ "." ^ suffix) in
  let raw = read (file "raw") in
  let n = Array1.dim raw in
  let cap =
    List.fold_left (fun m (_, bound, _, _) -> max m (bound n)) n encoders
  in
  let buf = create cap and back = create n in
  List.iter
    (fun (suffix, row, decode) ->
      if keep row then begin
        let src = read (file suffix) and dst = Array1.sub buf 0 n in
        decoded row raw dst (decode src dst);
        let t = best drun (fun () -> ignore (decode src dst)) in
        report name row n 0 drun t
      end)
    decoders;
  List.iter
    (fun (row, bound, encode, decode) ->
      if keep row then begin
        let dst = Array1.sub buf 0 (bound n) in
        let size =
          match encode raw dst with
          | Some k -> k
          | None -> failwith (row ^ ": the output exceeds the bound")
        in
        decoded row raw back (decode (Array1.sub dst 0 size) back);
        let t = best erun (fun () -> ignore (encode raw dst)) in
        report name row n size erun t
      end)
    encoders;
  if keep "crc32" then begin
    let t = best crc_runs (fun () -> ignore (Lz.Crc32.bigbytes raw)) in
    report name "crc32" n 0 crc_runs t
  end

let raw data =
  List.iter
    (fun (name, kind) ->
      Out_channel.with_open_bin
        (Filename.concat data (name ^ ".raw"))
        (fun oc -> output_string oc (Lz_support.corpus kind (8 lsl 20))))
    [ ("text", Lz_support.Text); ("columns", Columns); ("random", Random) ]

let usage () =
  prerr_endline
    "usage: vendor_table.exe [-r ROW] DATA DECODE_RUNS ENCODE_RUNS CORPUS...\n\
    \       vendor_table.exe raw DATA";
  exit 2

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let keep, args =
    match args with
    | "-r" :: row :: rest -> (String.equal row, rest)
    | _ -> ((fun _ -> true), args)
  in
  match args with
  | [ "raw"; data ] -> raw data
  | data :: drun :: erun :: (_ :: _ as corpora) -> (
      match (int_of_string_opt drun, int_of_string_opt erun) with
      | Some drun, Some erun when drun > 0 && erun > 0 ->
          List.iter (corpus ~keep data drun erun) corpora
      | _ -> usage ())
  | _ -> usage ()
