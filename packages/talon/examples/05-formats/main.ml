(* Read and write CSV text, and read a Parquet file. *)

open Talon

let ( let* ) = Result.bind

let csv =
  {|station,day,temp,note
oslo,2024-03-01,-3.5,
oslo,2024-03-02,NA,"sensor ""B"" offline"
lima,2024-03-01,22,"humid, calm"
|}

let read_csv () =
  let reader () = Bytesrw.Bytes.Reader.of_string csv in
  let* f = Talon_csv.sniff ~nulls:[ "NA" ] (reader ()) in
  Format.printf "%a@.@." Talon_csv.pp_format f;
  let f = Talon_csv.with_type "temp" (Type.Any Type.float32) f in
  let* t = Talon_csv.decode f (reader ()) in
  Format.printf "%a@.@." Talon.pp t;

  (* Encoding writes what decoding reads back. *)
  let buf = Buffer.create 256 in
  let* () =
    Talon_csv.encode f (Query.of_table t) (Bytesrw.Bytes.Writer.of_buffer buf)
  in
  Format.printf "%s@." (Buffer.contents buf);
  Ok ()

let bad_field () =
  let csv = "station,temp\noslo,-3.5\nlima,warm\n" in
  let f =
    Talon_csv.format
      [ ("station", Type.Any Type.string); ("temp", Type.Any Type.float64) ]
  in
  match Talon_csv.decode f (Bytesrw.Bytes.Reader.of_string csv) with
  | Ok _ -> ()
  | Error e -> Format.printf "%a@.@." Error.pp e

let read_parquet () =
  let path = "carriers.parquet" in
  let* b =
    Nx_device.Buffer.of_file path |> Result.map_error (Error.v ~file:path)
  in
  let* f = Talon_parquet.sniff b in
  Format.printf "%a@.@." Talon_parquet.pp_format f;

  (* The source skips the row groups whose statistics rule a filter out. *)
  let q =
    Query.(
      of_source (Talon_parquet.source f b)
      |> filter Expr.(Col.string "carrier" >= string "F"))
  in
  Format.printf "%a@.@." Query.pp (Query.optimize q);
  let* t = Query.run q in
  Format.printf "%a@." Talon.pp t;
  Ok ()

let () =
  Error.get_ok (read_csv ());
  bad_field ();
  Error.get_ok (read_parquet ())
