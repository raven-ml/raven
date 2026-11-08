(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The tables whose files pyarrow reads: every written type, nulls at every
   level, in each framing and codec. *)

open Talon
module T = Type

type item = {
  name : string;
  framing : Talon_arrow.framing;
  compression : Talon_arrow.compression;
  batch : int;
  table : Talon.t;
}

let n = 37
let opt f = Array.init n (fun i -> if i mod 7 = 0 then None else Some (f i))
let col ty f = Column.of_options ty (opt f)
let span ns = Time.Span.of_ns (Int64.of_int ns)
let stamp f i = Option.get (f (Int64.of_int ((i * 1000) - 5000)))

let table =
  let text i = [| ""; "a"; "é"; "日本語"; "abc" |].(i mod 5) in
  let cat = T.categorical [| "x"; "y"; "z" |] in
  let tensor =
    Column.of_tensor (Nx.reshape [| n; 3 |] (Nx.arange Nx.int32 0 (n * 3) 1))
  in
  let record = T.record [ ("a", Any T.int64); ("b", Any T.string) ] in
  let cell i =
    Record.empty
    |> Record.add Kind.int "a" (if i mod 3 = 0 then None else Some i)
    |> Record.add Kind.string "b" (Some (text i))
  in
  Talon.v
    [
      ("bool", col T.bool (fun i -> i mod 2 = 0));
      ("int8", col T.int8 (fun i -> i - 18));
      ("int16", col T.int16 (fun i -> i * 100));
      ("int32", col T.int32 (fun i -> i * 10_000));
      ("int64", col T.int64 (fun i -> i * 1_000_000_000));
      ("uint8", col T.uint8 Fun.id);
      ("uint16", col T.uint16 (fun i -> i * 1000));
      ("uint32", col T.uint32 (fun i -> i * 100_000));
      ("uint64", col T.uint64 (fun i -> i * 1_000_000_000));
      ("float16", col T.float16 (fun i -> float_of_int i /. 2.));
      ("float32", col T.float32 (fun i -> float_of_int i /. 4.));
      ("float64", col T.float64 (fun i -> float_of_int i /. 8.));
      ("string", col T.string text);
      ("binary", col T.binary (fun i -> Binary.of_string (String.make (i mod 4) '\001')));
      ("category", col cat (fun i -> [| "x"; "y"; "z" |].(i mod 3)));
      ("date", col T.date (fun i -> Option.get (Time.Date.of_days (i * 7))));
      ("clock_s", col (T.clock S) (fun i -> span (i * 1_000_000_000)));
      ("clock_ms", col (T.clock Ms) (fun i -> span (i * 1_000_000)));
      ("clock_us", col (T.clock Us) (fun i -> span (i * 1_000)));
      ("clock_ns", col (T.clock Ns) (fun i -> span i));
      ("duration_ms", col (T.duration Ms) (fun i -> span (i * 1_000_000)));
      ("duration_ns", col (T.duration Ns) (fun i -> span (i - 10)));
      ("instant_us", col (T.instant Us) (stamp Time.of_us));
      ("wall_ms", col (T.wall Ms) (stamp Time.of_ms));
      ("list", col (T.list T.int32) (fun i -> Array.init (i mod 4) (fun j -> i + j)));
      ("record", col record cell);
      ("tensor", tensor);
    ]

let rows k = Talon.take (Nx.arange Nx.int64 0 k 1) table

let items =
  let item name framing compression batch table =
    { name; framing; compression; batch; table }
  in
  [
    item "file.arrow" File Uncompressed 10 table;
    item "stream.arrows" Stream Uncompressed 1000 table;
    item "lz4.arrow" File Lz4_frame 16 table;
    item "zstd.arrows" Stream Zstd 16 table;
    item "empty.arrow" File Uncompressed 10 (rows 0);
    item "one_row.arrows" Stream Uncompressed 10 (rows 2);
  ]

let bytes it =
  let q = Query.of_table it.table in
  let f =
    Talon_arrow.format ~framing:it.framing ~compression:it.compression
      ~batch:it.batch (Query.schema q)
  in
  let b = Buffer.create 4096 in
  Error.get_ok (Talon_arrow.encode f q (Bytesrw.Bytes.Writer.of_buffer b));
  Buffer.contents b
