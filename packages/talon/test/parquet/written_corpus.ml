(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The tables whose files pyarrow reads: every type, nulls at every level of
   nesting, the statistics' cases and the codecs, drawn from a fixed seed. *)

open Talon
module T = Type

let rng = Random.State.make [| 0x7a10 |]
let chance p = Random.State.float rng 1. < p
let opt p f () = if chance p then None else Some (f ())
let ints lo hi () = lo + Random.State.full_int rng (hi - lo + 1)
let picks vs () = vs.(Random.State.int rng (Array.length vs))

let floats = [| 1.5; -2.25; 0.; -0.; Float.infinity; Float.neg_infinity; 3.25 |]
let doubles = Array.append floats [| 1e300; Float.nan; 1e-310 |]

let texts =
  [| ""; "a"; "é"; "日本語"; "🦊 fox"; "abc"; "abd"; "zzz"; "xxx" |]

let v (ty : 'a T.t) (f : unit -> 'a option) n =
  Column.of_options ty (Array.init n (fun _ -> f ()))

let n = 64
let ticks f lo hi () = f (Int64.of_int (ints lo hi ()))
let span unit' = ticks (fun t -> Time.Span.of_ns (Int64.mul unit' t))
let stamp f = ticks (fun t -> Option.get (f t)) (-1_000_000) 2_000_000_000
let days lo hi () = Option.get (Time.Date.of_days (ints lo hi ()))

let random_bytes () =
  let n = ints 0 3 () in
  Binary.of_string (String.init n (fun _ -> Char.chr (ints 0 255 ())))

(* The flat columns, every type at nulls one row in five. *)
let flat () =
  let c name ty f = (name, v ty (opt 0.2 f) n) in
  Talon.v
    [
      c "bool" T.bool (fun () -> chance 0.5);
      c "int8" T.int8 (ints (-128) 127);
      c "int16" T.int16 (ints (-32768) 32767);
      c "int32" T.int32 (ints (-5) 5);
      c "int64" T.int64 (ints (-1_000_000) 1_000_000);
      c "uint8" T.uint8 (ints 0 255);
      c "uint16" T.uint16 (ints 0 65535);
      c "uint32" T.uint32 (ints 0 0x7FFFFFFF);
      c "uint64" T.uint64 (ints 0 (max_int - 1));
      c "float16" T.float16 (picks [| 1.5; -2.5; 0.; Float.nan |]);
      c "float32" T.float32 (picks (Array.append floats [| Float.nan |]));
      c "float64" T.float64 (picks doubles);
      c "string" T.string (picks texts);
      c "binary" T.binary random_bytes;
      c "category" (T.categorical [| "red"; "green"; "blue" |])
        (picks [| "red"; "green"; "blue" |]);
      c "date" T.date (days (-10000) 30000);
      c "clock_s" (T.clock S) (span 1_000_000_000L 0 86399);
      c "clock_ms" (T.clock Ms) (span 1_000_000L 0 86399999);
      c "clock_us" (T.clock Us) (span 1000L 0 8639999);
      c "clock_ns" (T.clock Ns) (span 1L 0 86399999);
      c "wall_s" (T.wall S) (stamp Time.of_s);
      c "wall_ms" (T.wall Ms) (stamp Time.of_ms);
      c "wall_us" (T.wall Us) (stamp Time.of_us);
      c "wall_ns" (T.wall Ns) (stamp (fun t -> Some (Time.of_ns t)));
      c "instant_ms" (T.instant Ms) (stamp Time.of_ms);
      c "instant_us" (T.instant Us) (stamp Time.of_us);
      c "instant_ns" (T.instant Ns) (stamp (fun t -> Some (Time.of_ns t)));
    ]

(* A list of nullable elements cannot read as an array of options (a
   column's list reader refuses a null element), so nested columns are built
   through [Column.of_layout]. *)
let laid_out ty layout =
  match Column.of_layout ty layout with
  | Ok c -> c
  | Error (_, why) -> failwith why

let rec nested_col (T.Any ty as any) n : Column.t =
  let bit () =
    Nx.cast Nx.bit
      (Nx.create Nx.bool [| n |] (Array.init n (fun _ -> not (chance 0.2))))
  in
  match ty with
  | List e ->
      let lens = Array.init n (fun _ -> ints 0 3 ()) in
      let offsets =
        Nx.create Nx.int64 [| n + 1 |]
          (Array.init (n + 1) (fun i ->
               Int64.of_int (Array.fold_left ( + ) 0 (Array.sub lens 0 i))))
      in
      let child = nested_col (Any e) (Array.fold_left ( + ) 0 lens) in
      laid_out any (Varsize { validity = Some (bit ()); offsets; child })
  | Record fields ->
      let fields = List.map (fun (f, t) -> (f, nested_col t n)) fields in
      laid_out any (Children { validity = Some (bit ()); length = n; fields })
  | Int32 -> v T.int32 (opt 0.3 (ints (-9) 9)) n
  | Int16 -> v T.int16 (opt 0.3 (ints (-9) 9)) n
  | Float64 -> v T.float64 (opt 0.3 (picks doubles)) n
  | String -> v T.string (opt 0.3 (picks texts)) n
  | Categorical words ->
      v ty (opt 0.3 (picks (Iarray.to_array words))) n
  | Date -> v T.date (opt 0.3 (days 0 1000)) n
  | Bool -> v T.bool (opt 0.3 (fun () -> chance 0.5)) n
  | Ext { storage; _ } ->
      laid_out any (Column.layout (nested_col (Any storage) n))
  | _ -> assert false

let nested () =
  let n = 40 in
  let c name (T.Any _ as t) = (name, nested_col t n) in
  let record = T.record in
  Talon.v
    [
      c "ints" (T.Any (T.list T.int32));
      c "texts" (T.Any (T.list T.string));
      c "ll" (T.Any (T.list (T.list T.int16)));
      c "rec" (T.Any (record [ ("a", T.Any T.int32); ("b", T.Any T.string) ]));
      c "lrec"
        (T.Any
           (T.list (record [ ("x", T.Any T.float64); ("y", T.Any T.bool) ])));
      c "rlist" (T.Any (record [ ("xs", T.Any (T.list T.date)) ]));
      c "rrec"
        (T.Any (record [ ("r", T.Any (record [ ("m", T.Any T.int32) ])) ]));
      c "cats" (T.Any (T.list (T.categorical [| "u"; "v" |])));
      c "ext" (T.Any (T.ext ~name:"unit" (T.list T.int32)));
    ]

let tensor () =
  let n = 5 in
  let valid = Nx.create Nx.bit [| n |] [| true; false; true; true; false |] in
  let x = Nx.arange Nx.int32 0 (n * 6) 1 in
  let x = Nx.reshape [| n; 2; 3 |] x in
  let c = Column.of_tensor ~validity:valid x in
  Talon.v [ ("t", c) ]

(* Statistics: truncated text and bytes, all null, all NaN, empty strings and
   text around a truncation boundary. *)
let stats () =
  let n = 8 in
  let long c = String.make 70 c in
  let boundary =
    [| String.make 63 'q' ^ "日本"; String.make 62 'q' ^ "🦊" |]
  in
  let scalars =
    "zzz" ^ String.concat "" (List.init 30 (fun _ -> "\u{10FFFF}"))
  in
  let ffs = String.make 70 '\xff' in
  let some ty f = v ty (fun () -> Some (f ())) n in
  Talon.v
    [
      ( "long_text",
        some T.string (picks [| long 'a' ^ "é"; long 'b'; "short" |]) );
      ("boundary", some T.string (picks boundary));
      ("max_scalar", some T.string (fun () -> scalars));
      ("binary_ff", some T.binary (fun () -> Binary.of_string ffs));
      ("all_null", v T.int32 (fun () -> None) n);
      ("all_nan", v T.float64 (fun () -> Some Float.nan) n);
      ("zeros", v T.float64 (picks [| Some 0.; Some (-0.) |]) n);
      ("empty_strings", v T.string (fun () -> Some "") n);
      ("big_u32", v T.uint32 (picks [| Some 1; Some 0xFFFFFFFF; None |]) n);
      ("big_u64", v T.uint64 (picks [| Some 1; Some max_int; None |]) n);
    ]

let empty () =
  Talon.v
    ~rows:0
    [
      ("i", Column.of_options T.int32 [||]);
      ("s", Column.of_options T.string [||]);
      ("l", nested_col (T.Any (T.list T.int32)) 0);
    ]

let one_row () =
  Talon.v
    [
      ("i", Column.of_options T.int64 [| Some 7 |]);
      ("s", Column.of_options T.string [| Some "only" |]);
      ("n", Column.of_options T.float64 [| None |]);
    ]

(* Tables past a data page's size, whose files are not committed. *)
let pages_int () =
  Talon.v [ ("i", v T.int64 (opt 0.1 (ints 0 (max_int - 1))) 300_000) ]

let pages_text () =
  let row = ref 0 in
  let text () =
    incr row;
    Some (Printf.sprintf "%d-%x" !row (ints 0 (max_int - 1) ()))
  in
  Talon.v [ ("t", v T.string text 300_000) ]

let pages_nested () =
  let rows = 21 and per_row = 20_000 in
  let offsets =
    Nx.create Nx.int64 [| rows + 1 |]
      (Array.init (rows + 1) (fun i -> Int64.of_int (i * per_row)))
  in
  let child =
    v T.float64
      (opt 0.2 (fun () -> Random.State.float rng 1e6))
      (rows * per_row)
  in
  let any = T.Any (T.list T.float64) in
  Talon.v
    [
      ( "l",
        laid_out any
          (Varsize { validity = None; offsets; child }) );
    ]

(* The corpus. *)
type item = {
  name : string;
  table : Talon.t;
  compression : Talon_parquet.compression;
  row_group : int;
  large : bool;
}

let item ?(compression = Talon_parquet.Snappy) ?(row_group = 1 lsl 20)
    ?(large = false) name table =
  { name; table; compression; row_group; large }

(* Each table is drawn in turn, so that the generator's order is the same with
   any compiler. *)
let items =
  let f = flat () in
  let nested = nested () in
  let tensor = tensor () in
  let stats = stats () in
  let empty = empty () in
  let one_row = one_row () in
  let pages_int = pages_int () in
  let pages_text = pages_text () in
  let pages_nested = pages_nested () in
  [
    item "flat_snappy" f;
    item "flat_none" f ~compression:Uncompressed;
    item "flat_gzip" f ~compression:Gzip;
    item "flat_zstd" f ~compression:Zstd;
    item "flat_lz4" f ~compression:Lz4_raw;
    item "row_groups" f ~row_group:10;
    item "nested" nested;
    item "tensor" tensor;
    item "statistics" stats;
    item "empty" empty;
    item "one_row" one_row;
    item "pages_int" pages_int ~large:true;
    item "pages_text" pages_text ~large:true;
    item "pages_nested" pages_nested ~row_group:14 ~large:true;
  ]
