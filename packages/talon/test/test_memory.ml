(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A column's buffers count in Rig.allocated while it is live, and return to the
   host once it is collected (Column's preamble), which {!Talon_held.check}
   asserts of the columns talon makes. Drawn columns hold up to forty rows, some
   of them null and none in at least one case of five. *)

open Talon
open Windtrap
module G = Talon_gen

(* [computed c out] is the column [out] computes over the table whose column [s]
   is [c]. *)
let computed c out =
  let t = Talon.v [ ("s", c) ] in
  let q = Query.select [ out ] (Query.of_table t) in
  Talon.column (Result.get_ok (Query.run q)) "r"

(* The text kernels, each the column [r] over the text column [s]. *)
let kernels =
  let s = Col.string "s" in
  let kernel name e = (name, Expr.("r" := e)) in
  Expr.
    [
      kernel "slice" (Str.slice ~offset:1 ~length:2 s);
      kernel "slice from the end" (Str.slice ~offset:(-2) ~length:5 s);
      kernel "trim" (Str.trim s);
      kernel "length" (Str.length s);
      kernel "matches a literal" (Str.matches (Str.literal "a") s);
      kernel "matches a prefix" (Str.matches (Str.prefix "a") s);
      kernel "matches a suffix" (Str.matches (Str.suffix "a") s);
      kernel "matches pieces" (Str.matches (Str.pieces [ "a"; "b" ]) s);
      kernel "split" (Str.split (Str.literal "a") s);
      kernel "replace" (Str.replace (Str.literal "a") ~by:"xyz" s);
      kernel "concat" (Str.concat "," [ s; string "é"; s ]);
      kernel "compare" (s < string "m");
      kernel "equal" (s = string "a");
    ]

let kernel =
  Gen.pair (Gen.of_list (List.map fst kernels)) (G.options Type.string)
  |> Gen.with_pp (fun ppf (k, rows) ->
      Format.fprintf ppf "%s computed %d rows" k (Array.length rows))

(* Each of the 13 kernels is drawn in a case of 13: at 300 cases one is missed
   with probability (12/13)^300 = 4e-11. *)
let law_kernel (k, rows) =
  List.iter (fun (k', _) -> cover k' (String.equal k k')) kernels;
  cover "no row" (Array.length rows = 0);
  cover "a null" (Array.exists Option.is_none rows);
  let c = Column.of_options Type.string rows in
  Talon_held.check ~inputs:[ c ] (fun () -> computed c (List.assoc k kernels))

(* A value and a null of each type that [Column.parse] reads and [Column.print]
   writes, which the law runs first. *)
let parsed =
  let day n = Option.get (Time.Date.of_days n) in
  let rows ty x = G.Sample (ty, [| Some x; None |]) in
  Type.
    [
      rows bool true;
      rows int8 (-128);
      rows int16 32767;
      rows int32 (-1);
      rows int64 min_int;
      rows uint8 255;
      rows uint16 0;
      rows uint32 0xffff_ffff;
      rows uint64 max_int;
      rows float16 0.5;
      rows float32 (-0.);
      rows float64 Float.nan;
      rows string "é";
      rows binary (Binary.of_string "\xff");
      rows date (day (-1));
      rows (wall S) (Time.of_ns 0L);
      rows (wall Ns) (Time.of_ns 1L);
      rows (instant Us) (Time.of_ns (-1000L));
      rows (categorical [| "a"; "b"; "é" |]) "é";
    ]

let sample =
  Gen.bind
    (Gen.of_list (List.map (fun (G.Sample (ty, _)) -> Type.Any ty) parsed))
    (fun (Type.Any ty) ->
      Gen.map (fun rows -> G.Sample (ty, rows)) (G.options ty))
  |> Gen.with_pp G.pp_sample

let name ty = Format.asprintf "%a" Type.pp ty

(* [Column.print] of a column and [Column.parse] of its texts. *)
let law_parse (G.Sample (ty, rows)) =
  List.iter
    (fun (G.Sample (t, _)) -> cover (name t) (String.equal (name ty) (name t)))
    parsed;
  cover "no row" (Array.length rows = 0);
  let c = Column.of_options ty rows in
  Talon_held.check ~inputs:[ c ] (fun () -> Column.print c);
  let texts = Column.print c in
  Talon_held.check ~inputs:[ texts ] (fun () ->
      Result.get_ok (Column.parse (Any ty) texts))

(* [Temporal.parse] of dates printed in its format. *)
let law_parse_with rows =
  cover "no row" (Array.length rows = 0);
  let texts = Column.print (Column.of_options Type.date rows) in
  let parse =
    Expr.("r" := Temporal.parse "%Y-%m-%d" Type.date (Col.string "s"))
  in
  Talon_held.check ~inputs:[ texts ] (fun () -> computed texts parse)

(* A streamed aggregate holds its state, at most one row a group, and held rows
   that a compaction empties once they reach the groups: its host bytes between
   batches follow groups + a batch, [bytes] a row. A row of a held table holds
   24 bytes for a sum: the key, the sum and the row. A [var] holds 72: the key,
   the row and its state, seven fields of 8 bytes. A batch of the input holds 16
   more until it is reduced. *)
let law_peak (out, bytes) (groups, batch) =
  let batches = 40 in
  let table b =
    let key i = ((b * batch) + i) mod groups in
    Talon.v
      [
        ("k", Column.v Type.int64 (Array.init batch key));
        ("x", Column.v Type.float64 (Array.init batch float_of_int));
      ]
  in
  let produced = ref 0 and peak = ref 0 in
  let baseline = Talon_held.settled () in
  let next () =
    if !produced = batches then Ok None
    else begin
      peak := max !peak (Talon_held.settled () - baseline);
      let t = table !produced in
      incr produced;
      Ok (Some t)
    end
  in
  let open_ () = Ok { Source.next; close = ignore } in
  let part = { Source.rows = None; open_ } in
  let schema = Talon.schema (table 0) in
  let source = Source.v ~name:"stream" ~schema (fun _ -> Ok [ part ]) in
  let q = Query.(of_source source |> aggregate ~by:[ "k" ] [ out ]) in
  let result = Result.get_ok (Query.run q) in
  equal int groups (Talon.rows result);
  let limit = bytes * ((2 * groups) + (2 * batch)) in
  less ~msg:"peak bytes" int ~than:limit !peak

let () =
  exit
    (run "talon memory"
       [
         group "Column"
           [
             prop ~count:300 ~examples:parsed
               "a printed or parsed column is the host's memory" sample
               law_parse;
           ];
         group "Expr.Str"
           [
             prop ~count:300 "a text kernel's column is the host's memory"
               kernel law_kernel;
           ];
         group "Expr.Temporal"
           [
             prop ~count:50 ~examples:[ [||] ]
               "a column parse reads is the host's memory" (G.options Type.date)
               law_parse_with;
           ];
         group "Query.aggregate"
           [
             prop ~count:6
               "a streamed sum's bytes follow its groups and a batch"
               Gen.(pair (int_range 1 3000) (int_range 100 2000))
               (law_peak (Expr.("s" := sum (Col.float "x")), 64));
             prop ~count:6
               "a streamed var's bytes follow its groups and a batch"
               Gen.(pair (int_range 1 3000) (int_range 100 2000))
               (law_peak (Expr.("s" := var (Col.float "x")), 192));
           ];
       ])
