(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

let sizes = [ "1e6"; "1e7"; "1e8" ]

let rows size =
  if not (List.mem size sizes) then
    invalid_arg
      (Printf.sprintf "unknown H2O size %S; expected one of %s" size
         (String.concat ", " sizes));
  int_of_float (float_of_string size)

(* [pretty n] is db-benchmark's name for the power of ten [n]: [1e7]. *)
let pretty n = Printf.sprintf "1e%d" (String.length (string_of_int n) - 1)
let directory data size = Filename.concat data ("h2o-" ^ size)
let string = Type.Any Type.string
let int32 = Type.Any Type.int32
let float64 = Type.Any Type.float64
let question name query = { Workload.name; query }

(* Group-by *)

let groupby_schema =
  [
    ("id1", string);
    ("id2", string);
    ("id3", string);
    ("id4", int32);
    ("id5", int32);
    ("id6", int32);
    ("v1", int32);
    ("v2", int32);
    ("v3", float64);
  ]

let v1 = Col.int "v1"
let v2 = Col.int "v2"
let v3 = Col.float "v3"

(* [r2 x y] is the squared Pearson correlation of [x] and [y]: their sample
   covariance, from the mean of their product, over their standard
   deviations. *)
let r2 x y =
  Expr.(
    let x = cast Type.float64 x and y = cast Type.float64 y in
    let n = cast Type.float64 (count x) in
    let cov = (mean (x *. y) -. (mean x *. mean y)) *. n /. (n -. float 1.) in
    let r = cov /. (std x *. std y) in
    r *. r)

let groupby_questions =
  let by keys outputs t = Query.aggregate ~by:keys outputs (t "x") in
  [
    question "q01" (by [ "id1" ] Expr.[ "v1" := sum v1 ]);
    question "q02" (by [ "id1"; "id2" ] Expr.[ "v1" := sum v1 ]);
    question "q03" (by [ "id3" ] Expr.[ "v1" := sum v1; "v3" := mean v3 ]);
    question "q04"
      (by [ "id4" ] Expr.[ "v1" := mean v1; "v2" := mean v2; "v3" := mean v3 ]);
    question "q05"
      (by [ "id6" ] Expr.[ "v1" := sum v1; "v2" := sum v2; "v3" := sum v3 ]);
    question "q06"
      (by [ "id4"; "id5" ] Expr.[ "median_v3" := median v3; "sd_v3" := std v3 ]);
    question "q07" (by [ "id3" ] Expr.[ "range_v1_v2" := max v1 - min v2 ]);
    (* The two largest values per group rank 1 and 2 descending; a tie for
       second would keep a third row where DuckDB's [max(v3, 2)] keeps two. *)
    question "q08" (fun t ->
        Query.(
          t "x"
          |> filter Expr.(not (is_null v3))
          |> filter Expr.(over ~by:[ "id6" ] (rank (float 0. -. v3)) <= int 2)
          |> select Expr.[ keep (Sel.names [ "id6" ]); "largest2_v3" := v3 ]));
    question "q09" (by [ "id2"; "id4" ] Expr.[ "r2" := r2 v1 v2 ]);
    question "q10"
      (by
         [ "id1"; "id2"; "id3"; "id4"; "id5"; "id6" ]
         Expr.[ "v3" := sum v3; "count" := rows ]);
  ]

let groupby ~data size =
  ignore (rows size : int);
  let file =
    Filename.concat (directory data size)
      (Printf.sprintf "G1_%s_1e2_0_0.csv" size)
  in
  {
    Workload.id = "groupby/" ^ size;
    tables = [ ("x", Workload.Csv (file, groupby_schema)) ];
    questions = groupby_questions;
    ordered = false;
  }

(* Join *)

let join_schemas =
  [
    ( "x",
      [
        ("id1", int32);
        ("id2", int32);
        ("id3", int32);
        ("id4", string);
        ("id5", string);
        ("id6", string);
        ("v1", float64);
      ] );
    ("small", [ ("id1", int32); ("id4", string); ("v2", float64) ]);
    ( "medium",
      [
        ("id1", int32);
        ("id2", int32);
        ("id4", string);
        ("id5", string);
        ("v2", float64);
      ] );
    ( "big",
      [
        ("id1", int32);
        ("id2", int32);
        ("id3", int32);
        ("id4", string);
        ("id5", string);
        ("id6", string);
        ("v2", float64);
      ] );
  ]

(* [joined ?kind right key t] is [x] joined with the table [right] on [key],
   [right]'s other columns but [v2] prefixed with its name, as the SQL's [AS]
   clauses name them. *)
let joined ?kind right key t =
  let keep = [ key; "v2" ] in
  let prefixed =
    List.filter_map
      (fun (c, _) ->
        if List.mem c keep then None else Some (c, right ^ "_" ^ c))
      (List.assoc right join_schemas)
  in
  Query.join ?kind ~on:(Join.keys [ key ])
    (Kit.rename prefixed (t right))
    (t "x")

let join_questions =
  [
    question "q01" (joined "small" "id1");
    question "q02" (joined "medium" "id2");
    question "q03" (joined ~kind:Left "medium" "id2");
    question "q04" (joined "medium" "id5");
    question "q05" (joined "big" "id3");
  ]

let join ~data size =
  let n = rows size in
  let file part =
    Filename.concat (directory data size)
      (Printf.sprintf "J1_%s_%s_0_0.csv" size part)
  in
  let parts =
    [
      ("x", "NA");
      ("small", pretty (n / 1_000_000));
      ("medium", pretty (n / 1_000));
      ("big", size);
    ]
  in
  {
    Workload.id = "join/" ^ size;
    tables =
      List.map
        (fun (name, part) ->
          (name, Workload.Csv (file part, List.assoc name join_schemas)))
        parts;
    questions = join_questions;
    ordered = false;
  }
