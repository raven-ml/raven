(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let limit = 200
let rel = 1e-9
let abs = 1e-12
let names t = List.map fst (Schema.columns t)

let canonical ~ordered t =
  let q =
    Query.(
      of_table t
      |> derive
           Expr.
             [
               across Kind.int (Sel.of_kind Kind.int) (fun n x ->
                   n := cast Type.int64 x);
               across Kind.float (Sel.of_kind Kind.float) (fun n x ->
                   n := cast Type.float64 x);
               across Kind.string (Sel.of_kind Kind.string) (fun n x ->
                   n := cast Type.string x);
             ])
  in
  let q =
    if ordered then q
    else Query.sort (List.map Order.asc (names (Query.schema q))) q
  in
  Error.get_ok (Query.run q)

let stride rows = Int.max 1 ((rows + limit - 1) / limit)
let thin s t = if s = 1 then t else take (Nx.arange Nx.int64 0 (rows t) s) t

(* Comparing *)

let close x y =
  x = y
  || (Float.is_nan x && Float.is_nan y)
  || Float.abs (x -. y)
     <= Float.max abs (rel *. Float.max (Float.abs x) (Float.abs y))

let agree same e g =
  match (e, g) with
  | None, None -> true
  | Some x, Some y -> same x y
  | _ -> false

(* [differs ty e g i] is [true] iff the columns [e] and [g] of type [ty]
   disagree at row [i]. *)
let differs (type a) (ty : a Type.t) e g =
  match ty with
  | Type.Float16 | Type.Float32 | Type.Float64 ->
      let e = Column.options Kind.float e and g = Column.options Kind.float g in
      fun i -> not (agree close e.(i) g.(i))
  | _ ->
      let k = Type.kind ty and order = Type.compare_value ty in
      let e = Column.options k e and g = Column.options k g in
      fun i -> not (agree (fun x y -> order x y = 0) e.(i) g.(i))

let text c i =
  match (Column.options Kind.string (Column.print c)).(i) with
  | None -> "null"
  | Some s -> Format.asprintf "%a" Type.pp_quoted s

let column_problem expected got (name, Type.Any ty) =
  let e = column expected name and g = column got name in
  let differs = differs ty e g in
  match List.filter differs (List.init (rows expected) Fun.id) with
  | [] -> None
  | first :: _ as bad ->
      Some
        (Printf.sprintf
           "column %s: %d rows differ, first at row %d: %s, expected %s" name
           (List.length bad) first (text g first) (text e first))

let compare expected got =
  let se = schema expected and sg = schema got in
  if not (Schema.equal se sg) then
    [ Format.asprintf "schema %a, expected %a" Schema.pp sg Schema.pp se ]
  else if rows expected <> rows got then
    [ Printf.sprintf "%d rows, expected %d" (rows got) (rows expected) ]
  else List.filter_map (column_problem expected got) (Schema.columns se)

let check ~expected ~rows:n ~stride got =
  if rows got <> n then [ Printf.sprintf "%d rows, expected %d" (rows got) n ]
  else compare expected (thin stride got)

(* Files *)

let path root id =
  let cut = String.rindex id '/' in
  let workload = String.sub id 0 cut in
  let question = String.sub id (cut + 1) (String.length id - cut - 1) in
  let dir = String.map (function '/' -> '-' | c -> c) workload in
  Filename.concat (Filename.concat root dir) (question ^ ".csv")

let read schema file =
  let format = Talon_csv.format (Schema.columns schema) in
  let source = Error.get_ok (Talon_csv.file ~format file) in
  Error.get_ok (Query.run (Query.of_source source))

let index_schema =
  Schema.v
    [
      ("question", Type.Any Type.string);
      ("rows", Type.Any Type.int64);
      ("stride", Type.Any Type.int64);
    ]

let read_index root =
  let t = read index_schema (Filename.concat root "index.csv") in
  let ids = Column.values Kind.string (column t "question") in
  let rows = Column.values Kind.int (column t "rows") in
  let strides = Column.values Kind.int (column t "stride") in
  List.init (Array.length ids) (fun i -> (ids.(i), (rows.(i), strides.(i))))

let write file t =
  let format = Talon_csv.format (Schema.columns (schema t)) in
  Out_channel.with_open_bin file @@ fun oc ->
  let w = Bytesrw.Bytes.Writer.of_out_channel oc in
  Error.get_ok (Talon_csv.encode ~eod:true format (Query.of_table t) w)
