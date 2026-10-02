(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

type table = Csv of string * (string * Type.any) list | Parquet of string
type question = { name : string; query : (string -> Query.t) -> Query.t }

type t = {
  id : string;
  tables : (string * table) list;
  questions : question list;
  ordered : bool;
}

let file = function Csv (file, _) | Parquet file -> file

let query = function
  | Csv (file, columns) ->
      let format = Talon_csv.format columns in
      let source = Error.get_ok (Talon_csv.file ~format file) in
      Query.of_table (Error.get_ok (Query.run (Query.of_source source)))
  | Parquet file -> Query.of_source (Error.get_ok (Talon_parquet.file file))

let missing w =
  List.filter
    (fun f -> not (Sys.file_exists f))
    (List.map (fun (_, t) -> file t) w.tables)

let load w =
  let missing = missing w in
  if missing <> [] then
    failwith
      (Printf.sprintf "%s: missing data files (generate them first):\n  %s" w.id
         (String.concat "\n  " missing));
  let tables = List.map (fun (name, t) -> (name, query t)) w.tables in
  fun name ->
    match List.assoc_opt name tables with
    | Some q -> q
    | None -> invalid_arg (Printf.sprintf "%s: no table %S" w.id name)
