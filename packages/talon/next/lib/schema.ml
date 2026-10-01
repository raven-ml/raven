(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Smap = Map.Make (String)

type t = { columns : (string * Type.any) list; index : Type.any Smap.t }

let make ~by columns =
  let add index (name, t) =
    if not (String.is_valid_utf_8 name) then
      invalid_arg (Printf.sprintf "%s: column name %S is not UTF-8" by name);
    if Smap.mem name index then
      invalid_arg (Printf.sprintf "%s: duplicate column %S" by name);
    Smap.add name t index
  in
  { columns; index = List.fold_left add Smap.empty columns }

let v columns = make ~by:"Schema.v" columns
let columns s = s.columns
let names s = List.map fst s.columns
let find s name = Smap.find_opt name s.index

let equal_column (n0, Type.Any t0) (n1, Type.Any t1) =
  String.equal n0 n1 && Type.equal t0 t1

let equal s0 s1 = List.equal equal_column s0.columns s1.columns

type change =
  | Added of string * Type.any
  | Removed of string * Type.any
  | Retyped of string * Type.any * Type.any

let diff s0 s1 =
  let before (name, (Type.Any t0 as a0)) =
    match find s1 name with
    | None -> Some (Removed (name, a0))
    | Some (Type.Any t1) when Type.equal t0 t1 -> None
    | Some a1 -> Some (Retyped (name, a0, a1))
  in
  let after (name, a1) =
    if Smap.mem name s0.index then None else Some (Added (name, a1))
  in
  List.filter_map before s0.columns @ List.filter_map after s1.columns

let pp ppf s =
  let column ppf (name, Type.Any t) =
    Format.fprintf ppf "%a %a" Type.pp_name name Type.pp t
  in
  let pp_sep ppf () = Format.pp_print_string ppf ", " in
  Format.pp_print_list ~pp_sep column ppf s.columns
