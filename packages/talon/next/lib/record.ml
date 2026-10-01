(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Kind.record

let kind = Kind.Record
let empty = { Kind.fields = [||] }

let find r name =
  Iarray.find_opt (fun (n, _) -> String.equal n name) r.Kind.fields

let add k name v r =
  if Option.is_some (find r name) then
    invalid_arg (Printf.sprintf "Record.add: duplicate field %S" name);
  if not (String.is_valid_utf_8 name) then
    invalid_arg (Printf.sprintf "Record.add: field name %S is not UTF-8" name);
  if Kind.has_ext k then
    invalid_arg
      (Format.asprintf "Record.add: field %S has the kind %a" name Kind.pp k);
  let field = Iarray.of_list [ (name, Kind.Value (k, v)) ] in
  { Kind.fields = Iarray.append r.Kind.fields field }

let field (type a) (k : a Kind.t) name r : a option =
  let err fmt = Format.kasprintf invalid_arg ("Record.field: " ^^ fmt) in
  match find r name with
  | None -> err "no field %S" name
  | Some (_, Storage _) -> err "field %S has an extension type" name
  | Some (_, Value (fk, v)) -> (
      match Kind.equal_witness fk k with
      | Some Equal -> v
      | None -> err "field %S is %a, not %a" name Kind.pp fk Kind.pp k)

let names r = Iarray.to_list (Iarray.map fst r.Kind.fields)
