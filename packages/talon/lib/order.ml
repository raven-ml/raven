(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { name : string; desc : bool; nulls_first : bool }

let asc name = { name; desc = false; nulls_first = false }
let desc name = { name; desc = true; nulls_first = false }
let nulls_first k = { k with nulls_first = true }

let check ks s =
  let problem seen k =
    if List.mem k.name seen then
      Some (Problem.v "the column %a is already a key." Type.pp_quoted k.name)
    else
      match Schema.find s k.name with
      | None -> Some (Problem.missing k.name s)
      | Some (Any (Ext _ as ty)) ->
          Some
            (Problem.v "%a is %a, which orders only through its declaration."
               Type.pp_quoted k.name Type.pp ty)
      | Some (Any ty) when Type.has_ext ty ->
          Some
            (Problem.v
               "%a is %a, which holds an extension type and has no order."
               Type.pp_quoted k.name Type.pp ty)
      | Some _ -> None
  in
  let add (seen, ps) k =
    let ps = match problem seen k with Some p -> (k, p) :: ps | None -> ps in
    (k.name :: seen, ps)
  in
  List.rev (snd (List.fold_left add ([], []) ks))

let equal k0 k1 =
  String.equal k0.name k1.name
  && Bool.equal k0.desc k1.desc
  && Bool.equal k0.nulls_first k1.nulls_first

let pp ppf k =
  let dir = if k.desc then "desc" else "asc" in
  if k.nulls_first then
    Format.fprintf ppf "nulls_first (%s %a)" dir Type.pp_quoted k.name
  else Format.fprintf ppf "%s %a" dir Type.pp_quoted k.name
