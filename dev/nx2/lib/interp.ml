(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

let add_formula i id =
  match i.formulas with
  | Some t -> Hashtbl.replace t id ()
  | None ->
      let t = Hashtbl.create 16 in
      Hashtbl.replace t id ();
      i.formulas <- Some t

let is_formula i id =
  match i.formulas with Some t -> Hashtbl.mem t id | None -> false

let traced i form payload = Traced { form; owner = i; payload }

let payload (type v s d) i (x : (v, s, d) t) =
  match x with
  | Traced { owner; payload; _ } when owner == i -> Some payload
  | Traced _ | Array _ | Shards _ | Donated _ | Deferred _ -> None

let owner (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> Some owner
  | Array _ | Shards _ | Donated _ | Deferred _ -> None

let later a b = a.start > b.start
