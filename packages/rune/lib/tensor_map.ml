(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type key = Key : ('a, 'b) Nx.t -> key

let same (type a b c d) (a : (a, b) Nx.t) (b : (c, d) Nx.t) =
  Obj.repr a == Obj.repr b

let fresh out x =
  if same out x then
    invalid_arg "Rune: an Nx operation returned its operand as its result"

(* Keys are tensors compared by physical identity. A placed or traced value
   hashes by its id, which it keeps for its whole life; a host value by its
   structure, which no table sees change, since values are immutable. *)
module Tbl = Hashtbl.Make (struct
  type t = key

  let equal (Key a) (Key b) = same a b

  let hash (Key x) =
    match Nx.Repr.v x with
    | Host _ -> Hashtbl.hash x
    | Placed p -> Nx.Repr.Placed.id p
    | Traced t -> Nx.Repr.Traced.id t
end)

type entry = Entry : ('a, 'b) Nx_dtype.t * ('a, 'b) Nx.t -> entry
type t = entry Tbl.t

let create () = Tbl.create 64

let find (type a b) m (x : (a, b) Nx.t) : (a, b) Nx.t option =
  match Tbl.find_opt m (Key x) with
  | None -> None
  | Some (Entry (dt, v)) -> (
      (* Entries are stored under the key of the tensor whose dtype they record,
         so the witness always matches. *)
      match Nx_dtype.equal_witness dt (Nx.dtype x) with
      | Some Type.Equal -> Some v
      | None -> assert false)

let set m x v = Tbl.replace m (Key x) (Entry (Nx.dtype x, v))

module Ids = struct
  type t = unit Tbl.t

  let create () = Tbl.create 64
  let add ids x = Tbl.replace ids (Key x) ()
  let mem ids x = Tbl.mem ids (Key x)
end
