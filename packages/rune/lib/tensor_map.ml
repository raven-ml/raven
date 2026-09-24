(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type key = Key : ('a, 'b) Nx_effect.t -> key

(* Keys are tensors compared by physical identity and hashed by
   [Nx_effect.identity_hash], which a value keeps for its whole life. *)
module Tbl = Hashtbl.Make (struct
  type t = key

  let equal (Key a) (Key b) = Obj.repr a == Obj.repr b
  let hash (Key x) = Nx_effect.identity_hash x
end)

type entry = Entry : ('a, 'b) Nx_core.Dtype.t * ('a, 'b) Nx.t -> entry
type t = entry Tbl.t

let create () = Tbl.create 64

let find (type a b) m (x : (a, b) Nx.t) : (a, b) Nx.t option =
  match Tbl.find_opt m (Key x) with
  | None -> None
  | Some (Entry (dt, v)) -> (
      (* Entries are stored under the key of the tensor whose dtype they record,
         so the witness always matches. *)
      match Nx_core.Dtype.equal_witness dt (Nx.dtype x) with
      | Some Type.Equal -> Some v
      | None -> assert false)

let set m x v = Tbl.replace m (Key x) (Entry (Nx.dtype x, v))

module Ids = struct
  type t = unit Tbl.t

  let create () = Tbl.create 64
  let add ids x = Tbl.replace ids (Key x) ()
  let mem ids x = Tbl.mem ids (Key x)
end
