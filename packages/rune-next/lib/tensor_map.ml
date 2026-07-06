(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Keys are tensors compared by physical identity. Structural hashing is
   consistent with physical equality as long as keyed tensors are not mutated
   while the map is live, which the differentiation handlers enforce. *)
module Tbl = Hashtbl.Make (struct
  type t = Obj.t

  let equal = ( == )
  let hash = Hashtbl.hash
end)

type entry = Entry : ('a, 'b) Nx_core.Dtype.t * ('a, 'b) Nx.t -> entry
type t = entry Tbl.t

let create () = Tbl.create 64

let find (type a b) m (x : (a, b) Nx.t) : (a, b) Nx.t option =
  match Tbl.find_opt m (Obj.repr x) with
  | None -> None
  | Some (Entry (dt, v)) -> (
      (* Entries are stored under the key of the tensor whose dtype they record,
         so the witness always matches. *)
      match Nx_core.Dtype.equal_witness dt (Nx.dtype x) with
      | Some Type.Equal -> Some v
      | None -> assert false)

let set m x v = Tbl.replace m (Obj.repr x) (Entry (Nx.dtype x, v))

module Ids = struct
  type t = unit Tbl.t

  let create () = Tbl.create 64
  let add ids x = Tbl.replace ids (Obj.repr x) ()
  let mem ids x = Tbl.mem ids (Obj.repr x)
end
