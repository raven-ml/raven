(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ephemeron-keyed store of batched forward-mode tangents. See tangent_store.mli
   for the memory contract: each binding lives exactly as long as its primal. *)

module Tbl = Ephemeron.K1.Make (struct
  type t = Obj.t

  let equal = ( == )
  let hash = Hashtbl.hash
end)

(* Entries record the key's dtype so lookups recover the static type through a
   dtype witness, as in Tensor_map: an entry is only ever stored under the key
   of the tensor whose dtype it records, so the witness check cannot fail. *)
type entry = Entry : ('a, 'b) Nx_core.Dtype.t * ('a, 'b) Nx.t -> entry
type t = { lane_count : int; tbl : entry Tbl.t }

let create ~k = { lane_count = k; tbl = Tbl.create 64 }
let k t = t.lane_count

(* The physical shape without performing [E_view] and without forcing a deferred
   tensor: its record carries the shape the fill thunk will produce, so the lens
   reads it straight off the constructor. *)
let shape_of (type a b) (x : (a, b) Nx.t) : int array =
  match x with
  | Nx_effect.T t -> Nx_core.View.shape (Nx_backend.view t)
  | Nx_effect.Deferred d -> Nx_core.View.shape d.Nx_effect.d_view

let shape_string s =
  String.concat "," (Array.to_list (Array.map string_of_int s))

let err_tangent_shape k x v =
  invalid_arg
    (Printf.sprintf
       "Rune.jvp_k: a tangent of shape [%s] was stored for a tensor of shape \
        [%s] under %d lanes (expected [%s]): a lane-stacked tangent can only \
        be combined by operations that keep the lane axis first, so an \
        enclosing transformation has probably batched the computation around \
        the tangent axis — batch dimensions belong inside it (vmap inside \
        jvp_k, not the other way around)"
       (shape_string (shape_of v))
       (shape_string (shape_of x))
       k
       (shape_string (Array.append [| k |] (shape_of x))))

(* Forcing a deferred tensor (an unread jit output) before keying makes the key
   stable — a later force would change its structural hash — and a keyed tensor
   is being differentiated, so its bytes are needed anyway. *)
let stable x = ignore (Nx_effect.unwrap x)

let set t x v =
  stable x;
  let shape_x = shape_of x in
  let expected = Array.append [| t.lane_count |] shape_x in
  if shape_of v <> expected then err_tangent_shape t.lane_count x v;
  Tbl.replace t.tbl (Obj.repr x) (Entry (Nx.dtype x, v))

let find (type a b) t (x : (a, b) Nx.t) : (a, b) Nx.t option =
  match Tbl.find_opt t.tbl (Obj.repr x) with
  | None -> None
  | Some (Entry (dt, v)) -> (
      match Nx_core.Dtype.equal_witness dt (Nx.dtype x) with
      | Some Type.Equal -> Some v
      | None -> assert false)

(* Bindings whose keys died are cleared by the collector; the table cleans its
   own buckets when it resizes, so this stays a walk over the table. *)
let live_entries t = (Tbl.stats_alive t.tbl).Hashtbl.num_bindings
