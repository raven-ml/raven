(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The npz file a command writes: [rows], each object's row in each source, and
   the command's entries, each allocated once for the objects and filled batch
   by batch. *)

(* An entry, the objects along its first dimension. *)
type ('a, 'b) t = { mutable x : ('a, 'b, Nx.host) Nx.t }

let v dt shape = { x = Nx.zeros dt shape }

(* [e] with [b], the results of the objects from [first], at [first] and then
   [at]. *)
let set e first ?(at = []) b =
  let objs = Nx.R (first, first + Nx.dim 0 b) in
  e.x <- Nx.set (objs :: at) b (Nx.donate e.x)

let entry name e = (name, Nx.P e.x)

let save file (o : Objects.t) entries =
  let rows =
    Nx.create Nx.int64
      [| Objects.length o; List.length o.sources |]
      (Array.concat (Array.to_list (Array.map (Array.map Int64.of_int) o.rows)))
  in
  Nx_io.save_npz file (Nx_io.Archive.of_list (("rows", Nx.P rows) :: entries))
