(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Where a command computes, as tessera's bin reads it: the set of the devices
   --device names, compiled there unless --eager. *)

type 'd t = { at : 'd Nx.Placement.t; jit : bool }

let doc =
  "D the devices, as Rig_gpus.open_all names them: cpu (the default), a GPU \
   (cuda, nv, amd or metal; cuda:1 the second), or a comma list"

let eager_doc =
  " compute eagerly, not compiled by Rune.jit: on the host, whose set has \
   kernels"

(* The set of the devices [names] names. *)
let set names =
  match Rig_gpus.open_all names with
  | Ok ds -> Nx.devices ds
  | Error m -> Printf.ksprintf failwith "--device %s: %s" names m

(* [f], compiled where [d] compiles, [s] its signature. *)
let compile d s f = if d.jit then Rune.jit s f else f
