(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The firmware a GPU boots with: the images its blocks' versions name, each
   read by the digest this library pins, cut into the pieces its security
   processor loads. Pure: any domain. *)

type t = {
  sos : (int * string) list;
      (* the security processor's own components, by firmware type *)
  smu : (int list * string) option;
      (* the power manager's image, loaded before the trusted memory region *)
  pieces : (int list * string) list;
      (* the others, in load order, each with the firmware types it loads as *)
  starts : (string * int) list;
      (* the start address of each RS64 engine ("PFP", "ME", "MEC") *)
  mec : int; (* the version of the compute queues' firmware *)
}

(* [pinned] is Device_amd_pci.pinned: every image's path, BLAKE2b-256 digest and
   URL in linux-firmware at the commit the generator names. *)
val pinned : (string * string * string) list

(* [load find d] reads each image the blocks of [d] name with [find name
   ~digest], [name] such as "amdgpu/psp_13_0_0_sos.bin". [Error msg] if a
   block's version names no pinned image, naming it, with [find]'s message, or
   if an image's header is of a version this library does not read, or points
   outside the image, naming the image. *)
val load :
  (string -> digest:string -> (string, string) result) ->
  Discovery.t ->
  (t, string) result
