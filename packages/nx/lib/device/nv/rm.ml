(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The resource manager: the GPU's objects, allocated under a client and
   controlled by commands, through the kernel driver or the GSP. *)

module D = Nv_defs

type t = {
  root : int; (* the client *)
  alloc : parent:int -> int -> Params.t option -> int;
      (* [alloc ~parent cls params] is a new object of class [cls] *)
  control : int -> int -> Params.t option -> unit;
      (* [control obj cmd params] runs [cmd] on [obj], in place on [params] *)
  free : parent:int -> int -> unit;
}

let status_name (module R : D.RELEASE) s =
  match List.assoc_opt s R.statuses with
  | Some n -> n
  | None -> Printf.sprintf "0x%x" s

(* Raises [Failure] for the status [s] of [what] if it is not [NV_OK]. *)
let check release what s =
  if s <> D.nv_ok then
    failwith (Printf.sprintf "%s: %s" what (status_name release s))
