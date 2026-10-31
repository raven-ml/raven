(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's GPUs, through {!Rig_nv_nvidia}. *)

val read : window:float -> (string * (int * int * float option) option) list
(** [read ~window] is each of the machine's NVIDIA GPUs, by bus address in
    bus order, with its {!Rig_nv_nvidia.usage}[ ~window] as its memory's
    total and free bytes and its load, or [None] where that is an [Error]. *)
