(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* No NVIDIA GPU is read off Linux. *)
let read ~window:(_ : float) : (string * (int * int * float option) option) list
    =
  []
