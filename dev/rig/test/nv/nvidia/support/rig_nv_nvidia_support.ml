(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The machine's GPU lock *)

let hold_gpu () = if Rig_nv_nvidia.count () > 0 then Rig_gpu_lock.hold ()

(* The process's files *)

external files : unit -> int = "rig_nv_nvidia_test_files"
external limit : unit -> int = "rig_nv_nvidia_test_limit"
external set_limit : int -> unit = "rig_nv_nvidia_test_set_limit"
external limit_for : int -> int = "rig_nv_nvidia_test_limit_for"

let with_limit n f =
  let soft = limit () in
  set_limit n;
  Fun.protect ~finally:(fun () -> set_limit soft) f

(* The process's addresses *)

external occupy : int -> int -> bool = "rig_nv_nvidia_test_occupy"
external vacate : int -> int -> unit = "rig_nv_nvidia_test_vacate"
