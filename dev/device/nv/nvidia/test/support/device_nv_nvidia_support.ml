(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

external lock : string -> bool = "device_nv_nvidia_test_lock"
external files : unit -> int = "device_nv_nvidia_test_files"
external limit : unit -> int = "device_nv_nvidia_test_limit"
external set_limit : int -> unit = "device_nv_nvidia_test_set_limit"
external limit_for : int -> int = "device_nv_nvidia_test_limit_for"

let gpu_lock = "DEVICE_NV_TEST_GPU_LOCK"

(* The lock is taken once and kept: [Some true] once taken. *)
let held = ref None

let hold_gpu () =
  if Device_nv_nvidia.count () = 0 then
    skip ~reason:"the machine has no NVIDIA GPU" ();
  let taken =
    match !held with
    | Some taken -> taken
    | None ->
        let taken =
          match Sys.getenv_opt gpu_lock with
          | None | Some "" -> skip ~reason:(gpu_lock ^ " names no lock file") ()
          | Some file -> lock file
        in
        held := Some taken;
        taken
  in
  if not taken then skip ~reason:"another process holds the GPU lock" ()

let with_limit n f =
  let soft = limit () in
  set_limit n;
  Fun.protect ~finally:(fun () -> set_limit soft) f
