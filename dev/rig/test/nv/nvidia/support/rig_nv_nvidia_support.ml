(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* The machine's GPU lock *)

external lock : string -> string -> int = "rig_nv_nvidia_test_lock"

let gpu_lock = "/tmp/raven-rig-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. *)
let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

(* Whether the process that started this one holds the lock for it. *)
let held_outside () = Sys.getenv_opt "RIG_GPU_LOCK_HELD" <> None

let hold_gpu () =
  if (not (held_outside ())) && Rig_nv_nvidia.count () > 0 then take 0

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
