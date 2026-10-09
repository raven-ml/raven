(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

external lock : string -> string -> int = "rig_gpu_lock_try"

let file = "/tmp/raven-rig-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let wait_s = 300

let holder () =
  match In_channel.with_open_bin file In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. The first refusal prints the
   file's note, so a run that waits says for what. A holder that took the
   lock with the shell's flock writes none, so the note may be an earlier
   holder's. *)
let rec take refused =
  match lock file Sys.executable_name with
  | 0 -> ()
  | -1 when refused < wait_s * 10 ->
      if refused = 0 then
        prerr_endline (strf "%s: waiting; last noted by %s" file (holder ()));
      take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" file wait_s (holder ()))
  | errno -> failwith (strf "%s: errno %d" file errno)

let hold () = if Sys.getenv_opt "RIG_GPU_LOCK_HELD" = None then take 0
