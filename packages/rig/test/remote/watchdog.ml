(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every suite here runs in under 25 s, within runtest's 2 minutes. *)
let limit = 100.

(* Ends this process with SIGKILL after [limit] seconds, so a suite that hangs
   inside C, where neither a test's timeout nor SIGTERM reaches it, cannot hang
   runtest. The watch is a child process, which the hang cannot stop; it leaves
   within 0.2 s once this process ends. Call it first, before any thread or
   domain. *)
let start () =
  let parent = Unix.getpid () in
  match Unix.fork () with
  | 0 ->
      let deadline = Unix.gettimeofday () +. limit in
      let rec watch () =
        if Unix.getppid () <> parent then Unix._exit 0;
        if Unix.gettimeofday () >= deadline then begin
          prerr_endline "watchdog: the suite ran out of time; killing it";
          (try Unix.kill parent Sys.sigkill with Unix.Unix_error _ -> ());
          Unix._exit 0
        end;
        Unix.sleepf 0.2;
        watch ()
      in
      watch ()
  | _ -> ()
