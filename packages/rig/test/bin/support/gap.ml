(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_bin

(* [gap.exe chld|term] sends a signal rig handles just before [Proc.wait]
   blocks, with nothing else to wake it: [chld] ends a child, [term] sends
   SIGTERM. Prints what woke the wait, that it blocked past the signal, or
   that no signal came. *)

external arm : int -> unit = "gap_arm"
external blocked : unit -> bool = "gap_blocked"
external missed : unit -> bool = "gap_missed"

(* The wait's own bound. A wait that misses gap.c's calls, as one through a
   symbol gap.c doesn't define, returns there with no signal sent. *)
let bound_s = 5.

let wait () = Proc.wait ~until:(Unix.gettimeofday () +. bound_s) []

(* [child], if any, is still running when no signal came. *)
let woke ?child what =
  if missed () then begin
    Option.iter Proc.kill child;
    print_endline "no signal: the wait missed gap.c";
    exit 1
  end;
  if blocked () then begin
    print_endline "blocked past the signal";
    exit 1
  end;
  print_endline what

let () =
  match Sys.argv with
  | [| _; "chld" |] ->
      Proc.signals [];
      let pid =
        Proc.spawn "sleep" [| "sleep"; "60" |] ~stdin:Unix.stdin
          ~stdout:Unix.stdout ~stderr:Unix.stderr
      in
      arm pid;
      wait ();
      woke ~child:pid "woke: child ended";
      Option.iter
        (fun st -> print_endline (Proc.cause st))
        (Proc.reap pid)
  | [| _; "term" |] ->
      Proc.signals [ Sys.sigterm ];
      arm 0;
      wait ();
      woke "woke: interrupted";
      if Proc.interrupted () = Some Sys.sigterm then print_endline "SIGTERM"
  | _ ->
      prerr_endline "usage: gap.exe chld|term";
      exit 1
