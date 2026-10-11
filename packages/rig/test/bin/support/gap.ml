(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_bin

(* [gap.exe chld|term] sends a signal rig handles just before [Proc.wait]
   blocks, with nothing else to wake it: [chld] ends a child, [term] sends
   SIGTERM. Prints what woke the wait, or that it blocked past the signal. *)

external arm : int -> unit = "gap_arm"
external blocked : unit -> bool = "gap_blocked"

let woke what =
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
      Proc.wait [];
      woke "woke: child ended";
      Option.iter
        (fun st -> print_endline (Proc.cause st))
        (Proc.reap pid)
  | [| _; "term" |] ->
      Proc.signals [ Sys.sigterm ];
      arm 0;
      Proc.wait [];
      woke "woke: interrupted";
      if Proc.interrupted () = Some Sys.sigterm then print_endline "SIGTERM"
  | _ ->
      prerr_endline "usage: gap.exe chld|term";
      exit 1
