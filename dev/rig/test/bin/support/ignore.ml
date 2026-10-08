(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [ignore.exe SIGNAL PROG ARG...] runs PROG with SIGNAL (CHLD or PIPE) ignored,
   as a parent that ignores it starts its children. A shell's trap cannot: it
   keeps SIGCHLD for itself. *)

let () =
  match List.tl (Array.to_list Sys.argv) with
  | s :: prog :: args ->
      let s = match s with "CHLD" -> Sys.sigchld | _ -> Sys.sigpipe in
      Sys.set_signal s Sys.Signal_ignore;
      Unix.execvp prog (Array.of_list (prog :: args))
  | _ ->
      prerr_endline "usage: ignore.exe SIGNAL PROG [ARG...]";
      exit 1
