(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [defaults.exe PROGRAM ARG...] runs PROGRAM with SIGINT, SIGTERM and SIGHUP
   at their default actions, as a terminal starts its foreground job. A
   background job, or a process under nohup, starts with some of them ignored,
   and an ignored signal stays ignored across exec. *)

let () =
  List.iter
    (fun s -> Sys.set_signal s Sys.Signal_default)
    [ Sys.sigint; Sys.sigterm; Sys.sighup ];
  match List.tl (Array.to_list Sys.argv) with
  | [] ->
      prerr_endline "usage: defaults.exe PROGRAM [ARG...]";
      exit 1
  | prog :: _ as args -> Unix.execvp prog (Array.of_list args)
