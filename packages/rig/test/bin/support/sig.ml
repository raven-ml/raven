(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [sig.exe ACTION SIGNALS PROG ARG...] runs PROG with SIGNALS, a
   comma-separated list such as INT,TERM,HUP, set to ACTION, as a parent that
   set them starts its children: dispositions and the mask survive exec.

   default: a terminal starts its foreground job so. A background job, or a
   process under nohup, starts with some of them ignored.
   ignore: some supervisors ignore CHLD, which a shell's trap cannot do: it
   keeps SIGCHLD for itself. With XFSZ ignored, a write past the file size
   limit fails instead of killing the process.
   block: some supervisors block CHLD. *)

let signal = function
  | "INT" -> Sys.sigint
  | "TERM" -> Sys.sigterm
  | "HUP" -> Sys.sighup
  | "CHLD" -> Sys.sigchld
  | "PIPE" -> Sys.sigpipe
  | "XFSZ" -> Sys.sigxfsz
  | s -> failwith ("unknown signal " ^ s)

let () =
  match List.tl (Array.to_list Sys.argv) with
  | action :: signals :: (prog :: _ as args) ->
      let signals = List.map signal (String.split_on_char ',' signals) in
      (match action with
      | "default" ->
          List.iter (fun s -> Sys.set_signal s Sys.Signal_default) signals
      | "ignore" ->
          List.iter (fun s -> Sys.set_signal s Sys.Signal_ignore) signals
      | "block" -> ignore (Unix.sigprocmask Unix.SIG_BLOCK signals)
      | a -> failwith ("unknown action " ^ a));
      Unix.execvp prog (Array.of_list args)
  | _ ->
      prerr_endline "usage: sig.exe default|ignore|block SIGNALS PROG [ARG...]";
      exit 1
