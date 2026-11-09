(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A child forked before the process opened Metal prepares a function and
   enters it, and Metal answers there, as in a bench's worker. A forked
   child cannot reach Metal's compiler service, so the pipeline must be in
   Metal's cache: a fresh process, run as [--warm], compiles it first,
   whatever the cache held. Two children, each its own open. Its own suite:
   a process that ran a domain cannot fork. *)

open Windtrap
module S = Rig_metal_support

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> "killed by " ^ Sys.signal_to_string n
  | Unix.WSTOPPED n -> "stopped by " ^ Sys.signal_to_string n

(* Runs [child] in a forked child and is what it returned: the child writes its
   answer to a pipe and leaves without running this process's exit. *)
let in_child child =
  let r, w = Unix.pipe () in
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let lines = try child () with e -> [ Printexc.to_string e ] in
      let oc = Unix.out_channel_of_descr w in
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let lines = In_channel.input_lines ic in
      close_in ic;
      let _, status = Unix.waitpid [] pid in
      (lines, status)

let prepared_entry () =
  let { S.g; _ } = S.open_ () in
  match Rig_metal.image g (S.fixture ~dir:"fixtures" "threadgroup") with
  | Error why -> [ why ]
  | Ok (Rig_edge.Place _) -> [ "Metal asked to place its code" ]
  | Ok (Loaded i) -> (
      Rig_metal.prepare i "small";
      match Rig_metal.entry i "small" with
      | Entry _ -> [ "entered" ]
      | Absent -> [ "absent" ]
      | No_memory n -> [ Printf.sprintf "no memory for %d bytes" n ])

(* [prepared_entry] in a fresh process, which reaches Metal's compiler. *)
let warm () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--warm" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  equal string ~msg:"the warm process" "exited 0"
    (status (snd (Unix.waitpid [] pid)))

let forked () =
  if not (S.present ()) then
    skip ~reason:"the machine has no METAL GPU rig drives" ();
  warm ();
  for k = 1 to 2 do
    let lines, st = in_child prepared_entry in
    let msg = Printf.sprintf "child %d" k in
    equal ~msg (list string) [ "entered" ] lines;
    equal ~msg string "exited 0" (status st)
  done

let () =
  (match Sys.argv with
  | [| _; "--warm" |] ->
      exit (if prepared_entry () = [ "entered" ] then 0 else 1)
  | _ -> ());
  exit
    (run "rig_metal_fork"
       [
         test ~timeout:60.
           "a child forked before Metal opened prepares and enters a function"
           forked;
       ])
