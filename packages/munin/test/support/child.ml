(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A second process for the tests, [child MODE ROOT ARGS...]:

   [hold ROOT] starts a run, prints its id, waits for the end of its standard
   input and exits without finishing the run.

   [log ROOT N] starts a run, prints its id, logs the samples 0 to N - 1 of
   "big", each beside a key 70 KiB long so that a record spans several writes,
   and finishes.

   [take ROOT ID] resumes the run ID at step 0, prints "taken", waits for the
   end of its standard input and exits without finishing the run.

   [gc ROOT] collects the store and prints the number of entries it removed.

   [notes ROOT ID TEXT] sets the notes of the run ID to TEXT.

   [tags ROOT ID TAG] adds TAG to the run ID.

   [starting ROOT] starts a run of the experiment "starting", prints its id and
   finishes it.

   [logging ROOT FILE] starts a run, prints its id, logs FILE at step 1 under
   "viz" once a line reaches its standard input, and finishes.

   [finishing ROOT] starts a run, prints its id, and finishes it once a line
   reaches its standard input.

   [fifo ROOT FILE] opens the FIFO FILE for writing, which returns once a reader
   has opened it, prints "opened", and writes a version's tombstone into it once
   a line reaches its standard input. An alarm ends it after 10 s, so a reader
   that never opens FILE does not hold the test.

   The modes below run under a limit on file sizes that the parent set, with
   SIGXFSZ ignored, which would kill the child before the write can fail.

   [refused ROOT] starts a run, prints its id and its log's length, sets a
   summary longer than the limit, and prints what the call raised.

   [logfile ROOT FILE] starts a run, prints its id, logs FILE and prints what
   the call raised.

   [fill ROOT FILE] starts a run, publishes FILE and prints what the publication
   raised. *)

open Munin

let provenance =
  {
    Provenance.command = [ "child" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    ocaml = Sys.ocaml_version;
    raven = "test";
    os = "test";
    arch = "test";
    cpu = None;
    git = None;
    seeds = [];
    env = [];
  }

let () =
  (* Windows ends a line with CR LF on a text channel, and the tests read the
     lines from a pipe byte for byte. *)
  set_binary_mode_out stdout true;
  let store = Store.open_ ~root:Sys.argv.(2) () in
  match Sys.argv.(1) with
  | "hold" ->
      let s = Session.start ~store ~provenance ~experiment:"held" () in
      print_endline (Session.id s);
      ignore (In_channel.input_all stdin)
  | "log" ->
      let n = int_of_string Sys.argv.(3) in
      let s = Session.start ~store ~provenance ~experiment:"logged" () in
      print_endline (Session.id s);
      let big = Session.metric s "big" in
      let wide = Session.metric s ("w/" ^ String.make (70 * 1024) 'k') in
      for step = 0 to n - 1 do
        Session.log_metrics s ~step [ (big, Float.of_int step); (wide, 0.) ]
      done;
      Session.finish s
  | "take" ->
      let run = Option.get (Store.run store Sys.argv.(3)) in
      ignore (Session.resume ~provenance run ~step:0);
      print_endline "taken";
      ignore (In_channel.input_all stdin)
  | "fifo" ->
      ignore (Unix.alarm 10);
      let fd = Unix.openfile Sys.argv.(3) [ Unix.O_WRONLY; Unix.O_CLOEXEC ] 0 in
      print_endline "opened";
      ignore (In_channel.input_line stdin);
      let tombstone = {|{"format":1,"time":1.0,"deleted":true}|} ^ "\n" in
      ignore (Unix.write_substring fd tombstone 0 (String.length tombstone));
      Unix.close fd
  | "tags" -> Store.add_tags store Sys.argv.(3) [ Sys.argv.(4) ]
  | "starting" ->
      let s = Session.start ~store ~provenance ~experiment:"starting" () in
      print_endline (Session.id s);
      Session.finish s
  | "notes" -> Store.set_notes store Sys.argv.(3) (Some Sys.argv.(4))
  | "logging" ->
      let s = Session.start ~store ~provenance ~experiment:"logging" () in
      print_endline (Session.id s);
      ignore (In_channel.input_line stdin);
      Session.log_file s ~step:1 ~key:"viz" Sys.argv.(3);
      Session.finish s
  | "finishing" ->
      let s = Session.start ~store ~provenance ~experiment:"finishing" () in
      print_endline (Session.id s);
      ignore (In_channel.input_line stdin);
      Session.finish s
  | "gc" -> print_endline (string_of_int (Store.gc store))
  | "refused" -> (
      Sys.set_signal Sys.sigxfsz Sys.Signal_ignore;
      let s = Session.start ~store ~provenance ~experiment:"refused" () in
      print_endline (Session.id s);
      let run = Option.get (Store.run store (Session.id s)) in
      let log = Filename.concat (Run.dir run) "events.jsonl" in
      print_endline (string_of_int (Unix.stat log).Unix.st_size);
      match Session.set_summary s [ ("k", `String (String.make 4096 'x')) ] with
      | () -> print_endline "set"
      | exception Sys_error _ -> print_endline "Sys_error")
  | "logfile" -> (
      Sys.set_signal Sys.sigxfsz Sys.Signal_ignore;
      let s = Session.start ~store ~provenance ~experiment:"logfile" () in
      print_endline (Session.id s);
      match Session.log_file s ~step:0 ~key:"big" Sys.argv.(3) with
      | () -> print_endline "logged"
      | exception Sys_error _ -> print_endline "Sys_error")
  | "fill" -> (
      Sys.set_signal Sys.sigxfsz Sys.Signal_ignore;
      let s = Session.start ~store ~provenance ~experiment:"fill" () in
      match Session.log_artifact s ~name:"big" Sys.argv.(3) with
      | _ -> print_endline "published"
      | exception Sys_error _ -> print_endline "Sys_error"
      | exception Unix.Unix_error _ -> print_endline "Unix_error")
  | m -> failwith ("child: unknown mode " ^ m)
