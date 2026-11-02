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

   [publish ROOT NAME N FILE] waits for the end of its standard input, then
   publishes N versions of NAME from FILE.

   [resume ROOT ID STEP] resumes the run ID at STEP and finishes.

   [take ROOT ID] resumes the run ID at step 0, prints "taken", waits for the
   end of its standard input and exits without finishing the run.

   [publishing ROOT] holds the store's shared lock, as a publication does, with
   a [.]-named entry under blobs/, prints "held", waits for the end of its
   standard input, and prints whether the entry is still there.

   [file ROOT FILE] starts a run, prints its id, logs FILE at step 5 and exits
   without finishing the run.

   [logfile ROOT FILE] starts a run, prints its id, logs FILE under the same
   limit as [fill], and prints what the call raised.

   [fill ROOT FILE] starts a run and publishes FILE, under a limit on file sizes
   that its parent set, and prints the exception the publication raised. *)

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
  | "publish" ->
      let name = Sys.argv.(3) and n = int_of_string Sys.argv.(4) in
      let s = Session.start ~store ~provenance ~experiment:"publisher" () in
      ignore (In_channel.input_all stdin);
      for _ = 1 to n do
        ignore (Session.log_artifact s ~name Sys.argv.(5))
      done;
      Session.finish s
  | "resume" ->
      let run = Option.get (Store.run store Sys.argv.(3)) in
      let s =
        Session.resume ~provenance run ~step:(int_of_string Sys.argv.(4))
      in
      Session.finish s
  | "take" ->
      let run = Option.get (Store.run store Sys.argv.(3)) in
      ignore (Session.resume ~provenance run ~step:0);
      print_endline "taken";
      ignore (In_channel.input_all stdin)
  | "publishing" ->
      Store.with_publication_lock store (fun () ->
          let blobs = Filename.concat (Store.root store) "blobs" in
          let dir = Filename.concat blobs "blake2b-256" in
          let entry = Filename.concat dir ".in-progress" in
          List.iter
            (fun d ->
              try Unix.mkdir d 0o755
              with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
            [ blobs; dir ];
          Out_channel.with_open_bin entry (fun oc -> output_string oc "x");
          print_endline "held";
          ignore (In_channel.input_all stdin);
          print_endline (if Sys.file_exists entry then "intact" else "removed");
          Sys.remove entry)
  | "file" ->
      let s = Session.start ~store ~provenance ~experiment:"files" () in
      print_endline (Session.id s);
      Session.log_file s ~step:5 ~key:"viz" Sys.argv.(3)
  | "logfile" -> (
      Sys.set_signal Sys.sigxfsz Sys.Signal_ignore;
      let s = Session.start ~store ~provenance ~experiment:"logfile" () in
      print_endline (Session.id s);
      match Session.log_file s ~step:0 ~key:"big" Sys.argv.(3) with
      | () -> print_endline "logged"
      | exception Sys_error _ -> print_endline "Sys_error")
  | "fill" -> (
      (* The limit sends SIGXFSZ, which would kill the child before the write
         can fail. *)
      Sys.set_signal Sys.sigxfsz Sys.Signal_ignore;
      let s = Session.start ~store ~provenance ~experiment:"fill" () in
      match Session.log_artifact s ~name:"big" Sys.argv.(3) with
      | _ -> print_endline "published"
      | exception Sys_error _ -> print_endline "Sys_error"
      | exception Unix.Unix_error _ -> print_endline "Unix_error")
  | m -> failwith ("child: unknown mode " ^ m)
