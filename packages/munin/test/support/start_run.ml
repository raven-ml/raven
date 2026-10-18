(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [start_run root] waits for the end of its standard input, then starts and
   finishes one run of experiment "sweep" in the store at [root] and prints its
   id. Tests start several at once to change one store from several
   processes. *)

open Munin

let provenance =
  {
    Provenance.command = [ "start_run" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    git_commit = None;
    git_dirty = None;
    env = [];
  }

let () =
  let root = Sys.argv.(1) in
  ignore (In_channel.input_all stdin);
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~provenance ~experiment:"sweep" () in
  Session.finish session;
  print_string (Session.id session)
