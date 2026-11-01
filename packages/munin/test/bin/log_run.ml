(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [log_run root] starts a run in the store at [root] with params "lr" and
   "seed", logs [samples] of a metric "loss" at steps 0 to 4, finishes it and
   prints its id. *)

open Munin

let provenance =
  {
    Provenance.command = [ "log_run" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    git_commit = None;
    git_dirty = None;
    env = [];
  }

(* Values and their timestamps: the first an epoch whose microseconds round
   down, 1709385958.677423 for 1709385958.67742347... *)
let samples =
  [
    (1e6, 1709385958.6774235);
    (4.58e-72, 2.);
    (0.1, 3.);
    (Float.infinity, 4.);
    (Float.nan, 5.);
  ]

let () =
  let store = Store.open_ ~root:Sys.argv.(1) () in
  let session =
    Session.start ~store ~provenance ~experiment:"exp"
      ~params:[ ("lr", `Float 0.5); ("seed", `Int 7) ]
      ()
  in
  let loss = Session.metric session "loss" in
  List.iteri
    (fun step (x, timestamp) ->
      Session.log_metrics session ~step ~timestamp [ (loss, x) ])
    samples;
  Session.finish session;
  print_string (Session.id session)
