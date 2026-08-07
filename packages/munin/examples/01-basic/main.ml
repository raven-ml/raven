(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

let () =
  let store = Store.open_ ~root:"_munin" () in
  let artifact_path = Filename.concat (Store.root store) "artifact.txt" in
  let write path text =
    let oc = open_out path in
    Fun.protect
      ~finally:(fun () -> close_out oc)
      (fun () -> output_string oc text)
  in
  let session =
    Session.start ~store ~experiment:"demo" ~name:"baseline"
      ~params:[ ("lr", `Float 0.001) ]
      ()
  in
  write artifact_path "hello from munin\n";
  let loss = Session.metric session "loss" in
  Metric.log loss ~step:1 1.25;
  Metric.log loss ~step:2 0.94;
  Session.set_summary session [ ("best_loss", `Float 0.94) ];
  ignore
    (Session.log_artifact session ~name:"notes" ~kind:`File ~path:artifact_path
       ());
  Session.finish session;
  Printf.printf "run: %s\n" (Session.id session)
