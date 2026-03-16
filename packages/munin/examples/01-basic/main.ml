let () =
  let root = "_munin" in
  let artifact_path = Filename.concat root "artifact.txt" in
  let write path text =
    let oc = open_out path in
    Fun.protect
      ~finally:(fun () -> close_out oc)
      (fun () -> output_string oc text)
  in
  let session =
    Session.start ~root ~experiment:"demo" ~name:"baseline"
      ~params:[ ("lr", `Float 0.001) ]
      ()
  in
  write artifact_path "hello from munin\n";
  Session.log_metric session ~step:1 "loss" 1.25;
  Session.log_metric session ~step:2 "loss" 0.94;
  Session.set_summary session [ ("best_loss", `Float 0.94) ];
  ignore
    (Session.log_artifact session ~name:"notes" ~kind:`file ~path:artifact_path
       ());
  Session.finish session ();
  Printf.printf "run: %s\n" (Run.id (Session.run session))
