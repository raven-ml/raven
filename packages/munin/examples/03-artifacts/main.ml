(** Artifact versioning and lineage across runs.

    Run 1 produces a dataset artifact. Run 2 consumes it and produces a result.
    Demonstrates versioning, aliases, and cross-run lineage. *)

open Munin

let write_file path text =
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc text)

let () =
  let root = "_munin" in
  let store = Store.open_ ~root () in

  (* Run 1: produce a dataset. *)
  let session1 =
    Session.start ~root ~experiment:"pipeline" ~name:"prepare-data"
      ~tags:[ "data" ] ()
  in
  let data_path = Filename.concat root "measurements.csv" in
  write_file data_path "wavelength,flux\n450.0,1.23\n550.0,2.45\n650.0,1.87\n";
  let dataset =
    Session.log_artifact session1 ~name:"measurements" ~kind:`dataset
      ~path:data_path
      ~metadata:[ ("rows", `Int 3) ]
      ~aliases:[ "latest" ] ()
  in
  Session.finish session1 ();
  Printf.printf "produced: %s v%s (aliases: %s)\n" (Artifact.name dataset)
    (Artifact.version dataset)
    (String.concat ", " (Artifact.aliases dataset));

  (* Run 2: consume the dataset, produce a result. *)
  let session2 =
    Session.start ~root ~experiment:"pipeline" ~name:"analyse"
      ~tags:[ "analysis" ] ()
  in
  Session.use_artifact session2 dataset;
  let result_path = Filename.concat root "result.txt" in
  write_file result_path "peak_wavelength=550.0\npeak_flux=2.45\n";
  let result =
    Session.log_artifact session2 ~name:"analysis-result" ~kind:`file
      ~path:result_path ~aliases:[ "latest"; "best" ] ()
  in
  Session.finish session2 ();
  Printf.printf "produced: %s v%s\n" (Artifact.name result)
    (Artifact.version result);

  (* Query artifacts from the store. *)
  (match Store.find_artifact store ~name:"measurements" ~version:"latest" with
  | Some a ->
      Printf.printf "\nresolved 'measurements:latest' -> v%s (%d bytes)\n"
        (Artifact.version a) (Artifact.size_bytes a);
      Printf.printf "  producer: %s\n"
        (Option.value ~default:"unknown" (Artifact.producer_run_id a));
      Printf.printf "  consumers: %s\n"
        (String.concat ", " (Artifact.consumer_run_ids a))
  | None -> Printf.printf "artifact not found\n");

  let all = Store.list_artifacts store () in
  Printf.printf "\nall artifacts: %d\n" (List.length all);
  List.iter
    (fun a -> Printf.printf "  %s v%s\n" (Artifact.name a) (Artifact.version a))
    all
