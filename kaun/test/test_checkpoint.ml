module A = Alcotest
module Ptree = Kaun.Ptree

let test_save_and_load () =
  let dtype = Rune.float32 in

  (* Create some test parameters *)
  let w = Rune.ones dtype [| 3; 3 |] in
  let b = Rune.zeros dtype [| 3 |] in
  let params =
    Ptree.dict [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ]
  in

  (* Save checkpoint to specific file *)
  let path = "/tmp/test_checkpoint.safetensors" in
  (* Use the lower-level save_params_file for a single snapshot file *)
  let module C = Kaun.Checkpoint in
  (match C.save_params_file ~path ~params with
  | Ok () -> ()
  | Error err -> A.failf "save_params_file failed: %s" (C.error_to_string err));

  (* Load checkpoint from file *)
  let loaded_params =
    match C.load_params_file ~path with
    | Ok params -> params
    | Error err -> A.failf "failed to read params: %s" (C.error_to_string err)
  in

  let loaded_fields =
    Ptree.Dict.fields_exn ~ctx:"loaded params" loaded_params
  in
  let loaded_w = Ptree.Dict.get_tensor_exn loaded_fields ~name:"weight" dtype in
  let loaded_b = Ptree.Dict.get_tensor_exn loaded_fields ~name:"bias" dtype in
  let weight_equal = Rune.all (Rune.equal w loaded_w) |> Rune.to_array in
  A.check A.bool "weights match" true weight_equal.(0);
  let bias_equal = Rune.all (Rune.equal b loaded_b) |> Rune.to_array in
  A.check A.bool "bias matches" true bias_equal.(0)

let test_checkpoint_manager () =
  let dtype = Rune.float32 in

  (* Create test parameters *)
  let create_params value =
    let w = Rune.full dtype [| 2; 2 |] value in
    Ptree.dict [ ("weight", Ptree.tensor w) ]
  in

  (* Create repository *)
  let dir = "/tmp/test_checkpoint_manager" in
  let _ = Sys.command (Printf.sprintf "rm -rf %s" dir) in

  let module C = Kaun.Checkpoint in
  let retention : C.retention = { max_to_keep = Some 2; keep_every = None } in
  let repository = C.create_repository ~directory:dir ~retention () in

  (* Save multiple checkpoints *)
  for i = 1 to 5 do
    let params = create_params (float_of_int i) in
    let snapshot = C.Snapshot.of_ptree params in
    let artifacts =
      [ C.artifact ~label:"params" ~kind:C.Params ~snapshot () ]
    in
    match C.write repository ~step:(i * 10) ~artifacts with
    | Ok _ -> ()
    | Error err ->
        A.failf "write failed for step %d: %s" (i * 10) (C.error_to_string err)
  done;

  (* Check that we have at least one checkpoint *)
  let steps = C.steps repository in
  A.check A.bool "has checkpoints" true (List.length steps >= 1);
  (* Retention should keep only the latest two checkpoints *)
  A.check (A.list A.int) "retained steps" [ 40; 50 ] steps;

  (* Check latest step *)
  let latest = C.latest_step repository in
  A.check (A.option A.int) "latest step" (Some 50) latest;

  (* Restore latest *)
  let restored_params, restored_step =
    match C.read_latest repository with
    | Error err -> A.failf "read_latest failed: %s" (C.error_to_string err)
    | Ok (manifest, artifacts) ->
        let step =
          match manifest.step with
          | Some value -> value
          | None -> A.fail "manifest missing step"
        in
        let params =
          match
            List.find_map
              (fun artifact ->
                if C.artifact_kind artifact = C.Params then
                  match C.Snapshot.to_ptree (C.artifact_snapshot artifact) with
                  | Ok ptree -> Some ptree
                  | Error msg -> A.failf "to_ptree failed: %s" msg
                else None)
              artifacts
          with
          | Some params -> params
          | None -> A.fail "missing params artifact"
        in
        (params, step)
  in
  A.check A.int "restored step" 50 restored_step;

  (* Check restored value *)
  let restored_fields =
    Ptree.Dict.fields_exn ~ctx:"restored params" restored_params
  in
  let restored_w =
    Ptree.Dict.get_tensor_exn restored_fields ~name:"weight" dtype
  in
  let value = Rune.item [ 0; 0 ] restored_w in
  A.check (A.float 0.01) "restored value" 5.0 value

let test_sequential_roundtrip () =
  let dtype = Rune.float32 in
  let rng = Rune.Rng.key 42 in
  let obs_dim = 4 in
  let n_actions = 2 in
  let network =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:obs_dim ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:n_actions ();
      ]
  in
  let params = network.init ~rngs:rng ~dtype in
  let input = Rune.create dtype [| obs_dim |] [| 0.5; 0.5; 0.5; 0.5 |] in
  let output_before = network.apply params ~training:false input in
  let tmp_dir = Filename.get_temp_dir_name () in
  let path = Filename.temp_file ~temp_dir:tmp_dir "kaun_seq" ".safetensors" in
  let module C = Kaun.Checkpoint in
  (match C.save_params_file ~path ~params with
  | Ok () -> ()
  | Error err -> A.failf "save_params_file failed: %s" (C.error_to_string err));
  let loaded_params =
    match C.load_params_file ~path with
    | Ok params -> params
    | Error err -> A.failf "failed to read params: %s" (C.error_to_string err)
  in
  let flattened = Kaun.Ptree.flatten_with_paths loaded_params in
  let expect_path key =
    if
      not
        (List.exists
           (fun (path, _) -> String.equal (Kaun.Ptree.Path.to_string path) key)
           flattened)
    then A.failf "missing checkpoint tensor: %s" key
  in
  List.iter expect_path [ "[0].weight"; "[0].bias"; "[2].weight"; "[2].bias" ];
  let output_after = network.apply loaded_params ~training:false input in
  let before = Rune.to_array output_before in
  let after = Rune.to_array output_after in
  let same =
    Array.for_all2 (fun a b -> Float.abs (a -. b) < 1e-6) before after
  in
  A.check A.bool "sequential roundtrip produces identical output" true same;
  try Sys.remove path with Sys_error _ -> ()

let () =
  A.run "Kaun.Checkpoint"
    [
      ( "basic",
        [
          A.test_case "save_and_load" `Quick test_save_and_load;
          A.test_case "checkpoint_manager" `Quick test_checkpoint_manager;
          A.test_case "sequential_roundtrip" `Quick test_sequential_roundtrip;
        ] );
    ]
