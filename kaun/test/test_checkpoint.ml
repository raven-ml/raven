module A = Alcotest
module Ptree = Kaun.Ptree

let test_save_and_load () =
  let dtype = Rune.float32 in

  (* Create some test parameters *)
  let w = Rune.ones dtype [| 3; 3 |] in
  let b = Rune.zeros dtype [| 3 |] in
  let params =
    Ptree.record_of [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ]
  in

  (* Save checkpoint to specific file *)
  let path = "/tmp/test_checkpoint.safetensors" in
  (* Use the lower-level save_params for single file *)
  let module C = Kaun.Checkpoint in
  C.save_params ~path ~params ~metadata:[ ("epoch", "10") ] ();

  (* Load checkpoint from file *)
  let loaded_params = C.load_params ~path ~dtype in

  (* Check parameters *)
  (match Ptree.find_in_record "weight" loaded_params with
  | Some weight_tree -> (
      match Ptree.get_tensor weight_tree with
      | Some loaded_w ->
          let is_equal = Rune.all (Rune.equal w loaded_w) in
          let is_equal_val = Rune.to_array is_equal in
          A.check A.bool "weights match" true is_equal_val.(0)
      | None -> A.fail "weight is not a tensor")
  | None -> A.fail "weight not found");

  match Ptree.find_in_record "bias" loaded_params with
  | Some bias_tree -> (
      match Ptree.get_tensor bias_tree with
      | Some loaded_b ->
          let is_equal = Rune.all (Rune.equal b loaded_b) in
          let is_equal_val = Rune.to_array is_equal in
          A.check A.bool "bias matches" true is_equal_val.(0)
      | None -> A.fail "bias is not a tensor")
  | None -> A.fail "bias not found"

let test_checkpoint_manager () =
  let dtype = Rune.float32 in

  (* Create test parameters *)
  let create_params value =
    let w = Rune.full dtype [| 2; 2 |] value in
    Ptree.record_of [ ("weight", Ptree.tensor w) ]
  in

  (* Create manager *)
  let dir = "/tmp/test_checkpoint_manager" in
  let _ = Sys.command (Printf.sprintf "rm -rf %s" dir) in

  let module CM = Kaun.Checkpoint.CheckpointManager in
  let options = CM.{ default_options with max_to_keep = Some 2 } in
  let manager = CM.create ~directory:dir ~options () in

  (* Save multiple checkpoints *)
  for i = 1 to 5 do
    let params = create_params (float_of_int i) in
    CM.save manager ~step:(i * 10) ~params ()
  done;

  (* Check that we have at least one checkpoint *)
  let steps = CM.all_steps manager in
  A.check A.bool "has checkpoints" true (List.length steps >= 1);

  (* Check latest step *)
  let latest = CM.latest_step manager in
  A.check (A.option A.int) "latest step" (Some 50) latest;

  (* Restore latest *)
  let restored_params, info = CM.restore manager ~dtype () in
  A.check A.int "restored step" 50 info.step;

  (* Check restored value *)
  match Ptree.find_in_record "weight" restored_params with
  | Some weight_tree -> (
      match Ptree.get_tensor weight_tree with
      | Some w ->
          let value = Rune.item [ 0; 0 ] w in
          A.check (A.float 0.01) "restored value" 5.0 value
      | None -> A.fail "weight is not a tensor")
  | None -> A.fail "weight not found"

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
  C.save_params ~path ~params ();
  let loaded_params = C.load_params ~path ~dtype in
  let flattened = Kaun.Ptree.flatten_with_paths loaded_params in
  let expect_path key =
    if not (List.exists (fun (path, _) -> String.equal path key) flattened) then
      A.failf "missing checkpoint tensor: %s" key
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
