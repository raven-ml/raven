open Kaun

let array_all_close arr1 arr2 tol =
  Array.length arr1 = Array.length arr2 &&
  Array.for_all2 (fun a b -> abs_float (a -. b) < tol) arr1 arr2

let test_kaun_serialization () =
  let obs_dim = 4 in
  let n_actions = 2 in
  let q_network =
    Kaun.Layer.sequential [
      Kaun.Layer.linear ~in_features:obs_dim ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:n_actions ();
    ]
  in
  let rng = Rune.Rng.key 42 in
  let params = q_network.init ~rngs:rng ~dtype:Rune.float32 in
  let input = Rune.create Rune.float32 [| obs_dim |] [| 0.5; 0.5; 0.5; 0.5 |] in
  let output_before = q_network.apply params ~training:false input in

  let temp_dir = Filename.concat "/tmp" ("kaun_test_" ^ string_of_int (Random.bits ())) in
  if not (Sys.file_exists temp_dir) then Unix.mkdir temp_dir 0o755;
  let param_path = Filename.concat temp_dir "q_params.safetensors" in
  let checkpointer = Kaun.Checkpoint.Checkpointer.create () in
  Kaun.Checkpoint.Checkpointer.save_file checkpointer ~params ~path:param_path ();

  let loaded_q_network =
    Kaun.Layer.sequential [
      Kaun.Layer.linear ~in_features:obs_dim ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:n_actions ();
    ]
  in
  let loaded_params = Kaun.Checkpoint.Checkpointer.restore_file checkpointer
    ~path:param_path
    ~dtype:Rune.float32
  in
  let output_after = loaded_q_network.apply loaded_params ~training:false input in

  Printf.printf "Output before: %s\n" (Rune.to_string output_before);
  Printf.printf "Output after: %s\n" (Rune.to_string output_after);

  let arr_before = Rune.to_array output_before in
  let arr_after = Rune.to_array output_after in
  let same = array_all_close arr_before arr_after 1e-6 in
  Printf.printf "Outputs match: %b\n" same

let () = test_kaun_serialization ()