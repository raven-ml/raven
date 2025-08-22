open Kaun.Ptree
module A = Alcotest

let test_save_and_load () =
  let device = Rune.c in
  let dtype = Rune.float32 in

  (* Create some test parameters *)
  let w = Rune.ones device dtype [| 3; 3 |] in
  let b = Rune.zeros device dtype [| 3 |] in
  let params = Record [ ("weight", Tensor w); ("bias", Tensor b) ] in

  (* Save checkpoint to specific file *)
  let path = "/tmp/test_checkpoint.safetensors" in
  (* Use the lower-level save_params for single file *)
  let module C = Kaun.Checkpoint in
  C.save_params ~path ~params ~metadata:[ ("epoch", "10") ] ();

  (* Load checkpoint from file *)
  let loaded_params = C.load_params ~path ~device ~dtype in

  (* Check parameters *)
  match loaded_params with
  | Record fields -> (
      (match List.assoc_opt "weight" fields with
      | Some (Tensor loaded_w) ->
          let is_equal = Rune.all (Rune.equal w loaded_w) in
          let is_equal_val = Rune.unsafe_to_array is_equal in
          A.check A.bool "weights match" true (is_equal_val.(0) > 0)
      | _ -> A.fail "weight not found or wrong type");
      match List.assoc_opt "bias" fields with
      | Some (Tensor loaded_b) ->
          let is_equal = Rune.all (Rune.equal b loaded_b) in
          let is_equal_val = Rune.unsafe_to_array is_equal in
          A.check A.bool "bias matches" true (is_equal_val.(0) > 0)
      | _ -> A.fail "bias not found or wrong type")
  | _ -> A.fail "loaded params has wrong structure"

let test_checkpoint_manager () =
  let device = Rune.c in
  let dtype = Rune.float32 in

  (* Create test parameters *)
  let create_params value =
    let w = Rune.full device dtype [| 2; 2 |] value in
    Record [ ("weight", Tensor w) ]
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
  let restored_params, info = CM.restore manager ~device ~dtype () in
  A.check A.int "restored step" 50 info.step;

  (* Check restored value *)
  match restored_params with
  | Record fields -> (
      match List.assoc_opt "weight" fields with
      | Some (Tensor w) ->
          let value = Rune.unsafe_get [ 0; 0 ] w in
          A.check (A.float 0.01) "restored value" 5.0 value
      | _ -> A.fail "weight not found")
  | _ -> A.fail "wrong params structure"

let () =
  A.run "Kaun.Checkpoint"
    [
      ( "basic",
        [
          A.test_case "save_and_load" `Quick test_save_and_load;
          A.test_case "checkpoint_manager" `Quick test_checkpoint_manager;
        ] );
    ]
