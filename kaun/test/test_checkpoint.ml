open Kaun
module A = Alcotest

let test_save_and_load () =
  let device = Rune.c in
  let dtype = Rune.float32 in

  (* Create some test parameters *)
  let w = Rune.ones device dtype [| 3; 3 |] in
  let b = Rune.zeros device dtype [| 3 |] in
  let _params = Record [ ("weight", Tensor w); ("bias", Tensor b) ] in

  (* Save checkpoint to specific file *)
  let path = "/tmp/test_checkpoint.json" in
  (* Use the lower-level save_params for single file *)
  let module C = Kaun_checkpoint in
  let checkpoint_params =
    C.Record [ ("weight", C.Tensor w); ("bias", C.Tensor b) ]
  in
  C.save_params ~path ~params:checkpoint_params ~metadata:[ ("epoch", "10") ] ();

  (* Load checkpoint from file *)
  let loaded_checkpoint_params = C.load_params ~path ~device ~dtype in

  (* Convert back to Kaun params *)
  let rec from_checkpoint : type l d. (l, d) C.params -> (l, d) params =
    function
    | C.Tensor t -> Tensor t
    | C.List l -> List (List.map from_checkpoint l)
    | C.Record r -> Record (List.map (fun (k, v) -> (k, from_checkpoint v)) r)
  in
  let loaded_params = from_checkpoint loaded_checkpoint_params in

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

  let module CM = Kaun_checkpoint.CheckpointManager in
  let options = CM.{ default_options with max_to_keep = Some 2 } in
  let manager = CM.create ~directory:dir ~options () in

  (* Save multiple checkpoints *)
  for i = 1 to 5 do
    let params = create_params (float_of_int i) in
    (* Convert params *)
    let checkpoint_params : _ Kaun_checkpoint.params =
      match params with
      | Record fields ->
          Kaun_checkpoint.Record
            (List.map
               (fun (k, v) ->
                 match v with
                 | Tensor t -> (k, Kaun_checkpoint.Tensor t)
                 | _ -> failwith "unsupported")
               fields)
      | _ -> failwith "unsupported"
    in
    CM.save manager ~step:(i * 10) ~params:checkpoint_params ()
  done;

  (* Check that we have at least one checkpoint *)
  let steps = CM.all_steps manager in
  A.check A.bool "has checkpoints" true (List.length steps >= 1);

  (* Check latest step *)
  let latest = CM.latest_step manager in
  A.check (A.option A.int) "latest step" (Some 10) latest;

  (* Restore latest *)
  let restored_params, info = CM.restore manager ~device ~dtype () in
  A.check A.int "restored step" 10 info.step;

  (* Check restored value *)
  match restored_params with
  | Kaun_checkpoint.Record fields -> (
      match List.assoc_opt "weight" fields with
      | Some (Kaun_checkpoint.Tensor w) ->
          let value = Rune.unsafe_get [ 0; 0 ] w in
          A.check (A.float 0.01) "restored value" 1.0 value
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
