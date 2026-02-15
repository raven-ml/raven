(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Ckpt = Checkpoint
module Snapshot = Checkpoint.Snapshot

let xor_inputs =
  Rune.create Rune.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]

let xor_targets = Rune.create Rune.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |]

let model =
  Layer.sequential
    [
      Layer.linear ~in_features:2 ~out_features:4 ();
      Layer.tanh ();
      Layer.linear ~in_features:4 ~out_features:1 ();
      Layer.sigmoid ();
    ]

let repo_directory () =
  let cwd = Sys.getcwd () in
  Filename.concat cwd "checkpointing_repo"

let ensure_dir path = if Sys.file_exists path then () else Unix.mkdir path 0o755

let repo () =
  let dir = repo_directory () in
  ensure_dir dir;
  Ckpt.create_repository ~directory:dir ()

let save_checkpoint repo ~step params optimizer_state =
  let params_snapshot = Snapshot.ptree params in
  let optimizer_snapshot = Optimizer.serialize optimizer_state in
  let artifacts =
    [
      Ckpt.artifact ~label:"model" ~kind:Ckpt.Params ~snapshot:params_snapshot
        ();
      Ckpt.artifact ~label:"adam" ~kind:Ckpt.Optimizer
        ~snapshot:optimizer_snapshot ();
    ]
  in
  (match Ckpt.mem repo ~step with
  | true -> ignore (Ckpt.delete repo ~step)
  | false -> ());
  match Ckpt.write repo ~step ~metadata:[ ("example", "xor") ] ~artifacts with
  | Ok manifest ->
      let step_str =
        match manifest.step with Some s -> string_of_int s | None -> "none"
      in
      Printf.printf "Saved checkpoint step %s → %s\n%!" step_str
        (repo_directory ())
  | Error err ->
      Printf.printf "Checkpoint write error: %s\n%!" (Ckpt.error_to_string err)

let load_checkpoint repo ~step optimizer =
  match Ckpt.read repo ~step with
  | Error err ->
      failwith
        (Printf.sprintf "Unable to read checkpoint %d: %s" step
           (Ckpt.error_to_string err))
  | Ok (_manifest, artifacts) ->
      let pick kind label =
        Ckpt.filter_artifacts ~kinds:[ kind ] artifacts
        |> List.find_opt (fun a -> String.equal (Ckpt.artifact_label a) label)
      in
      let params_tree =
        match pick Ckpt.Params "model" with
        | None -> failwith "Checkpoint missing params artifact"
        | Some artifact -> (
            let snapshot = Ckpt.artifact_snapshot artifact in
            match Snapshot.to_ptree snapshot with
            | Ok tree -> tree
            | Error msg ->
                failwith (Printf.sprintf "Failed to decode params: %s" msg))
      in
      let optimizer_state =
        match pick Ckpt.Optimizer "adam" with
        | None -> failwith "Checkpoint missing optimizer artifact"
        | Some artifact -> (
            let snapshot = Ckpt.artifact_snapshot artifact in
            match Optimizer.restore optimizer snapshot with
            | Ok state -> state
            | Error msg ->
                failwith (Printf.sprintf "Failed to restore optimizer: %s" msg))
      in
      (params_tree, optimizer_state)

let training_step model params optimizer opt_state inputs targets =
  let loss, grads =
    value_and_grad
      (fun params ->
        let predictions = Kaun.apply model params ~training:true inputs in
        Loss.binary_cross_entropy predictions targets)
      params
  in
  let updates, new_state = Optimizer.step optimizer opt_state params grads in
  Optimizer.apply_updates_inplace params updates;
  (Rune.item [] loss, new_state)

let evaluate params =
  let predictions = Kaun.apply model params ~training:false xor_inputs in
  let loss = Loss.binary_cross_entropy predictions xor_targets in
  Rune.item [] loss

let () =
  let dtype = Rune.float32 in
  let lr = Optimizer.Schedule.constant 0.1 in
  let optimizer = Optimizer.adam ~lr () in
  let repo = repo () in

  let params = ref (Kaun.init model ~rngs:(Rune.Rng.key 0) ~dtype) in
  let opt_state = ref (Optimizer.init optimizer !params) in

  let total_steps = 300 in
  let checkpoint_step = 150 in

  for step = 1 to total_steps do
    let loss, new_state =
      training_step model !params optimizer !opt_state xor_inputs xor_targets
    in
    opt_state := new_state;

    if step mod 25 = 0 then Printf.printf "step %3d loss %.4f\n%!" step loss;

    if step = checkpoint_step then (
      save_checkpoint repo ~step !params !opt_state;
      (* Simulate a restart by reinitialising parameters and optimiser state *)
      params := Kaun.init model ~rngs:(Rune.Rng.key 999) ~dtype;
      opt_state := Optimizer.init optimizer !params;
      let loss_after_reset = evaluate !params in
      Printf.printf "Loss after random reset: %.4f\n%!" loss_after_reset;
      let restored_params, restored_state =
        load_checkpoint repo ~step optimizer
      in
      params := restored_params;
      opt_state := restored_state;
      let loss_after_restore = evaluate !params in
      Printf.printf "Loss right after restore: %.4f\n%!" loss_after_restore)
  done;

  let final_predictions = Kaun.apply model !params ~training:false xor_inputs in
  Printf.printf "Final XOR predictions: \n";
  Rune.print final_predictions;
  Printf.printf "Checkpoints stored in %s\n%!" (repo_directory ())
