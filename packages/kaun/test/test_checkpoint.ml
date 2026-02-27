(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Checkpoint = Kaun.Checkpoint
module Ptree = Kaun.Ptree
module Optim = Kaun.Optim

let with_tmpfile f =
  let path = Filename.temp_file "ckpt" ".safetensors" in
  Fun.protect ~finally:(fun () -> Sys.remove path) (fun () -> f path)

let to_array t = Nx.to_array (Nx.reshape [| -1 |] (Nx.cast Nx.float32 t))

(* Checkpoint save/load *)

let test_roundtrip_single_tensor () =
  with_tmpfile (fun path ->
      let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      let tree = Ptree.tensor t in
      Checkpoint.save path tree;
      let loaded = Checkpoint.load path ~like:tree in
      match loaded with
      | Ptree.Tensor (Ptree.P lt) ->
          let vals = to_array lt in
          equal ~msg:"length" int 6 (Array.length vals);
          equal ~msg:"first" (float 1e-6) 1.0 vals.(0);
          equal ~msg:"last" (float 1e-6) 6.0 vals.(5)
      | _ -> fail "expected Tensor")

let test_roundtrip_nested_tree () =
  with_tmpfile (fun path ->
      let w = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
      let b = Nx.create Nx.float32 [| 2 |] [| 0.1; 0.2 |] in
      let tree =
        Ptree.dict
          [
            ( "layer0",
              Ptree.dict
                [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ] );
            ("layer1", Ptree.dict [ ("weight", Ptree.tensor w) ]);
          ]
      in
      Checkpoint.save path tree;
      let loaded = Checkpoint.load path ~like:tree in
      let pairs = Ptree.flatten_with_paths loaded in
      equal ~msg:"num leaves" int 3 (List.length pairs);
      let names = List.map fst pairs in
      equal ~msg:"paths" (list string)
        [ "layer0.weight"; "layer0.bias"; "layer1.weight" ]
        names)

let test_roundtrip_list_tree () =
  with_tmpfile (fun path ->
      let t0 = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
      let t1 = Nx.create Nx.float32 [| 2 |] [| 4.; 5. |] in
      let tree = Ptree.list [ Ptree.tensor t0; Ptree.tensor t1 ] in
      Checkpoint.save path tree;
      let loaded = Checkpoint.load path ~like:tree in
      let pairs = Ptree.flatten_with_paths loaded in
      equal ~msg:"num leaves" int 2 (List.length pairs);
      let _, Ptree.P lt1 = List.nth pairs 1 in
      let vals = to_array lt1 in
      equal ~msg:"second tensor" (float 1e-6) 5.0 vals.(1))

let test_missing_key () =
  with_tmpfile (fun path ->
      let t = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
      let small = Ptree.dict [ ("a", Ptree.tensor t) ] in
      Checkpoint.save path small;
      let big = Ptree.dict [ ("a", Ptree.tensor t); ("b", Ptree.tensor t) ] in
      raises_invalid_arg "Checkpoint.load: missing key \"b\"" (fun () ->
          ignore (Checkpoint.load path ~like:big)))

let test_shape_mismatch () =
  with_tmpfile (fun path ->
      let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      let tree = Ptree.tensor t in
      Checkpoint.save path tree;
      let wrong =
        Ptree.tensor
          (Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
      in
      raises_invalid_arg
        "Checkpoint.load: shape mismatch for \"\": expected [3; 2], got [2; 3]"
        (fun () -> ignore (Checkpoint.load path ~like:wrong)))

let test_dtype_casting () =
  with_tmpfile (fun path ->
      let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
      Checkpoint.save path (Ptree.tensor t);
      let template =
        Ptree.tensor (Nx.create Nx.float64 [| 3 |] [| 0.; 0.; 0. |])
      in
      let loaded = Checkpoint.load path ~like:template in
      match loaded with
      | Ptree.Tensor (Ptree.P lt) ->
          let vals = to_array lt in
          equal ~msg:"casted value" (float 1e-6) 2.0 vals.(1)
      | _ -> fail "expected Tensor")

let test_empty_tree () =
  with_tmpfile (fun path ->
      Checkpoint.save path Ptree.empty;
      let loaded = Checkpoint.load path ~like:Ptree.empty in
      match loaded with Ptree.List [] -> () | _ -> fail "expected empty list")

(* Optim state serialization *)

let test_optim_sgd_no_momentum () =
  let params = Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  let algo = Optim.sgd ~lr:(Optim.Schedule.constant 0.01) () in
  let st = Optim.init algo params in
  let count, trees = Optim.state_to_trees st in
  equal ~msg:"count" int 0 count;
  equal ~msg:"no trees" int 0 (List.length trees);
  let st' = Optim.state_of_trees algo ~count trees in
  let count', trees' = Optim.state_to_trees st' in
  equal ~msg:"count roundtrip" int 0 count';
  equal ~msg:"trees roundtrip" int 0 (List.length trees')

let test_optim_sgd_momentum () =
  let params = Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  let algo = Optim.sgd ~lr:(Optim.Schedule.constant 0.01) ~momentum:0.9 () in
  let st = Optim.init algo params in
  let count, trees = Optim.state_to_trees st in
  equal ~msg:"count" int 0 count;
  equal ~msg:"one tree" int 1 (List.length trees)

let test_optim_adam_roundtrip () =
  let params = Ptree.tensor (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  let algo = Optim.adam ~lr:(Optim.Schedule.constant 0.001) () in
  let st = Optim.init algo params in
  let count, trees = Optim.state_to_trees st in
  equal ~msg:"count" int 0 count;
  equal ~msg:"two trees" int 2 (List.length trees);
  let st' = Optim.state_of_trees algo ~count trees in
  let count', trees' = Optim.state_to_trees st' in
  equal ~msg:"count roundtrip" int 0 count';
  equal ~msg:"trees roundtrip" int 2 (List.length trees')

let test_optim_wrong_tree_count () =
  let algo = Optim.adam ~lr:(Optim.Schedule.constant 0.001) () in
  raises_invalid_arg "Optim.state_of_trees: adam expects 2 trees, got 1"
    (fun () -> ignore (Optim.state_of_trees algo ~count:0 [ Ptree.empty ]))

let () =
  run "Kaun.Checkpoint"
    [
      group "save/load"
        [
          test "roundtrip single tensor" test_roundtrip_single_tensor;
          test "roundtrip nested tree" test_roundtrip_nested_tree;
          test "roundtrip list tree" test_roundtrip_list_tree;
          test "missing key" test_missing_key;
          test "shape mismatch" test_shape_mismatch;
          test "dtype casting" test_dtype_casting;
          test "empty tree" test_empty_tree;
        ];
      group "optim serialization"
        [
          test "sgd no momentum" test_optim_sgd_no_momentum;
          test "sgd momentum" test_optim_sgd_momentum;
          test "adam roundtrip" test_optim_adam_roundtrip;
          test "wrong tree count" test_optim_wrong_tree_count;
        ];
    ]
