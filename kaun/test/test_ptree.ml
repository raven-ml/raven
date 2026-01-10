(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Alcotest
module Ptree = Kaun.Ptree

let test_map () =
  let tree =
    Ptree.dict
      [
        ("layer1", Ptree.tensor (Rune.ones Rune.float32 [| 2; 3 |]));
        ( "layer2",
          Ptree.list
            [
              Ptree.tensor (Rune.ones Rune.float32 [| 4 |]);
              Ptree.tensor (Rune.ones Rune.float32 [| 5 |]);
            ] );
      ]
  in
  let doubled =
    Ptree.map (fun t -> Rune.mul t (Rune.scalar Rune.float32 2.0)) tree
  in

  (* Check that all tensors are doubled *)
  Ptree.iter
    (fun tensor ->
      Ptree.with_tensor tensor
        {
          run =
            (fun (type a) (type layout) (t : (a, layout) Rune.t) ->
              let t = Rune.cast Rune.float32 t in
              let first_val = Rune.item [ 0 ] (Rune.reshape [| -1 |] t) in
              check (float 0.01) "tensor doubled" 2.0 first_val);
        })
    doubled

let test_map2 () =
  let tree1 = Ptree.tensor (Rune.ones Rune.float32 [| 3 |]) in
  let tree2 = Ptree.tensor (Rune.full Rune.float32 [| 3 |] 2.0) in
  let sum = Ptree.map2 Rune.add tree1 tree2 in

  match sum with
  | Ptree.Tensor tensor ->
      Ptree.with_tensor tensor
        {
          run =
            (fun (type a) (type layout) (t : (a, layout) Rune.t) ->
              let t = Rune.cast Rune.float32 t in
              let first_val = Rune.item [ 0 ] t in
              check (float 0.01) "1 + 2 = 3" 3.0 first_val);
        }
  | _ -> fail "Expected Tensor"

let test_count () =
  let tree =
    Ptree.dict
      [
        ("a", Ptree.tensor (Rune.zeros Rune.float32 [| 2; 3 |]));
        ("b", Ptree.tensor (Rune.zeros Rune.float32 [| 4; 5 |]));
        ( "c",
          Ptree.list
            [
              Ptree.tensor (Rune.zeros Rune.float32 [| 6 |]);
              Ptree.tensor (Rune.zeros Rune.float32 [| 7 |]);
            ] );
      ]
  in

  check int "count tensors" 4 (Ptree.count_tensors tree);
  check int "count parameters" (6 + 20 + 6 + 7) (Ptree.count_parameters tree)

let test_flat_list () =
  let tree =
    Ptree.list
      [
        Ptree.tensor (Rune.full Rune.float32 [| 2 |] 1.0);
        Ptree.tensor (Rune.full Rune.float32 [| 3 |] 2.0);
      ]
  in

  let flat, rebuild = Ptree.flatten tree in
  check int "flat list length" 2 (List.length flat);

  (* Modify the flat list *)
  let modified =
    List.map
      (fun (Ptree.P t) ->
        let dt = Rune.dtype t in
        let ten = Nx_core.Dtype.of_float dt 10.0 in
        Ptree.P (Rune.add t (Rune.scalar dt ten)))
      flat
  in

  (* Reconstruct the tree *)
  let reconstructed = rebuild modified in

  match reconstructed with
  | Ptree.List [ Ptree.Tensor tensor1; Ptree.Tensor tensor2 ] ->
      Ptree.with_tensor tensor1
        {
          run =
            (fun (type a) (type layout) (t1 : (a, layout) Rune.t) ->
              let t1 = Rune.cast Rune.float32 t1 in
              check (float 0.01) "first tensor modified" 11.0
                (Rune.item [ 0 ] t1));
        };
      Ptree.with_tensor tensor2
        {
          run =
            (fun (type a) (type layout) (t2 : (a, layout) Rune.t) ->
              let t2 = Rune.cast Rune.float32 t2 in
              check (float 0.01) "second tensor modified" 12.0
                (Rune.item [ 0 ] t2));
        }
  | _ -> fail "Unexpected structure"

let () =
  run "Kaun.Ptree"
    [
      ( "operations",
        [
          test_case "map" `Quick test_map;
          test_case "map2" `Quick test_map2;
          test_case "count" `Quick test_count;
          test_case "flat_list" `Quick test_flat_list;
        ] );
    ]
