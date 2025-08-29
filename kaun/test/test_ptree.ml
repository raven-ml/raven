open Alcotest
module Ptree = Kaun.Ptree

let ctx = Rune.c

let test_map () =
  let tree =
    Ptree.record_of
      [
        ("layer1", Ptree.tensor (Rune.ones ctx Rune.float32 [| 2; 3 |]));
        ( "layer2",
          Ptree.list_of
            [
              Ptree.tensor (Rune.ones ctx Rune.float32 [| 4 |]);
              Ptree.tensor (Rune.ones ctx Rune.float32 [| 5 |]);
            ] );
      ]
  in
  let doubled =
    Ptree.map (fun t -> Rune.mul t (Rune.scalar ctx Rune.float32 2.0)) tree
  in

  (* Check that all tensors are doubled *)
  Ptree.iter
    (fun t ->
      let first_val = Rune.item [ 0 ] (Rune.reshape [| -1 |] t) in
      check (float 0.01) "tensor doubled" 2.0 first_val)
    doubled

let test_map2 () =
  let tree1 = Ptree.Tensor (Rune.ones ctx Rune.float32 [| 3 |]) in
  let tree2 = Ptree.Tensor (Rune.full ctx Rune.float32 [| 3 |] 2.0) in
  let sum = Ptree.add tree1 tree2 in

  match sum with
  | Tensor t ->
      let first_val = Rune.item [ 0 ] t in
      check (float 0.01) "1 + 2 = 3" 3.0 first_val
  | _ -> fail "Expected Tensor"

let test_count () =
  let tree =
    Ptree.record_of
      [
        ("a", Ptree.Tensor (Rune.zeros ctx Rune.float32 [| 2; 3 |]));
        ("b", Ptree.Tensor (Rune.zeros ctx Rune.float32 [| 4; 5 |]));
        ( "c",
          Ptree.List
            [
              Ptree.Tensor (Rune.zeros ctx Rune.float32 [| 6 |]);
              Ptree.Tensor (Rune.zeros ctx Rune.float32 [| 7 |]);
            ] );
      ]
  in

  check int "count tensors" 4 (Ptree.count_tensors tree);
  check int "count parameters" (6 + 20 + 6 + 7) (Ptree.count_parameters tree)

let test_flat_list () =
  let tree =
    Ptree.List
      [
        Ptree.Tensor (Rune.full ctx Rune.float32 [| 2 |] 1.0);
        Ptree.Tensor (Rune.full ctx Rune.float32 [| 3 |] 2.0);
      ]
  in

  let flat, rebuild = Ptree.flatten tree in
  check int "flat list length" 2 (List.length flat);

  (* Modify the flat list *)
  let modified =
    List.map (fun t -> Rune.add t (Rune.scalar ctx Rune.float32 10.0)) flat
  in

  (* Reconstruct the tree *)
  let reconstructed = rebuild modified in

  match reconstructed with
  | List [ Tensor t1; Tensor t2 ] ->
      check (float 0.01) "first tensor modified" 11.0 (Rune.item [ 0 ] t1);
      check (float 0.01) "second tensor modified" 12.0 (Rune.item [ 0 ] t2)
  | _ -> fail "Unexpected structure"

let test_equal_structure () =
  let tree1 =
    Ptree.record_of
      [
        ("a", Ptree.Tensor (Rune.zeros ctx Rune.float32 [| 2 |]));
        ("b", Ptree.List []);
      ]
  in
  let tree2 =
    Ptree.record_of
      [
        ("a", Ptree.Tensor (Rune.ones ctx Rune.float32 [| 3 |]));
        ("b", Ptree.List []);
      ]
  in
  let tree3 =
    Ptree.record_of
      [ ("a", Ptree.Tensor (Rune.zeros ctx Rune.float32 [| 2 |])) ]
  in

  check bool "same structure" true (Ptree.equal_structure tree1 tree2);
  check bool "different structure" false (Ptree.equal_structure tree1 tree3)

let test_arithmetic () =
  let a = Ptree.Tensor (Rune.full ctx Rune.float32 [| 2 |] 10.0) in
  let b = Ptree.Tensor (Rune.full ctx Rune.float32 [| 2 |] 3.0) in

  let test_op name op expected =
    match op a b with
    | Ptree.Tensor t -> check (float 0.01) name expected (Rune.item [ 0 ] t)
    | _ -> fail "Expected Tensor"
  in

  test_op "add" Ptree.add 13.0;
  test_op "sub" Ptree.sub 7.0;
  test_op "mul" Ptree.mul 30.0;
  test_op "div" Ptree.div (10.0 /. 3.0);

  match Ptree.scale 2.0 a with
  | Tensor t -> check (float 0.01) "scale" 20.0 (Rune.item [ 0 ] t)
  | _ -> fail "Expected Tensor"

let () =
  run "Kaun.Ptree"
    [
      ( "operations",
        [
          test_case "map" `Quick test_map;
          test_case "map2" `Quick test_map2;
          test_case "count" `Quick test_count;
          test_case "flat_list" `Quick test_flat_list;
          test_case "equal_structure" `Quick test_equal_structure;
          test_case "arithmetic" `Quick test_arithmetic;
        ] );
    ]
