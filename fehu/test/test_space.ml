(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

let test_discrete_basic () =
  let space = Space.Discrete.create 5 in
  let rng = Rune.Rng.key 42 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool)
    "discrete sample in range" true
    (Space.contains space sample);
  let shape = Space.shape space in
  Alcotest.(check (option (array int))) "discrete shape" None shape

let test_discrete_with_start () =
  let space = Space.Discrete.create ~start:10 5 in
  let rng = Rune.Rng.key 99 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool)
    "discrete sample valid" true
    (Space.contains space sample);
  let value =
    let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] sample) in
    Int32.to_int arr.(0)
  in
  Alcotest.(check bool) "discrete start offset" true (value >= 10 && value < 15)

let test_box_1d () =
  let space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let rng = Rune.Rng.key 123 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool) "box 1d sample valid" true (Space.contains space sample);
  let shape = Space.shape space in
  Alcotest.(check (option (array int))) "box 1d shape" (Some [| 1 |]) shape

let test_box_multidim () =
  let space =
    Space.Box.create ~low:[| -1.0; -2.0; -3.0 |] ~high:[| 1.0; 2.0; 3.0 |]
  in
  let rng = Rune.Rng.key 456 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool) "box multidim valid" true (Space.contains space sample);
  let shape = Space.shape space in
  Alcotest.(check (option (array int)))
    "box multidim shape" (Some [| 3 |]) shape;
  let arr : float array = Rune.to_array (Rune.reshape [| 3 |] sample) in
  Alcotest.(check bool)
    "box bounds respected" true
    (arr.(0) >= -1.0
    && arr.(0) <= 1.0
    && arr.(1) >= -2.0
    && arr.(1) <= 2.0
    && arr.(2) >= -3.0
    && arr.(2) <= 3.0)

let test_multi_binary () =
  let space = Space.Multi_binary.create 4 in
  let rng = Rune.Rng.key 789 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool)
    "multi_binary sample valid" true
    (Space.contains space sample);
  let shape = Space.shape space in
  Alcotest.(check (option (array int)))
    "multi_binary shape" (Some [| 4 |]) shape;
  let arr : Int32.t array = Rune.to_array (Rune.reshape [| 4 |] sample) in
  Array.iter
    (fun v -> Alcotest.(check bool) "binary value" true (v = 0l || v = 1l))
    arr

let test_multi_discrete () =
  let space = Space.Multi_discrete.create [| 3; 4; 5 |] in
  let rng = Rune.Rng.key 321 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool)
    "multi_discrete sample valid" true
    (Space.contains space sample);
  let shape = Space.shape space in
  Alcotest.(check (option (array int)))
    "multi_discrete shape" (Some [| 3 |]) shape

let test_tuple_space () =
  let space1 = Space.Discrete.create 3 in
  let space2 = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let tuple_space =
    Space.Tuple.create [ Space.Pack space1; Space.Pack space2 ]
  in
  let rng = Rune.Rng.key 654 in
  let sample, _ = Space.sample ~rng tuple_space in
  Alcotest.(check bool)
    "tuple sample valid" true
    (Space.contains tuple_space sample);
  Alcotest.(check int) "tuple has 2 elements" 2 (List.length sample)

let test_dict_space () =
  let disc_space = Space.Discrete.create 5 in
  let box_space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let dict_space =
    Space.Dict.create
      [ ("action", Space.Pack disc_space); ("value", Space.Pack box_space) ]
  in
  let rng = Rune.Rng.key 987 in
  let sample, _ = Space.sample ~rng dict_space in
  Alcotest.(check bool)
    "dict sample valid" true
    (Space.contains dict_space sample);
  Alcotest.(check int) "dict has 2 keys" 2 (List.length sample);
  let keys = List.map fst sample in
  Alcotest.(check bool) "dict has action key" true (List.mem "action" keys);
  Alcotest.(check bool) "dict has value key" true (List.mem "value" keys)

let test_sequence_space () =
  let elem_space = Space.Discrete.create 3 in
  let seq_space =
    Space.Sequence.create ~min_length:2 ~max_length:5 elem_space
  in
  let rng = Rune.Rng.key 111 in
  let sample, _ = Space.sample ~rng seq_space in
  Alcotest.(check bool)
    "sequence sample valid" true
    (Space.contains seq_space sample);
  let len = List.length sample in
  Alcotest.(check bool) "sequence length in range" true (len >= 2 && len <= 5)

let test_sequence_space_unbounded () =
  let elem_space = Space.Discrete.create 4 in
  let seq_space = Space.Sequence.create ~min_length:1 elem_space in
  let rng = Rune.Rng.key 512 in
  let sample, _ = Space.sample ~rng seq_space in
  Alcotest.(check int)
    "default sample length is min_length" 1 (List.length sample);
  let extended =
    List.init 4 (fun i ->
        let rng = Rune.Rng.key (800 + i) in
        fst (Space.sample ~rng elem_space))
  in
  Alcotest.(check bool)
    "contains extended sequence" true
    (Space.contains seq_space extended);
  let packed = Space.pack seq_space extended in
  match Space.unpack seq_space packed with
  | Ok unpacked ->
      Alcotest.(check int)
        "unbounded unpack preserves length" 4 (List.length unpacked)
  | Error msg -> Alcotest.fail ("unbounded sequence unpack failed: " ^ msg)

let test_text_space () =
  let space = Space.Text.create ~max_length:10 () in
  let rng = Rune.Rng.key 222 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool) "text sample valid" true (Space.contains space sample);
  Alcotest.(check bool)
    "text length respects max" true
    (String.length sample <= 10)

let test_text_custom_charset () =
  let space = Space.Text.create ~charset:"ABC" ~max_length:5 () in
  let rng = Rune.Rng.key 333 in
  let sample, _ = Space.sample ~rng space in
  Alcotest.(check bool)
    "text custom charset valid" true
    (Space.contains space sample);
  String.iter
    (fun c ->
      Alcotest.(check bool) "char in charset" true (String.contains "ABC" c))
    sample

let test_pack_unpack () =
  let space = Space.Discrete.create 10 in
  let rng = Rune.Rng.key 444 in
  let sample, _ = Space.sample ~rng space in
  let packed = Space.pack space sample in
  match Space.unpack space packed with
  | Ok unpacked ->
      let sample_val =
        let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] sample) in
        arr.(0)
      in
      let unpacked_val =
        let arr : Int32.t array =
          Rune.to_array (Rune.reshape [| 1 |] unpacked)
        in
        arr.(0)
      in
      Alcotest.(check int32)
        "pack/unpack preserves value" sample_val unpacked_val
  | Error msg -> Alcotest.fail ("Unpack failed: " ^ msg)

let () =
  let open Alcotest in
  run "Space"
    [
      ( "Discrete",
        [
          test_case "basic discrete space" `Quick test_discrete_basic;
          test_case "discrete with start offset" `Quick test_discrete_with_start;
        ] );
      ( "Box",
        [
          test_case "1D box space" `Quick test_box_1d;
          test_case "multidimensional box" `Quick test_box_multidim;
        ] );
      ( "Multi",
        [
          test_case "multi-binary space" `Quick test_multi_binary;
          test_case "multi-discrete space" `Quick test_multi_discrete;
        ] );
      ( "Composite",
        [
          test_case "tuple space" `Quick test_tuple_space;
          test_case "dict space" `Quick test_dict_space;
          test_case "sequence space" `Quick test_sequence_space;
          test_case "sequence space unbounded" `Quick
            test_sequence_space_unbounded;
        ] );
      ( "Text",
        [
          test_case "text space" `Quick test_text_space;
          test_case "text with custom charset" `Quick test_text_custom_charset;
        ] );
      ("Serialization", [ test_case "pack/unpack" `Quick test_pack_unpack ]);
    ]
