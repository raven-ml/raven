(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap

let test_discrete_basic () =
  let space = Space.Discrete.create 5 in
  let rng = Rune.Rng.key 42 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"discrete sample in range" bool true
    (Space.contains space sample);
  let shape = Space.shape space in
  equal ~msg:"discrete shape" (option (array int)) None shape

let test_discrete_with_start () =
  let space = Space.Discrete.create ~start:10 5 in
  let rng = Rune.Rng.key 99 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"discrete sample valid" bool true
    (Space.contains space sample);
  let value =
    let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] sample) in
    Int32.to_int arr.(0)
  in
  equal ~msg:"discrete start offset" bool true (value >= 10 && value < 15)

let test_box_1d () =
  let space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let rng = Rune.Rng.key 123 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"box 1d sample valid" bool true (Space.contains space sample);
  let shape = Space.shape space in
  equal ~msg:"box 1d shape" (option (array int)) (Some [| 1 |]) shape

let test_box_multidim () =
  let space =
    Space.Box.create ~low:[| -1.0; -2.0; -3.0 |] ~high:[| 1.0; 2.0; 3.0 |]
  in
  let rng = Rune.Rng.key 456 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"box multidim valid" bool true (Space.contains space sample);
  let shape = Space.shape space in
  equal ~msg:"box multidim shape" (option (array int))
    (Some [| 3 |]) shape;
  let arr : float array = Rune.to_array (Rune.reshape [| 3 |] sample) in
  equal ~msg:"box bounds respected" bool true
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
  equal ~msg:"multi_binary sample valid" bool true
    (Space.contains space sample);
  let shape = Space.shape space in
  equal ~msg:"multi_binary shape" (option (array int))
    (Some [| 4 |]) shape;
  let arr : Int32.t array = Rune.to_array (Rune.reshape [| 4 |] sample) in
  Array.iter
    (fun v -> equal ~msg:"binary value" bool true (v = 0l || v = 1l))
    arr

let test_multi_discrete () =
  let space = Space.Multi_discrete.create [| 3; 4; 5 |] in
  let rng = Rune.Rng.key 321 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"multi_discrete sample valid" bool true
    (Space.contains space sample);
  let shape = Space.shape space in
  equal ~msg:"multi_discrete shape" (option (array int))
    (Some [| 3 |]) shape

let test_tuple_space () =
  let space1 = Space.Discrete.create 3 in
  let space2 = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let tuple_space =
    Space.Tuple.create [ Space.Pack space1; Space.Pack space2 ]
  in
  let rng = Rune.Rng.key 654 in
  let sample, _ = Space.sample ~rng tuple_space in
  equal ~msg:"tuple sample valid" bool true
    (Space.contains tuple_space sample);
  equal ~msg:"tuple has 2 elements" int 2 (List.length sample)

let test_dict_space () =
  let disc_space = Space.Discrete.create 5 in
  let box_space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let dict_space =
    Space.Dict.create
      [ ("action", Space.Pack disc_space); ("value", Space.Pack box_space) ]
  in
  let rng = Rune.Rng.key 987 in
  let sample, _ = Space.sample ~rng dict_space in
  equal ~msg:"dict sample valid" bool true
    (Space.contains dict_space sample);
  equal ~msg:"dict has 2 keys" int 2 (List.length sample);
  let keys = List.map fst sample in
  equal ~msg:"dict has action key" bool true (List.mem "action" keys);
  equal ~msg:"dict has value key" bool true (List.mem "value" keys)

let test_sequence_space () =
  let elem_space = Space.Discrete.create 3 in
  let seq_space =
    Space.Sequence.create ~min_length:2 ~max_length:5 elem_space
  in
  let rng = Rune.Rng.key 111 in
  let sample, _ = Space.sample ~rng seq_space in
  equal ~msg:"sequence sample valid" bool true
    (Space.contains seq_space sample);
  let len = List.length sample in
  equal ~msg:"sequence length in range" bool true (len >= 2 && len <= 5)

let test_sequence_space_unbounded () =
  let elem_space = Space.Discrete.create 4 in
  let seq_space = Space.Sequence.create ~min_length:1 elem_space in
  let rng = Rune.Rng.key 512 in
  let sample, _ = Space.sample ~rng seq_space in
  equal ~msg:"default sample length is min_length" int 1 (List.length sample);
  let extended =
    List.init 4 (fun i ->
        let rng = Rune.Rng.key (800 + i) in
        fst (Space.sample ~rng elem_space))
  in
  equal ~msg:"contains extended sequence" bool true
    (Space.contains seq_space extended);
  let packed = Space.pack seq_space extended in
  match Space.unpack seq_space packed with
  | Ok unpacked ->
      equal ~msg:"unbounded unpack preserves length" int 4 (List.length unpacked)
  | Error msg -> fail ("unbounded sequence unpack failed: " ^ msg)

let test_text_space () =
  let space = Space.Text.create ~max_length:10 () in
  let rng = Rune.Rng.key 222 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"text sample valid" bool true (Space.contains space sample);
  equal ~msg:"text length respects max" bool true
    (String.length sample <= 10)

let test_text_custom_charset () =
  let space = Space.Text.create ~charset:"ABC" ~max_length:5 () in
  let rng = Rune.Rng.key 333 in
  let sample, _ = Space.sample ~rng space in
  equal ~msg:"text custom charset valid" bool true
    (Space.contains space sample);
  String.iter
    (fun c ->
      equal ~msg:"char in charset" bool true (String.contains "ABC" c))
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
      equal ~msg:"pack/unpack preserves value" int32 sample_val unpacked_val
  | Error msg -> fail ("Unpack failed: " ^ msg)

let () =
  run "Space"
    [
      group "Discrete"
        [
          test "basic discrete space" test_discrete_basic;
          test "discrete with start offset" test_discrete_with_start;
        ];
      group "Box"
        [
          test "1D box space" test_box_1d;
          test "multidimensional box" test_box_multidim;
        ];
      group "Multi"
        [
          test "multi-binary space" test_multi_binary;
          test "multi-discrete space" test_multi_discrete;
        ];
      group "Composite"
        [
          test "tuple space" test_tuple_space;
          test "dict space" test_dict_space;
          test "sequence space" test_sequence_space;
          test "sequence space unbounded"
            test_sequence_space_unbounded;
        ];
      group "Text"
        [
          test "text space" test_text_space;
          test "text with custom charset" test_text_custom_charset;
        ];
      group "Serialization" [ test "pack/unpack" test_pack_unpack ];
    ]
