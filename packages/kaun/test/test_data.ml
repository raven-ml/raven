(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* [arange n cols] is an [n; cols] float32 tensor whose row [i] is [i * cols +
   0, ..., i * cols + cols - 1]: every element identifies its example, so batch
   contents are checkable exactly. *)
let arange n cols =
  Nx.create Nx.float32 [| n; cols |] (Array.init (n * cols) float_of_int)

let shapes seq = List.of_seq (Seq.map (fun b -> Array.to_list (Nx.shape b)) seq)

let elements seq =
  List.concat_map (fun b -> Array.to_list (Nx.to_array b)) (List.of_seq seq)

let exact = float 0.
let sorted l = List.sort Float.compare l

(* Batch shapes *)

let test_consecutive_batches () =
  let data = Data.batches ~batch_size:4 (arange 10 3) in
  equal (list (list int)) [ [ 4; 3 ]; [ 4; 3 ]; [ 2; 3 ] ] (shapes data);
  equal (list exact)
    (List.init 30 float_of_int)
    (elements data) ~msg:"example order preserved"

let test_drop_last () =
  let data = Data.batches ~drop_last:true ~batch_size:4 (arange 10 3) in
  equal (list (list int)) [ [ 4; 3 ]; [ 4; 3 ] ] (shapes data);
  equal (list exact)
    (List.init 24 float_of_int)
    (elements data) ~msg:"remainder examples dropped"

let test_short_batch () =
  let data = Data.batches ~batch_size:8 (arange 3 1) in
  equal (list (list int)) [ [ 3; 1 ] ] (shapes data)

let test_short_batch_drop_last () =
  let data = Data.batches ~drop_last:true ~batch_size:8 (arange 3 1) in
  is_true ~msg:"no batch survives drop_last" (Seq.is_empty data)

let test_exact_fit () =
  let data = Data.batches ~batch_size:4 (arange 8 2) in
  equal (list (list int)) [ [ 4; 2 ]; [ 4; 2 ] ] (shapes data)

let test_empty_dataset () =
  let empty = Nx.create Nx.float32 [| 0; 2 |] [||] in
  is_true ~msg:"no batches" (Seq.is_empty (Data.batches ~batch_size:4 empty));
  is_true ~msg:"no shuffled batches"
    (Seq.is_empty (Data.batches ~shuffle:true ~batch_size:4 empty))

(* Shuffling *)

let shuffled_elements ~seed =
  Nx.Rng.run ~seed @@ fun () ->
  elements (Data.batches ~shuffle:true ~batch_size:4 (arange 10 1))

let test_shuffle_deterministic () =
  equal (list exact) (shuffled_elements ~seed:0) (shuffled_elements ~seed:0)

let test_shuffle_is_permutation () =
  let seen = shuffled_elements ~seed:0 in
  not_equal (list exact)
    (List.init 10 float_of_int)
    seen ~msg:"order changed by seed 0";
  equal (list exact)
    (List.init 10 float_of_int)
    (sorted seen) ~msg:"same multiset of examples"

let test_shuffle_keeps_batch_shapes () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let data = Data.batches ~shuffle:true ~batch_size:4 (arange 10 3) in
  equal (list (list int)) [ [ 4; 3 ]; [ 4; 3 ]; [ 2; 3 ] ] (shapes data)

let test_epochs_reshuffle () =
  Nx.Rng.run ~seed:1 @@ fun () ->
  let data = Data.batches ~shuffle:true ~batch_size:4 (arange 32 1) in
  let epoch1 = elements data in
  let epoch2 = elements data in
  not_equal (list exact) epoch1 epoch2 ~msg:"traversals reshuffle";
  equal (list exact) (sorted epoch1) (sorted epoch2)
    ~msg:"same multiset each epoch"

(* Paired datasets *)

let paired n =
  let x = Nx.create Nx.float32 [| n |] (Array.init n float_of_int) in
  let y =
    Nx.create Nx.float64 [| n |] (Array.init n (fun i -> float_of_int (2 * i)))
  in
  (x, y)

let test_batches2_shapes () =
  let x = arange 10 3 in
  let y = Nx.create Nx.float32 [| 10; 1 |] (Array.init 10 float_of_int) in
  let data = Data.batches2 ~batch_size:4 (x, y) in
  equal
    (list (pair (list int) (list int)))
    [ ([ 4; 3 ], [ 4; 1 ]); ([ 4; 3 ], [ 4; 1 ]); ([ 2; 3 ], [ 2; 1 ]) ]
    (List.of_seq
       (Seq.map
          (fun (xb, yb) ->
            (Array.to_list (Nx.shape xb), Array.to_list (Nx.shape yb)))
          data))

let test_batches2_shuffle_keeps_pairing () =
  Nx.Rng.run ~seed:3 @@ fun () ->
  let data = Data.batches2 ~shuffle:true ~batch_size:4 (paired 10) in
  let xs, ys =
    Seq.fold_left
      (fun (xs, ys) (xb, yb) ->
        ( xs @ Array.to_list (Nx.to_array xb),
          ys @ Array.to_list (Nx.to_array yb) ))
      ([], []) data
  in
  equal (list exact)
    (List.map (fun v -> 2. *. v) xs)
    ys ~msg:"targets track their inputs";
  equal (list exact)
    (List.init 10 float_of_int)
    (sorted xs) ~msg:"all examples visited once"

(* Composition with Seq *)

let test_seq_map () =
  let data =
    Data.batches ~batch_size:4 (arange 4 1) |> Seq.map (fun b -> Nx.mul_s b 2.)
  in
  equal (list exact) [ 0.; 2.; 4.; 6. ] (elements data)

let test_seq_take () =
  let data = Data.batches ~batch_size:2 (arange 10 1) |> Seq.take 2 in
  equal (list exact) [ 0.; 1.; 2.; 3. ] (elements data)

let test_seq_fold_left () =
  let data = Data.batches ~batch_size:4 (arange 10 1) in
  let steps = Seq.fold_left (fun k _ -> k + 1) 0 data in
  equal int 3 steps

(* Errors *)

let test_invalid_batch_size () =
  raises_invalid_arg "Data.batches: batch_size must be positive, got 0"
    (fun () -> Data.batches ~batch_size:0 (arange 4 1));
  raises_invalid_arg "Data.batches2: batch_size must be positive, got -1"
    (fun () -> Data.batches2 ~batch_size:(-1) (paired 4))

let test_scalar_input () =
  let scalar = Nx.create Nx.float32 [||] [| 1. |] in
  raises_invalid_arg "Data.batches: input must not be a scalar" (fun () ->
      Data.batches ~batch_size:1 scalar)

let test_mismatched_examples () =
  let x = arange 4 1 and y = arange 3 1 in
  raises_invalid_arg "Data.batches2: x has 4 examples but y has 3" (fun () ->
      Data.batches2 ~batch_size:2 (x, y))

let tests =
  [
    group "batch shapes"
      [
        test "cuts consecutive batches and keeps the remainder"
          test_consecutive_batches;
        test "drop_last drops a short final batch" test_drop_last;
        test "one short batch when batch_size exceeds examples" test_short_batch;
        test "drop_last yields nothing when batch_size exceeds examples"
          test_short_batch_drop_last;
        test "exact fit yields only full batches" test_exact_fit;
        test "empty dataset yields no batches" test_empty_dataset;
      ];
    group "shuffling"
      [
        test "same seed gives the same order" test_shuffle_deterministic;
        test "shuffling permutes without loss" test_shuffle_is_permutation;
        test "shuffling keeps batch shapes" test_shuffle_keeps_batch_shapes;
        test "re-traversal reshuffles each epoch" test_epochs_reshuffle;
      ];
    group "paired datasets"
      [
        test "pairs input and target slices" test_batches2_shapes;
        test "shuffling keeps examples paired"
          test_batches2_shuffle_keeps_pairing;
      ];
    group "composition"
      [
        test "Seq.map transforms batches" test_seq_map;
        test "Seq.take truncates an epoch" test_seq_take;
        test "Seq.fold_left threads state" test_seq_fold_left;
      ];
    group "errors"
      [
        test "rejects non-positive batch_size" test_invalid_batch_size;
        test "rejects scalar input" test_scalar_input;
        test "rejects mismatched example counts" test_mismatched_examples;
      ];
  ]

let () = run "kaun data" tests
