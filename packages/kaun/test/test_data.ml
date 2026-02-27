(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Data = Kaun.Data

let dtype = Nx.float32

(* Constructors *)

let test_of_array () =
  let d = Data.of_array [| 10; 20; 30 |] in
  equal ~msg:"length" (option int) (Some 3) (Data.length d);
  let a = Data.to_array d in
  equal ~msg:"elements" (array int) [| 10; 20; 30 |] a

let test_of_fn () =
  let d = Data.of_fn 4 (fun i -> i * i) in
  equal ~msg:"length" (option int) (Some 4) (Data.length d);
  let a = Data.to_array d in
  equal ~msg:"elements" (array int) [| 0; 1; 4; 9 |] a

let test_of_fn_negative () =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Data.of_fn (-1) Fun.id))

let test_of_tensor () =
  let t = Nx.create dtype [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let d = Data.of_tensor t in
  equal ~msg:"length" (option int) (Some 3) (Data.length d);
  let a = Data.to_array d in
  equal ~msg:"count" int 3 (Array.length a);
  equal ~msg:"shape" (list int) [ 2 ] (Array.to_list (Nx.shape a.(0)));
  equal ~msg:"first elem" (float 1e-6) 1.0 (Nx.item [ 0 ] a.(0))

let test_of_tensors () =
  let x = Nx.create dtype [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let y = Nx.create dtype [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let d = Data.of_tensors (x, y) in
  equal ~msg:"length" (option int) (Some 3) (Data.length d);
  let a = Data.to_array d in
  equal ~msg:"count" int 3 (Array.length a);
  let x0, y0 = a.(0) in
  equal ~msg:"x0 shape" (list int) [ 2 ] (Array.to_list (Nx.shape x0));
  equal ~msg:"y0 scalar" (float 1e-6) 10.0 (Nx.item [] y0)

let test_of_tensors_mismatch () =
  let x = Nx.create dtype [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let y = Nx.create dtype [| 2 |] [| 10.0; 20.0 |] in
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Data.of_tensors (x, y)))

(* Transformers *)

let test_map () =
  let d = Data.of_array [| 1; 2; 3 |] |> Data.map (fun x -> x * 2) in
  equal ~msg:"mapped" (array int) [| 2; 4; 6 |] (Data.to_array d)

let test_batch () =
  let d = Data.of_array [| 1; 2; 3; 4; 5 |] |> Data.batch 2 in
  let batches = Data.to_array d in
  equal ~msg:"num batches" int 3 (Array.length batches);
  equal ~msg:"batch 0" (array int) [| 1; 2 |] batches.(0);
  equal ~msg:"batch 1" (array int) [| 3; 4 |] batches.(1);
  equal ~msg:"batch 2 (partial)" (array int) [| 5 |] batches.(2)

let test_batch_drop_last () =
  let d = Data.of_array [| 1; 2; 3; 4; 5 |] |> Data.batch ~drop_last:true 2 in
  let batches = Data.to_array d in
  equal ~msg:"num batches" int 2 (Array.length batches);
  equal ~msg:"batch 0" (array int) [| 1; 2 |] batches.(0);
  equal ~msg:"batch 1" (array int) [| 3; 4 |] batches.(1)

let test_batch_invalid_size () =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Data.of_array [| 1; 2 |] |> Data.batch 0))

let test_map_batch () =
  let d =
    Data.of_array [| 1; 2; 3; 4 |]
    |> Data.map_batch 2 (fun batch -> Array.fold_left ( + ) 0 batch)
  in
  equal ~msg:"map_batch" (array int) [| 3; 7 |] (Data.to_array d)

let test_shuffle_deterministic () =
  let d1 =
    Nx.Rng.run ~seed:42 @@ fun () ->
    Data.of_array [| 0; 1; 2; 3; 4; 5; 6; 7 |] |> Data.shuffle |> Data.to_array
  in
  let d2 =
    Nx.Rng.run ~seed:42 @@ fun () ->
    Data.of_array [| 0; 1; 2; 3; 4; 5; 6; 7 |] |> Data.shuffle |> Data.to_array
  in
  equal ~msg:"same seed same order" (array int) d1 d2

let test_shuffle_different_seed () =
  let a1 =
    Nx.Rng.run ~seed:1 @@ fun () ->
    Data.of_array [| 0; 1; 2; 3; 4; 5; 6; 7 |] |> Data.shuffle |> Data.to_array
  in
  let a2 =
    Nx.Rng.run ~seed:2 @@ fun () ->
    Data.of_array [| 0; 1; 2; 3; 4; 5; 6; 7 |] |> Data.shuffle |> Data.to_array
  in
  is_true ~msg:"different seed different order" (a1 <> a2)

(* Consumers *)

let test_fold () =
  let sum = Data.of_array [| 1; 2; 3; 4 |] |> Data.fold ( + ) 0 in
  equal ~msg:"fold sum" int 10 sum

let test_to_seq () =
  let s = Data.of_array [| 10; 20; 30 |] |> Data.to_seq in
  let a = Array.of_seq s in
  equal ~msg:"to_seq" (array int) [| 10; 20; 30 |] a

(* Properties *)

let test_reset () =
  let d = Data.of_array [| 1; 2; 3 |] in
  let a1 = Data.to_array d in
  Data.reset d;
  let a2 = Data.to_array d in
  equal ~msg:"reset re-iterates" (array int) a1 a2

let test_length () =
  let d = Data.of_array [| 1; 2; 3 |] in
  equal ~msg:"known length" (option int) (Some 3) (Data.length d);
  let d2 = Data.map (fun x -> x + 1) d in
  equal ~msg:"map preserves length" (option int) (Some 3) (Data.length d2)

(* Utilities *)

let test_stack_batch () =
  let tensors =
    [|
      Nx.create dtype [| 2 |] [| 1.0; 2.0 |];
      Nx.create dtype [| 2 |] [| 3.0; 4.0 |];
      Nx.create dtype [| 2 |] [| 5.0; 6.0 |];
    |]
  in
  let stacked = Data.stack_batch tensors in
  equal ~msg:"shape" (list int) [ 3; 2 ] (Array.to_list (Nx.shape stacked));
  equal ~msg:"value" (float 1e-6) 3.0 (Nx.item [ 1; 0 ] stacked)

let () =
  run "Kaun.Data"
    [
      group "constructors"
        [
          test "of_array" test_of_array;
          test "of_fn" test_of_fn;
          test "of_fn negative" test_of_fn_negative;
          test "of_tensor" test_of_tensor;
          test "of_tensors" test_of_tensors;
          test "of_tensors mismatch" test_of_tensors_mismatch;
        ];
      group "transformers"
        [
          test "map" test_map;
          test "batch" test_batch;
          test "batch drop_last" test_batch_drop_last;
          test "batch invalid size" test_batch_invalid_size;
          test "map_batch" test_map_batch;
          test "shuffle deterministic" test_shuffle_deterministic;
          test "shuffle different seed" test_shuffle_different_seed;
        ];
      group "consumers" [ test "fold" test_fold; test "to_seq" test_to_seq ];
      group "properties" [ test "reset" test_reset; test "length" test_length ];
      group "utilities" [ test "stack_batch" test_stack_batch ];
    ]
