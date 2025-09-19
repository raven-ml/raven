open Rune
module A = Alcotest

let test_key_creation () =
  let key1 = Rng.key 42 in
  let key2 = Rng.key 42 in
  let key3 = Rng.key 43 in
  A.check A.int "same seed produces same key" (Rng.to_int key1)
    (Rng.to_int key2);
  A.check A.bool "different seeds produce different keys" true
    (Rng.to_int key1 <> Rng.to_int key3)

let test_key_splitting () =
  let key = Rng.key 42 in
  let keys = Rng.split key in
  A.check A.int "default split produces 2 keys" 2 (Array.length keys);

  let keys3 = Rng.split ~n:3 key in
  A.check A.int "split with n=3 produces 3 keys" 3 (Array.length keys3);

  (* Check keys are different *)
  A.check A.bool "split keys are different" true
    (Rng.to_int keys.(0) <> Rng.to_int keys.(1));

  (* Check deterministic *)
  let keys2 = Rng.split key in
  A.check A.int "split is deterministic"
    (Rng.to_int keys.(0))
    (Rng.to_int keys2.(0))

let test_fold_in () =
  let key = Rng.key 42 in
  let key1 = Rng.fold_in key 1 in
  let key2 = Rng.fold_in key 2 in
  let key1_again = Rng.fold_in key 1 in

  A.check A.bool "fold_in with different data produces different keys" true
    (Rng.to_int key1 <> Rng.to_int key2);
  A.check A.int "fold_in is deterministic" (Rng.to_int key1)
    (Rng.to_int key1_again)

let test_uniform () =
  let key = Rng.key 42 in
  let shape = [| 3; 4 |] in
  let t = Rng.uniform key Float32 shape in

  A.check (A.array A.int) "uniform produces correct shape" shape (Rune.shape t);

  (* Check values are in [0, 1) *)
  let values = Rune.to_array (Rune.reshape [| 12 |] t) in
  Array.iter
    (fun v ->
      A.check A.bool "uniform values in [0, 1)" true (v >= 0. && v < 1.))
    values;

  (* Check deterministic *)
  let t2 = Rng.uniform key Float32 shape in
  let is_equal = Rune.all (Rune.equal t t2) in
  let is_equal_val = Rune.to_array is_equal in
  A.check A.bool "uniform is deterministic" true (is_equal_val.(0) > 0)

let test_normal () =
  let key = Rng.key 42 in
  let shape = [| 100 |] in
  let t = Rng.normal key Float32 shape in

  A.check (A.array A.int) "normal produces correct shape" shape (Rune.shape t);

  (* Check roughly normal distribution (mean ~0, std ~1) *)
  let values = Rune.to_array t in
  let mean =
    Array.fold_left ( +. ) 0. values /. float_of_int (Array.length values)
  in
  let variance =
    Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.)) 0. values
    /. float_of_int (Array.length values)
  in
  let std = Stdlib.sqrt variance in

  A.check (A.float 0.2) "normal mean ~0" 0. mean;
  A.check (A.float 0.3) "normal std ~1" 1. std

let test_randint () =
  let key = Rng.key 42 in
  let shape = [| 10 |] in
  let t = Rng.randint key ~min:5 ~max:15 shape in

  A.check (A.array A.int) "randint produces correct shape" shape (Rune.shape t);

  (* Check values are in [min, max) *)
  let values = Rune.to_array t in
  Array.iter
    (fun v ->
      let v = Int32.to_int v in
      A.check A.bool "randint values in [5, 15)" true (v >= 5 && v < 15))
    values

let test_bernoulli () =
  let key = Rng.key 42 in
  let shape = [| 1000 |] in
  let p = 0.3 in
  let t = Rng.bernoulli key ~p shape in

  A.check (A.array A.int) "bernoulli produces correct shape" shape
    (Rune.shape t);

  (* Check proportion roughly matches p *)
  let values = Rune.to_array t in
  let ones =
    Array.fold_left (fun acc v -> acc + if v > 0 then 1 else 0) 0 values
  in
  let prop = float_of_int ones /. float_of_int (Array.length values) in
  A.check (A.float 0.05) "bernoulli proportion ~p" p prop

(* TODO: Enable when argsort works with lazy views let test_permutation () = let
   Rng.permutation key n in

   A.check (A.array A.int) "permutation has correct shape" [| n |] (Rune.shape
   perm);

   (* Check it's a valid permutation *) let values = Rune.to_array perm |>
   Array.map Int32.to_int in let sorted = Array.copy values in Array.sort
   compare sorted; Array.iteri (fun i v -> A.check A.int (Printf.sprintf
   "permutation contains %d" i) i v) sorted *)

(* TODO: Enable when argsort works with lazy views let test_shuffle () = let key
   let x = Rune.reshape [| 10; 1 |] x in let shuffled = Rng.shuffle key x in

   A.check (A.array A.int) "shuffle preserves shape" (Rune.shape x) (Rune.shape
   shuffled);

   (* Check all elements are still there *) let original = Rune.to_array
   (Rune.reshape [| 10 |] x) in let shuffled_vals = Rune.to_array (Rune.reshape
   [| 10 |] shuffled) in let sorted_orig = Array.copy original in let
   sorted_shuf = Array.copy shuffled_vals in Array.sort compare sorted_orig;
   Array.sort compare sorted_shuf; A.check (A.array (A.float 0.001)) "shuffle
   preserves elements" sorted_orig sorted_shuf;

   (* Check it's actually shuffled *) let is_equal = Rune.all (Rune.array_equal
   x shuffled) in let is_equal_val = Rune.to_array is_equal in let is_different
   = is_equal_val.(0) = 0 in A.check A.bool "shuffle changes order" true
   is_different *)

let test_truncated_normal () =
  let key = Rng.key 42 in
  let shape = [| 100 |] in
  let lower = -1.5 in
  let upper = 2.0 in
  let t = Rng.truncated_normal key Float32 ~lower ~upper shape in

  A.check (A.array A.int) "truncated_normal produces correct shape" shape
    (Rune.shape t);

  (* Check all values are within bounds *)
  let values = Rune.to_array t in
  Array.iter
    (fun v ->
      A.check A.bool
        (Printf.sprintf "truncated_normal values in [%.1f, %.1f]: %.3f" lower
           upper v)
        true
        (v >= lower && v <= upper))
    values

let test_rng_with_autodiff () =
  let key = Rng.key 42 in

  (* Create a simple function that uses RNG *)
  let f x =
    let noise = Rng.normal key Float32 (Rune.shape x) in
    Rune.add x noise
  in

  (* Test gradient computation *)
  let x = Rune.ones Float32 [| 3; 3 |] in
  let grad_fn = Rune.grad f in
  let dx = grad_fn x in

  (* Gradient of x should be ones (since we just add noise) *)
  let expected = Rune.ones Float32 [| 3; 3 |] in
  let is_equal = Rune.all (Rune.equal dx expected) in
  let is_equal_val = Rune.to_array is_equal in
  A.check A.bool "RNG works with autodiff" true (is_equal_val.(0) > 0)

let () =
  A.run "Rune.Rng"
    [
      ( "key",
        [
          A.test_case "creation" `Quick test_key_creation;
          A.test_case "splitting" `Quick test_key_splitting;
          A.test_case "fold_in" `Quick test_fold_in;
        ] );
      ( "sampling",
        [
          A.test_case "uniform" `Quick test_uniform;
          A.test_case "normal" `Quick test_normal;
          A.test_case "randint" `Quick test_randint;
          A.test_case "bernoulli" `Quick test_bernoulli;
          (* TODO: Enable when argsort works with lazy views *)
          (* A.test_case "permutation" `Quick test_permutation;
          A.test_case "shuffle" `Quick test_shuffle; *)
          A.test_case "truncated_normal" `Quick test_truncated_normal;
        ] );
      ("autodiff", [ A.test_case "rng_with_grad" `Quick test_rng_with_autodiff ]);
    ]
