(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

let values_are ?msg ~tol expected t =
  let ft = if tol = 0. then float_exact else float tol in
  equal ?msg (array ft) expected (Nx.to_array t)

let pos_of rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int32 [| batch; seq |]
    (Array.concat (Array.to_list rows) |> Array.map Int32.of_int)

let test_standard_frequencies () =
  equal ~msg:"theta ** (-2 i / head_dim)"
    (array (float 1e-12))
    [| 1.0; 0.1; 0.01 |]
    (Rope.frequencies (Rope.make ~theta:1000.0 ~head_dim:6 ()))

(* The Llama 3.1 schedule on a head of 8: the two highest frequencies are kept,
   the lowest is divided by the factor, the one between is blended. *)
let test_llama3_frequencies () =
  let f =
    Rope.frequencies
      (Rope.llama3 ~theta:500000.0 ~head_dim:8 ~factor:8.0 ~low_freq_factor:1.0
         ~high_freq_factor:4.0 ~original_context:8192)
  in
  let base = Rope.frequencies (Rope.make ~theta:500000.0 ~head_dim:8 ()) in
  equal ~msg:"pair 0 kept" (float 1e-12) base.(0) f.(0);
  equal ~msg:"pair 1 kept" (float 1e-12) base.(1) f.(1);
  equal ~msg:"pair 3 divided by the factor" (float 1e-15)
    (base.(3) /. 8.0)
    f.(3);
  (* Computed independently from the published formula. *)
  equal ~msg:"pair 2 blended" (float 1e-15) 0.0005248461609929547 f.(2);
  equal ~msg:"pair 3" (float 1e-18) 6.647869871181235e-06 f.(3);
  is_true ~msg:"the blend lies between the two bands"
    (f.(2) > base.(2) /. 8.0 && f.(2) < base.(2))

let test_position_zero_is_identity () =
  let x = Nx.create Nx.float32 [| 1; 1; 2; 4 |] (Array.init 8 float_of_int) in
  values_are ~msg:"unrotated" ~tol:0.0
    (Array.init 8 float_of_int)
    (Rope.apply (Rope.make ~head_dim:4 ()) ~pos:(pos_of [| [| 0; 0 |] |]) x)

(* One pair with frequency 1: position p turns [1; 0] to [cos p; sin p], pairing
   feature 0 with feature 1. *)
let test_rotation () =
  let t = Rope.make ~theta:1.0 ~head_dim:2 () in
  let x = Nx.create Nx.float32 [| 1; 1; 2; 2 |] [| 1.; 0.; 0.; 1. |] in
  values_are ~msg:"rotated pairs" ~tol:1e-6
    [| cos 1.; sin 1.; -.sin 2.; cos 2. |]
    (Rope.apply t ~pos:(pos_of [| [| 1; 2 |] |]) x)

let test_pairs_first_half_with_second () =
  (* head_dim 4 at theta 1: both pairs turn by the position. Feature 0 pairs
     with feature 2, not with feature 1. *)
  let t = Rope.make ~theta:1.0 ~head_dim:4 () in
  let x = Nx.create Nx.float32 [| 1; 1; 1; 4 |] [| 1.; 0.; 0.; 0. |] in
  values_are ~msg:"feature 0 rotates into feature 2" ~tol:1e-6
    [| cos 1.; 0.; sin 1.; 0. |]
    (Rope.apply t ~pos:(pos_of [| [| 1 |] |]) x)

(* What attention needs: the product of a rotated query and a rotated key
   depends on the positions only through their difference. *)
let test_relative () =
  Nx.Rng.with_key (Nx.Rng.key 21) @@ fun () ->
  let t = Rope.make ~head_dim:8 () in
  let q = Nx.randn Nx.float32 [| 1; 1; 1; 8 |] in
  let k = Nx.randn Nx.float32 [| 1; 1; 1; 8 |] in
  let dot pq pk =
    let at p x = Rope.apply t ~pos:(pos_of [| [| p |] |]) x in
    Nx.item [] (Nx.sum (Nx.mul (at pq q) (at pk k)))
  in
  equal ~msg:"shifted by 100" (float 1e-4) (dot 7 3) (dot 107 103)

let test_per_row_positions () =
  let t = Rope.make ~theta:1.0 ~head_dim:2 () in
  let x = Nx.create Nx.float32 [| 2; 1; 1; 2 |] [| 1.; 0.; 1.; 0. |] in
  values_are ~msg:"each row at its own position" ~tol:1e-6
    [| cos 3.; sin 3.; cos 5.; sin 5. |]
    (Rope.apply t ~pos:(pos_of [| [| 3 |]; [| 5 |] |]) x)

(* At bfloat16 a long position times a small frequency is not representable; the
   angle is formed at float32 and only the rotation runs at bfloat16. *)
let test_half_precision_angles () =
  let t = Rope.make ~head_dim:2 ~theta:1.0 () in
  let x = Nx.create Nx.bfloat16 [| 1; 1; 1; 2 |] [| 1.; 0. |] in
  let y = Rope.apply t ~pos:(pos_of [| [| 100003 |] |]) x in
  equal ~msg:"dtype preserved" bool true (Nx.dtype y = Nx.bfloat16);
  values_are ~msg:"rotated by the float32 angle" ~tol:2e-2
    [| cos 100003.; sin 100003. |]
    (Nx.cast Nx.float32 y)

module One = struct
  type 'a t = 'a

  let map f x = f x
  let map2 f x y = f x y
  let iter f x = f x
end

let test_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 22) @@ fun () ->
  let t = Rope.make ~head_dim:4 () in
  let pos = pos_of [| [| 2; 5; 9 |] |] in
  let w = Nx.randn Nx.float64 [| 1; 2; 3; 4 |] in
  match
    Rune.check_grads
      (Kaun.ptree (module One))
      (fun x -> Nx.sum (Nx.mul w (Rope.apply t ~pos x)))
      (Nx.randn Nx.float64 [| 1; 2; 3; 4 |])
  with
  | Ok () -> ()
  | Error m -> fail m

let test_jit_matches_eager () =
  Nx.Rng.with_key (Nx.Rng.key 23) @@ fun () ->
  let t = Rope.make ~head_dim:4 () in
  let pos = pos_of [| [| 2; 5; 9 |] |] in
  let x = Nx.randn Nx.float32 [| 1; 2; 3; 4 |] in
  let f x = Rope.apply t ~pos x in
  values_are ~msg:"compiled rotation" ~tol:1e-5
    (Nx.to_array (f x))
    (Rune.jit' f x)

let test_rejects_bad_input () =
  raises
    (Invalid_argument "Rope.make: head_dim must be positive and even, got 3")
    (fun () -> Rope.make ~head_dim:3 ());
  raises
    (Invalid_argument
       "Rope.apply: x has head_dim 6 but the frequencies are for 4") (fun () ->
      Rope.apply (Rope.make ~head_dim:4 ()) ~pos:(pos_of [| [| 0 |] |])
        (Nx.zeros Nx.float32 [| 1; 1; 1; 6 |]));
  raises (Invalid_argument "Rope.apply: pos must have shape [2; 3] or [1; 3]")
    (fun () ->
      Rope.apply (Rope.make ~head_dim:4 ())
        ~pos:(pos_of [| [| 0; 1 |] |])
        (Nx.zeros Nx.float32 [| 2; 1; 3; 4 |]))

let () =
  run "kaun rope"
    [
      group "schedules"
        [
          test "standard frequencies" test_standard_frequencies;
          test "llama 3 bands" test_llama3_frequencies;
        ];
      group "rotation"
        [
          test "position zero is the identity" test_position_zero_is_identity;
          test "a pair turns by position times frequency" test_rotation;
          test "feature i pairs with i + head_dim / 2"
            test_pairs_first_half_with_second;
          test "products depend on the position difference" test_relative;
          test "each row has its own positions" test_per_row_positions;
          test "angles are formed at float32" test_half_precision_angles;
          test "gradients agree with finite differences" test_gradients;
          test "compiles to the eager result" test_jit_matches_eager;
          test "invalid inputs are rejected" test_rejects_bad_input;
        ];
    ]
