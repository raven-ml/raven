(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectorizing maps. The oracle is the loop: [vmap' f x] must equal stacking [f]
   applied to each slice of [x] along the mapped axis. Composition with grad and
   jvp is what vmap exists for, so it gets its own group. *)

open Windtrap
open Rune_test_support.Support

let loop_map f x =
  let b = (Nx.shape x).(0) in
  Nx.stack ~axis:0 (List.init b (fun i -> f (Nx.slice [ Nx.I i ] x)))

let check_vmap ~msg f x =
  check_arr ~msg (to_arr (loop_map f x)) (Rune.vmap' f x)

(* Batched inputs: 4 rows of 3, and a batch of 2x3 matrices. *)
let xs () =
  Nx.create f64 [| 4; 3 |]
    [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let ms () =
  Nx.create f64 [| 2; 2; 3 |]
    [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let w32 () = mat64 3 2 [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5 |]

(* Semantics against the loop oracle *)

let oracle_tests =
  [
    test "elementwise chain" (fun () ->
        check_vmap ~msg:"exp(sin x) + x^2"
          (fun r -> Nx.add (Nx.exp (Nx.sin r)) (Nx.mul r r))
          (xs ()));
    test "closure constants broadcast" (fun () ->
        let c = vec64 [| 2.0; -1.0; 0.5 |] in
        check_vmap ~msg:"x * c" (fun r -> Nx.mul r c) (xs ()));
    test "scalar closure constant" (fun () ->
        check_vmap ~msg:"x + 3" (fun r -> Nx.add_s r 3.0) (xs ()));
    test "full reduction" (fun () ->
        check_vmap ~msg:"sum" (fun r -> Nx.sum r) (xs ()));
    test "centering uses the unbatched mean" (fun () ->
        check_vmap ~msg:"x - mean x"
          (fun r -> Nx.sub r (Nx.mean r ~keepdims:true))
          (xs ()));
    test "axis reduction on matrix elements" (fun () ->
        check_vmap ~msg:"sum axis0" (fun m -> Nx.sum ~axes:[ 0 ] m) (ms ()));
    test "max reduction" (fun () ->
        check_vmap ~msg:"max" (fun r -> Nx.max r ~keepdims:true) (xs ()));
    test "vector-matrix multiply" (fun () ->
        let w = w32 () in
        check_vmap ~msg:"r @ w" (fun r -> Nx.matmul r w) (xs ()));
    test "matrix-matrix multiply" (fun () ->
        let w = w32 () in
        check_vmap ~msg:"m @ w" (fun m -> Nx.matmul m w) (ms ()));
    test "reshape and transpose" (fun () ->
        check_vmap ~msg:"transpose (reshape m)"
          (fun m -> Nx.transpose (Nx.reshape [| 3; 2 |] m))
          (ms ()));
    test "where selects per element" (fun () ->
        check_vmap ~msg:"relu"
          (fun r ->
            Nx.where (Nx.greater r (Nx.zeros_like r)) r (Nx.zeros_like r))
          (xs ()));
    test "sort" (fun () ->
        check_vmap ~msg:"sort" (fun r -> fst (Nx.sort ~axis:0 r)) (xs ()));
    test "cumsum" (fun () ->
        check_vmap ~msg:"cumsum" (fun r -> Nx.cumsum ~axis:0 r) (xs ()));
    test "concatenate with itself" (fun () ->
        check_vmap ~msg:"cat" (fun r -> Nx.concatenate ~axis:0 [ r; r ]) (xs ()));
    test "pad" (fun () ->
        check_vmap ~msg:"pad" (fun r -> Nx.pad [| (1, 1) |] 9.0 r) (xs ()));
    test "slice" (fun () ->
        check_vmap ~msg:"slice" (fun r -> Nx.slice [ Nx.R (1, 3) ] r) (xs ()));
    test "take_along_axis with constant indices" (fun () ->
        let idx = Nx.create Nx.int32 [| 2 |] [| 2l; 0l |] in
        check_vmap ~msg:"gather"
          (fun r -> Nx.take_along_axis ~axis:0 idx r)
          (xs ()));
    test "softmax composite" (fun () ->
        check_vmap ~msg:"softmax"
          (fun r ->
            let e = Nx.exp r in
            Nx.div e (Nx.sum e ~keepdims:true))
          (xs ()));
    test "constant output broadcasts" (fun () ->
        check_vmap ~msg:"const" (fun _ -> Nx.scalar f64 7.0) (xs ()));
  ]

(* Axes and structure *)

let test_in_axis () =
  let x =
    Nx.transpose (xs ())
    (* [3; 4], mapped axis 1 *)
  in
  check_arr ~msg:"in_axis 1"
    (to_arr (loop_map (fun r -> Nx.sum (Nx.mul r r)) (xs ())))
    (Rune.vmap' ~in_axis:1 (fun r -> Nx.sum (Nx.mul r r)) x)

let test_out_axis () =
  let y = Rune.vmap' ~out_axis:1 (fun r -> Nx.mul r r) (xs ()) in
  equal ~msg:"shape" (array int) [| 3; 4 |] (Nx.shape y);
  check_arr ~msg:"values"
    (to_arr (Nx.transpose (loop_map (fun r -> Nx.mul r r) (xs ()))))
    y

let test_vmap_structure () =
  (* Two mapped leaves: per-slice matrix products. *)
  let a = ms () in
  let b =
    Nx.create f64 [| 2; 3; 2 |]
      [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5; 0.9; -0.2; 0.7; 1.4; -0.3; 0.5 |]
  in
  let y =
    Rune.vmap
      (module Pair)
      (fun p -> Nx.matmul p.fst p.snd)
      { fst = a; snd = b }
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i ->
           Nx.matmul (Nx.slice [ Nx.I i ] a) (Nx.slice [ Nx.I i ] b)))
  in
  check_arr ~msg:"pair matmul" (to_arr expected) y

let test_in_axes_constant_leaf () =
  (* Second leaf held constant: per-slice a_i @ b. *)
  let a = ms () in
  let b = w32 () in
  let y =
    Rune.vmap ~in_axes:[ Some 0; None ]
      (module Pair)
      (fun p -> Nx.matmul p.fst p.snd)
      { fst = a; snd = b }
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i -> Nx.matmul (Nx.slice [ Nx.I i ] a) b))
  in
  check_arr ~msg:"constant leaf" (to_arr expected) y

let test_in_axes_non_leading () =
  (* First leaf mapped along axis 1. *)
  let a =
    Nx.moveaxis 0 1 (ms ())
    (* batch now at axis 1 *)
  in
  let b = w32 () in
  let y =
    Rune.vmap ~in_axes:[ Some 1; None ]
      (module Pair)
      (fun p -> Nx.matmul p.fst p.snd)
      { fst = a; snd = b }
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i -> Nx.matmul (Nx.slice [ Nx.I i ] (ms ())) b))
  in
  check_arr ~msg:"axis 1" (to_arr expected) y

let test_in_axes_negative_axis () =
  (* [Some (-2)] names the same axis as [Some 1] on a rank-3 leaf. *)
  let a = Nx.moveaxis 0 1 (ms ()) in
  let b = w32 () in
  let y =
    Rune.vmap ~in_axes:[ Some (-2); None ]
      (module Pair)
      (fun p -> Nx.matmul p.fst p.snd)
      { fst = a; snd = b }
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i -> Nx.matmul (Nx.slice [ Nx.I i ] (ms ())) b))
  in
  check_arr ~msg:"negative axis" (to_arr expected) y

let test_in_axes_out_of_bounds () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vmap ~in_axes:[ Some 2; None ]
           (module Pair)
           (fun p -> Nx.add p.fst p.snd)
           { fst = xs (); snd = xs () }))

let test_in_axes_length_mismatch () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vmap ~in_axes:[ Some 0 ]
           (module Pair)
           (fun p -> Nx.add p.fst p.snd)
           { fst = xs (); snd = xs () }))

let test_in_axes_maps_no_leaf () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vmap ~in_axes:[ None; None ]
           (module Pair)
           (fun p -> Nx.add p.fst p.snd)
           { fst = xs (); snd = xs () }))

let test_structural_out_axis () =
  let y =
    Rune.vmap ~out_axis:1
      (module Pair)
      (fun p -> Nx.add p.fst p.snd)
      { fst = xs (); snd = xs () }
  in
  equal ~msg:"shape" (array int) [| 3; 4 |] (Nx.shape y);
  check_arr ~msg:"values" (to_arr (Nx.transpose (Nx.add (xs ()) (xs ())))) y

let test_batch_size_mismatch () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vmap
           (module Pair)
           (fun p -> Nx.add p.fst p.snd)
           { fst = vec64 [| 1.0; 2.0 |]; snd = vec64 [| 1.0; 2.0; 3.0 |] }))

let test_scalar_leaf_rejected () =
  raises_invalid_arg (fun () ->
      ignore (Rune.vmap' (fun x -> x) (Nx.scalar f64 1.0)))

let test_reading_batched_value_raises () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vmap'
           (fun r ->
             (* Concretizing a batched tensor would expose the physical batched
                buffer. *)
             let (_ : float) = Nx.item [ 0 ] r in
             r)
           (xs ())))

let test_reading_constant_value_is_fine () =
  let c = vec64 [| 2.0 |] in
  let y = Rune.vmap' (fun r -> Nx.mul_s r (Nx.item [ 0 ] c)) (xs ()) in
  check_arr ~msg:"constant read" (to_arr (Nx.mul_s (xs ()) 2.0)) y

let test_no_rule_raises () =
  raises_invalid_arg (fun () ->
      ignore
        (Rune.vmap'
           (fun m -> Nx.cholesky m)
           (Nx.create f64 [| 2; 2; 2 |]
              [| 4.0; 1.0; 1.0; 3.0; 5.0; 0.5; 0.5; 2.0 |])))

let test_vmap2_structured_output () =
  (* Both output leaves gain a batch axis; one depends on the input, the other
     is constant and broadcasts. *)
  let c = vec64 [| 9.0 |] in
  let y =
    Rune.vmap2
      (module Pair)
      (module Pair)
      (fun p -> { fst = Nx.mul p.fst p.snd; snd = c })
      { fst = xs (); snd = xs () }
  in
  check_arr ~msg:"fst" (to_arr (Nx.mul (xs ()) (xs ()))) y.fst;
  equal ~msg:"snd shape" (array int) [| 4; 1 |] (Nx.shape y.snd);
  check_arr ~msg:"snd" [| 9.0; 9.0; 9.0; 9.0 |] y.snd

let test_rng_is_identical_per_lane () =
  (* Implicit RNG keys are constants of the map: every lane draws the same
     values. Pinned as documented behavior until nx grows tensor-typed keys;
     thread distinct randomness in as mapped inputs instead. *)
  let y =
    Nx.Rng.run ~seed:42 (fun () ->
        Rune.vmap' (fun r -> Nx.add r (Nx.rand f64 [| 3 |])) (xs ()))
  in
  let base = Nx.sub y (xs ()) in
  let row i = to_arr (Nx.slice [ Nx.I i ] base) in
  equal ~msg:"lanes share draws" (array (float 1e-12)) (row 0) (row 1)

(* Nesting *)

let test_nested_vmap () =
  let x = ms () in
  let f r = Nx.sum (Nx.mul r r) in
  let y = Rune.vmap' (Rune.vmap' f) x in
  let expected =
    Nx.stack ~axis:0 (List.init 2 (fun i -> loop_map f (Nx.slice [ Nx.I i ] x)))
  in
  check_arr ~msg:"nested" (to_arr expected) y

(* Composition with differentiation *)

let test_per_sample_gradients () =
  (* vmap of grad: gradient of sum(x²) per row is 2x. *)
  let f x = Nx.sum (Nx.mul x x) in
  let g = Rune.vmap' (fun x -> Rune.grad' f x) (xs ()) in
  check_arr ~msg:"per-sample grads"
    (Array.map (fun v -> 2.0 *. v) (to_arr (xs ())))
    g

let test_grad_through_vmap () =
  (* grad of vmap: d/dx sum_i sum(x_i²) = 2x. *)
  let f x = Nx.sum (Rune.vmap' (fun r -> Nx.sum (Nx.mul r r)) x) in
  let g = Rune.grad' f (xs ()) in
  check_arr ~msg:"grad through vmap"
    (Array.map (fun v -> 2.0 *. v) (to_arr (xs ())))
    g

let test_per_sample_gradients_of_gather () =
  (* The gather gradient scatter-adds its cotangent; under vmap the scatter
     effect must carry its Add mode or the per-sample gradients collapse to a
     Set-mode scatter. *)
  let idx = Nx.create Nx.int32 [| 2 |] [| 1l; 1l |] in
  let f x =
    let gathered = Nx.take_along_axis ~axis:0 idx x in
    Nx.sum (Nx.mul gathered gathered)
  in
  let g = Rune.vmap' (fun x -> Rune.grad' f x) (xs ()) in
  let expected =
    Nx.stack ~axis:0
      (List.init 4 (fun i -> Rune.grad' f (Nx.slice [ Nx.I i ] (xs ()))))
  in
  (* Row element 1 is gathered twice: its gradient is 2. *)
  check_arr ~msg:"per-sample gather grads" (to_arr expected) g

let test_jvp_through_vmap () =
  (* jvp of vmap of sum(x²) along v: per row, 2 <x_i, v_i>. *)
  let f x = Rune.vmap' (fun r -> Nx.sum (Nx.mul r r)) x in
  let v = tangent_like (xs ()) in
  let _, dy = Rune.jvp' f (xs ()) v in
  let expected = Nx.sum ~axes:[ 1 ] (Nx.mul_s (Nx.mul (xs ()) v) 2.0) in
  check_arr ~msg:"jvp through vmap" (to_arr expected) dy

let tests =
  [
    group "loop oracle" oracle_tests;
    group "axes and structure"
      [
        test "maps a non-leading axis" test_in_axis;
        test "places the batch axis on output" test_out_axis;
        test "maps all leaves of a structure" test_vmap_structure;
        test "in_axes holds a leaf constant" test_in_axes_constant_leaf;
        test "in_axes maps a non-leading axis" test_in_axes_non_leading;
        test "in_axes accepts a negative axis" test_in_axes_negative_axis;
        test "in_axes rejects an out-of-bounds axis" test_in_axes_out_of_bounds;
        test "in_axes must have one entry per leaf" test_in_axes_length_mismatch;
        test "in_axes must map at least one leaf" test_in_axes_maps_no_leaf;
        test "out_axis places the structural batch axis"
          test_structural_out_axis;
        test "rejects mismatched batch sizes" test_batch_size_mismatch;
        test "rejects scalar leaves" test_scalar_leaf_rejected;
        test "raises without a batching rule" test_no_rule_raises;
        test "reading a batched value raises" test_reading_batched_value_raises;
        test "reading a constant value is fine"
          test_reading_constant_value_is_fine;
      ];
    group "nesting" [ test "vmap of vmap" test_nested_vmap ];
    group "randomness"
      [
        test "implicit RNG draws are identical per lane"
          test_rng_is_identical_per_lane;
      ];
    group "structured outputs"
      [ test "vmap2 batches every output leaf" test_vmap2_structured_output ];
    group "composition"
      [
        test "vmap of grad: per-sample gradients" test_per_sample_gradients;
        test "per-sample gradients of a gather"
          test_per_sample_gradients_of_gather;
        test "grad of vmap" test_grad_through_vmap;
        test "jvp of vmap" test_jvp_through_vmap;
      ];
  ]

let () = run "rune vmap" tests
