(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectorizing maps. The oracle is the loop: [vmap' f x] must equal stacking [f]
   applied to each slice of [x] along the mapped axis. Composition with grad and
   jvp is what vmap exists for, so it gets its own group. *)

open Windtrap
open Rune_test_support.Support

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
    test "bitcast" (fun () ->
        check_vmap ~msg:"the bits of each row"
          (fun r -> Nx.cast f64 (Nx.bitcast Nx.int64 r))
          (xs ()));
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
    test "matmul against a constant with its own batch dimensions" (fun () ->
        (* The constant's leading dimension must not be taken for the map's:
           each row meets both matrices of [ws]. *)
        let ws =
          Nx.create f64 [| 2; 3; 2 |]
            [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5; 0.2; -0.9; 1.3; 0.5; -0.6; 0.7 |]
        in
        check_vmap ~msg:"r @ ws" (fun r -> Nx.matmul r ws) (xs ());
        check_vmap ~msg:"m @ ws" (fun m -> Nx.matmul m ws) (ms ()));
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
    test "sliding windows" (fun () ->
        check_vmap ~msg:"sliding window"
          (fun r -> sliding_window ~axis:0 ~window:2 ~step:1 r)
          (xs ()));
    test "sliding windows on a leading axis" (fun () ->
        check_vmap ~msg:"sliding window axis 0"
          (fun m -> sliding_window ~axis:0 ~window:2 ~step:1 m)
          (ms ()));
    test "extract_patches" (fun () ->
        check_vmap ~msg:"extract_patches"
          (fun m ->
            Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
              ~dilation:[| 1 |]
              ~padding:[| (0, 0) |]
              m)
          (ms ()));
    test "combine_patches" (fun () ->
        check_vmap ~msg:"combine_patches"
          (fun m ->
            Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
              ~dilation:[| 1 |]
              ~padding:[| (0, 0) |]
              m
            |> Nx.combine_patches ~output_size:[| 3 |] ~kernel_size:[| 2 |]
                 ~stride:[| 1 |] ~dilation:[| 1 |]
                 ~padding:[| (0, 0) |])
          (ms ()));
    test "slice" (fun () ->
        check_vmap ~msg:"slice" (fun r -> Nx.slice [ Nx.R (1, 3) ] r) (xs ()));
    test "take_along_axis with constant indices" (fun () ->
        let idx = Nx.create Nx.int64 [| 2 |] [| 2L; 0L |] in
        check_vmap ~msg:"gather"
          (fun r -> Nx.take_along_axis ~axis:0 ~indices:idx r)
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

let test_moved_axis () =
  (* Another axis is mapped by moving it to the front, a view. *)
  let x = Nx.transpose (xs ()) in
  check_arr ~msg:"axis 1"
    (to_arr (loop_map (fun r -> Nx.sum (Nx.mul r r)) (xs ())))
    (Rune.vmap' (fun r -> Nx.sum (Nx.mul r r)) (Nx.moveaxis 1 0 x))

let test_vmap_structure () =
  (* Two mapped leaves: per-slice matrix products. *)
  let a = ms () in
  let b =
    Nx.create f64 [| 2; 3; 2 |]
      [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5; 0.9; -0.2; 0.7; 1.4; -0.3; 0.5 |]
  in
  let y =
    Rune.vmap
      Nx.Ptree.(pair_ptree @-> returns tensor)
      (fun p -> Nx.matmul p.fst p.snd)
      { fst = a; snd = b }
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i ->
           Nx.matmul (Nx.slice [ Nx.I i ] a) (Nx.slice [ Nx.I i ] b)))
  in
  check_arr ~msg:"pair matmul" (to_arr expected) y;
  let y' =
    Rune.vmap Nx.Ptree.(tensor @-> tensor @-> returns tensor) Nx.matmul a b
  in
  check_arr ~msg:"curried matmul" (to_arr expected) y'

let test_vmap_structure_leading_dims () =
  (* Two mapped leaves whose elements carry different leading ranks: a batch of
     matrix stacks against a batch of single matrices. The sizes coincide with
     the map's, so misaligning the batch axis would pair the wrong matrices
     without any shape error. *)
  let a =
    Nx.create f64 [| 2; 2; 2; 3 |]
      [|
        0.5;
        -1.2;
        2.1;
        1.7;
        -0.4;
        0.9;
        0.2;
        1.3;
        -0.7;
        0.8;
        -1.6;
        0.4;
        -0.3;
        0.6;
        1.1;
        0.9;
        -0.5;
        0.2;
        1.4;
        -0.8;
        0.3;
        -1.1;
        0.7;
        0.1;
      |]
  in
  let b =
    Nx.create f64 [| 2; 3; 2 |]
      [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5; 0.9; -0.2; 0.7; 1.4; -0.3; 0.5 |]
  in
  let y =
    Rune.vmap
      Nx.Ptree.(pair_ptree @-> returns tensor)
      (fun p -> Nx.matmul p.fst p.snd)
      { fst = a; snd = b }
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i ->
           Nx.matmul (Nx.slice [ Nx.I i ] a) (Nx.slice [ Nx.I i ] b)))
  in
  check_arr ~msg:"pair matmul, leading dims" (to_arr expected) y

let test_captured_value_is_constant () =
  (* A value the function captures is not mapped: per-slice a_i @ b. *)
  let a = ms () in
  let b = w32 () in
  let y =
    Rune.vmap Nx.Ptree.(tensor @-> returns tensor) (fun a -> Nx.matmul a b) a
  in
  let expected =
    Nx.stack ~axis:0
      (List.init 2 (fun i -> Nx.matmul (Nx.slice [ Nx.I i ] a) b))
  in
  check_arr ~msg:"constant" (to_arr expected) y

(* A capture that is the same value as an argument is a constant: each lane adds
   the whole of [w]. *)
let test_capture_of_the_argument_is_constant () =
  let w = vec64 [| 1.0; 2.0; 3.0 |] in
  let y =
    Rune.vmap Nx.Ptree.(tensor @-> returns tensor) (fun x -> Nx.add x w) w
  in
  equal ~msg:"shape" (array int) [| 3; 3 |] (Nx.shape y);
  check_arr ~msg:"lanes" [| 2.0; 3.0; 4.0; 3.0; 4.0; 5.0; 4.0; 5.0; 6.0 |] y;
  check_arr ~msg:"vmap'"
    [| 2.0; 3.0; 4.0; 3.0; 4.0; 5.0; 4.0; 5.0; 6.0 |]
    (Rune.vmap' (fun x -> Nx.add x w) w)

let test_rejects_no_leaf () =
  raises (Invalid_argument "Rune.vmap: the arguments have no leaf to map")
    (fun () ->
      ignore (Rune.vmap Nx.Ptree.(unit @-> returns tensor) (fun () -> xs ()) ()))

let test_rejects_consumes () =
  raises
    (Invalid_argument
       "Rune.vmap: the argument at 0 is consumed; only a compiled call \
        consumes its arguments") (fun () ->
      let (_ : Nx.float64_t -> Nx.float64_t) =
        Rune.vmap Nx.Ptree.(consumes tensor @@ returns tensor) Fun.id
      in
      ())

let test_batch_size_mismatch () =
  raises (Invalid_argument "Rune.vmap: 0.snd: 3 rows along axis 0, 0.fst: 2")
    (fun () ->
      ignore
        (Rune.vmap
           Nx.Ptree.(pair_ptree @-> returns tensor)
           (fun p -> Nx.add p.fst p.snd)
           { fst = vec64 [| 1.0; 2.0 |]; snd = vec64 [| 1.0; 2.0; 3.0 |] }));
  raises (Invalid_argument "Rune.vmap: 1: 3 rows along axis 0, 0: 2") (fun () ->
      ignore
        (Rune.vmap
           Nx.Ptree.(tensor @-> tensor @-> returns tensor)
           Nx.add
           (vec64 [| 1.0; 2.0 |])
           (vec64 [| 1.0; 2.0; 3.0 |])))

let test_scalar_leaf_rejected () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rune.vmap' (fun x -> x) (Nx.scalar f64 1.0)));
  raises
    (Invalid_argument "Rune.vmap: 0.1: a scalar; vmap maps axis 0 of every leaf")
    (fun () ->
      ignore
        (Rune.vmap
           Nx.Ptree.(pair tensor tensor @-> returns tensor)
           (fun (x, y) -> Nx.add x y)
           (xs (), Nx.scalar f64 1.0)))

let test_reading_batched_value_raises () =
  raises_match Exn.invalid_arg (fun () ->
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
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.vmap'
           (fun m -> Nx.cholesky m)
           (Nx.create f64 [| 2; 2; 2 |]
              [| 4.0; 1.0; 1.0; 3.0; 5.0; 0.5; 0.5; 2.0 |])))

let test_vmap_structured_output () =
  (* Both output leaves gain a batch axis; one depends on the input, the other
     is constant and broadcasts. *)
  let c = vec64 [| 9.0 |] in
  let y =
    Rune.vmap
      Nx.Ptree.(pair_ptree @-> returns pair_ptree)
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
    Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
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
  let idx = Nx.create Nx.int64 [| 2 |] [| 1L; 1L |] in
  let f x =
    let gathered = Nx.take_along_axis ~axis:0 ~indices:idx x in
    Nx.sum (Nx.mul gathered gathered)
  in
  let g = Rune.vmap' (fun x -> Rune.grad' f x) (xs ()) in
  let expected =
    Nx.stack ~axis:0
      (List.init 4 (fun i -> Rune.grad' f (Nx.slice [ Nx.I i ] (xs ()))))
  in
  (* Row element 1 is gathered twice: its gradient is 2. *)
  check_arr ~msg:"per-sample gather grads" (to_arr expected) g

let test_per_sample_gradients_of_sliding_window () =
  (* The sliding-window adjoint overlap-adds through [fold], so [fold] has to
     batch or per-sample gradients of any windowed loss stop working. *)
  let f x =
    Nx.sum
      (Nx.mul
         (sliding_window ~axis:0 ~window:2 ~step:1 x)
         (sliding_window ~axis:0 ~window:2 ~step:1 x))
  in
  let g = Rune.vmap' (fun x -> Rune.grad' f x) (xs ()) in
  let expected =
    Nx.stack ~axis:0
      (List.init 4 (fun i -> Rune.grad' f (Nx.slice [ Nx.I i ] (xs ()))))
  in
  check_arr ~msg:"per-sample windowed grads" (to_arr expected) g

let test_jvp_through_vmap () =
  (* jvp of vmap of sum(x²) along v: per row, 2 <x_i, v_i>. *)
  let f x = Rune.vmap' (fun r -> Nx.sum (Nx.mul r r)) x in
  let v = tangent_like (xs ()) in
  let _, dy = Rune.jvp' f (xs ()) v in
  let expected = Nx.sum ~axes:[ 1 ] (Nx.mul_s (Nx.mul (xs ()) v) 2.0) in
  check_arr ~msg:"jvp through vmap" (to_arr expected) dy

(* A window write batches over the template and the value; each row gets its own
   window. *)
module Row_pos = struct
  type row_pos = { row : Nx.float32_t; pos : Nx.int64_t }
  type _ t = row_pos

  let walk c { row; pos } =
    let open Nx.Ptree.Walk in
    let row = field c "row" tensor row in
    let pos = field c "pos" tensor pos in
    { row; pos }
end

let row_pos = Nx.Ptree.(instantiate (module Row_pos) @-> returns tensor)

(* A batched window start: each example writes and reads at its own clamped
   position. *)
let test_vmap_set_window_batched_start () =
  let xs = Nx.create f32 [| 2; 4 |] [| 0.; 1.; 2.; 3.; 10.; 11.; 12.; 13. |] in
  let pos = Nx.create Nx.int64 [| 2 |] [| 1L; 5L |] in
  let v = vec32 [| 9.0; 8.0 |] in
  check_arr ~msg:"per-example windows, the second clamped"
    [| 0.; 9.; 8.; 3.; 10.; 11.; 9.; 8. |]
    (Rune.vmap row_pos
       (fun r -> Nx.set [ Nx.D (r.pos, 2) ] v r.row)
       { row = xs; pos });
  check_arr ~msg:"per-example reads" [| 1.; 2.; 12.; 13. |]
    (Rune.vmap row_pos
       (fun r -> Nx.slice [ Nx.D (r.pos, 2) ] r.row)
       { row = xs; pos })

let test_vmap_set_window () =
  let xs = Nx.create f32 [| 2; 4 |] [| 0.; 1.; 2.; 3.; 10.; 11.; 12.; 13. |] in
  let v = vec32 [| 9.0; 8.0 |] in
  let f row = Nx.set [ Nx.R (1, 3) ] v row in
  check_arr ~msg:"per-row windows"
    [| 0.; 9.; 8.; 3.; 10.; 9.; 8.; 13. |]
    (Rune.vmap' f xs);
  let vs = Nx.create f32 [| 2; 2 |] [| 9.; 8.; 7.; 6. |] in
  check_arr ~msg:"batched values over one template row"
    [| 0.; 9.; 8.; 3.; 0.; 7.; 6.; 3. |]
    (Rune.vmap' (fun v -> Nx.set [ Nx.R (1, 3) ] v (Nx.get [ 0 ] xs)) vs)

(* Lanes *)

let lane i x = Nx.slice [ Nx.I i ] x
let stack n f = Nx.stack ~axis:0 (List.init n f)

(* The map a gather names answers it with every lane's value, the same in each
   lane: [x_i . x_j] for every [j] is row [i] of [xs xsᵀ]. *)
let test_lanes_of_the_named_map () =
  let a = Rune.axis () and x = xs () in
  let y = Rune.vmap' ~axis:a (fun r -> Nx.matmul (Rune.lanes a r) r) x in
  check_arr ~msg:"xs xsᵀ" (to_arr (Nx.matmul x (Nx.transpose x))) y;
  let y = Rune.vmap' ~axis:a (fun r -> Rune.lanes a r) x in
  equal ~msg:"shape" (array int) [| 4; 4; 3 |] (Nx.shape y);
  check_arr ~msg:"every lane holds every row" (to_arr (stack 4 (fun _ -> x))) y

(* A value every lane shares is gathered as its copies. *)
let test_lanes_of_a_shared_value () =
  let a = Rune.axis () and c = vec64 [| 1.0; -2.0; 0.5 |] in
  let y = Rune.vmap' ~axis:a (fun r -> Nx.add (Rune.lanes a c) r) (xs ()) in
  check_arr ~msg:"c + x_i in every row"
    (to_arr
       (stack 4 (fun i ->
            Nx.add (Nx.broadcast_to [| 4; 3 |] c) (lane i (xs ())))))
    y

(* A map between the gather and the map it names keeps its own lanes, whether it
   is anonymous or has another name: lane [(i, j)] gathers [x_(i', j)] over
   [i']. *)
let test_lanes_through_another_map () =
  let a = Rune.axis () and b = Rune.axis () in
  let x =
    Nx.create f64 [| 2; 3; 4 |] (Array.init 24 (fun i -> Float.of_int i))
  in
  let y = Rune.vmap' ~axis:a (Rune.vmap' (fun r -> Rune.lanes a r)) x in
  let expected =
    stack 2 (fun _ -> stack 3 (fun j -> stack 2 (fun i -> lane j (lane i x))))
  in
  equal ~msg:"shape" (array int) [| 2; 3; 2; 4 |] (Nx.shape y);
  check_arr ~msg:"lane (i, j) gathers over i" (to_arr expected) y;
  check_arr ~msg:"a map of another name" (to_arr expected)
    (Rune.vmap' ~axis:a (Rune.vmap' ~axis:b (fun r -> Rune.lanes a r)) x);
  let y =
    Rune.vmap' ~axis:a
      (fun r -> Rune.vmap' (fun _ -> Rune.lanes a r) (Nx.zeros f64 [| 3; 1 |]))
      x
  in
  check_arr ~msg:"an operand the inner map does not batch"
    (to_arr (stack 2 (fun _ -> stack 3 (fun _ -> x))))
    y

(* With no map of that name around it, a gather is one lane. *)
let test_lanes_without_the_map () =
  let a = Rune.axis () and b = Rune.axis () and x = xs () in
  check_arr ~msg:"no map" (to_arr x) (Rune.lanes a x);
  equal ~msg:"no map, shape" (array int) [| 1; 4; 3 |]
    (Nx.shape (Rune.lanes a x));
  let y = Rune.vmap' (fun r -> Rune.lanes a r) x in
  equal ~msg:"another map, shape" (array int) [| 4; 1; 3 |] (Nx.shape y);
  check_arr ~msg:"another map" (to_arr x) y;
  let y = Rune.vmap' ~axis:b (fun r -> Rune.lanes a r) x in
  equal ~msg:"a map of another name, shape" (array int) [| 4; 1; 3 |]
    (Nx.shape y);
  check_arr ~msg:"a map of another name" (to_arr x) y;
  check_arr ~msg:"compiled" (to_arr x) (Rune.jit' (Rune.lanes a) x)

(* A named map passes the lane index on: a key folded with it inside the map
   draws the same values in every lane, and those of the anonymous map around
   it. *)
let test_named_map_passes_the_lane_index () =
  let key = Nx.Rng.key 7 and a = Rune.axis () in
  let draw () =
    Nx.Rng.uniform (Nx.Rng.fold_in_tensor key (Rune.lane_index ())) f64 [| 3 |]
  in
  let y =
    Rune.vmap' ~axis:a (fun r -> Nx.add (Nx.mul_s r 0.0) (draw ())) (xs ())
  in
  check_arr ~msg:"one lane's draw" (to_arr (stack 4 (fun _ -> draw ()))) y;
  let y =
    Rune.vmap'
      (fun t ->
        Rune.vmap' ~axis:a
          (fun r -> Nx.add (Nx.mul_s (Nx.add r t) 0.0) (draw ()))
          (xs ()))
      (Nx.zeros f64 [| 2; 3 |])
  in
  let trial i = Nx.Rng.uniform (Nx.Rng.fold_in key i) f64 [| 3 |] in
  check_arr ~msg:"the anonymous map's draws"
    (to_arr (stack 2 (fun i -> stack 4 (fun _ -> trial i))))
    y

(* A gather is linear: its tangent is the gather of the tangent. *)
let test_lanes_under_jvp () =
  let a = Rune.axis () and x = xs () and dx = Nx.mul_s (xs ()) 0.5 in
  let gathered x =
    Rune.vmap' ~axis:a (fun r -> Nx.sum ~axes:[ 0 ] (Rune.lanes a r)) x
  in
  let _, dy = Rune.jvp' gathered x dx in
  check_arr ~msg:"jvp of the map" (to_arr (gathered dx)) dy;
  let y, dy = Rune.jvp' (Rune.lanes a) x dx in
  equal ~msg:"no map, shape" (array int) [| 1; 4; 3 |] (Nx.shape dy);
  check_arr ~msg:"no map, primal" (to_arr x) y;
  check_arr ~msg:"no map, tangent" (to_arr dx) dy;
  let c = vec64 [| 0.3; -0.7; 1.1 |] in
  let dy =
    Rune.vmap' ~axis:a
      (fun d ->
        snd
          (Rune.jvp'
             (fun c -> Nx.sum ~axes:[ 0 ] (Rune.lanes a (Nx.mul c c)))
             c d))
      dx
  in
  let total = Nx.mul (Nx.mul_s c 2.0) (Nx.sum ~axes:[ 0 ] dx) in
  check_arr ~msg:"jvp inside the map" (to_arr (stack 4 (fun _ -> total))) dy

(* Outside the map named [a], reverse mode differentiates through the answer:
   [Σ_i (Σ_j x_j) · x_i = |Σ_j x_j|²] has gradient [2 Σ_j x_j] in every row. *)
let test_grad_outside_the_named_map () =
  let a = Rune.axis () and x = xs () in
  let f x =
    Nx.sum
      (Rune.vmap' ~axis:a
         (fun r -> Nx.sum (Nx.mul (Nx.sum ~axes:[ 0 ] (Rune.lanes a r)) r))
         x)
  in
  check_arr ~msg:"gradient"
    (to_arr (Nx.broadcast_to [| 4; 3 |] (Nx.mul_s (Nx.sum ~axes:[ 0 ] x) 2.0)))
    (Rune.grad' f x)

let test_lanes_under_grad_raises () =
  let a = Rune.axis () in
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.vmap' ~axis:a
           (fun r -> Rune.grad' (fun r -> Nx.sum (Rune.lanes a r)) r)
           (xs ())))

let tests =
  [
    group "loop oracle" oracle_tests;
    group "axes and structure"
      [
        test "maps a moved axis" test_moved_axis;
        test "maps all leaves of a structure" test_vmap_structure;
        test "maps leaves of different leading ranks"
          test_vmap_structure_leading_dims;
        test "a captured value is a constant" test_captured_value_is_constant;
        test "a capture of the argument is a constant"
          test_capture_of_the_argument_is_constant;
        test "rejects arguments with no leaf" test_rejects_no_leaf;
        test "rejects a consumed argument" test_rejects_consumes;
        test "rejects mismatched batch sizes" test_batch_size_mismatch;
        test "rejects scalar leaves" test_scalar_leaf_rejected;
        test "raises without a batching rule" test_no_rule_raises;
        test "reading a batched value raises" test_reading_batched_value_raises;
        test "reading a constant value is fine"
          test_reading_constant_value_is_fine;
      ];
    group "set"
      [
        test "window write batches" test_vmap_set_window;
        test "a batched window start batches" test_vmap_set_window_batched_start;
      ];
    group "nesting" [ test "vmap of vmap" test_nested_vmap ];
    group "lanes"
      [
        test "the named map answers" test_lanes_of_the_named_map;
        test "a shared value is broadcast" test_lanes_of_a_shared_value;
        test "another map keeps its lanes" test_lanes_through_another_map;
        test "one lane without the map" test_lanes_without_the_map;
        test "a named map passes the lane index on"
          test_named_map_passes_the_lane_index;
        test "linear under jvp" test_lanes_under_jvp;
        test "grad outside the named map" test_grad_outside_the_named_map;
        test "raises under grad inside the named map"
          test_lanes_under_grad_raises;
      ];
    group "randomness"
      [
        test "implicit RNG draws are identical per lane"
          test_rng_is_identical_per_lane;
      ];
    group "structured outputs"
      [ test "batches every output leaf" test_vmap_structured_output ];
    group "composition"
      [
        test "vmap of grad: per-sample gradients" test_per_sample_gradients;
        test "per-sample gradients of a gather"
          test_per_sample_gradients_of_gather;
        test "per-sample gradients of a sliding window"
          test_per_sample_gradients_of_sliding_window;
        test "grad of vmap" test_grad_through_vmap;
        test "jvp of vmap" test_jvp_through_vmap;
      ];
  ]

let () = exit (run "rune vmap" tests)
