(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Just-in-time compilation: trace/compile/replay correctness, signature
   retracing, composition with the other transformations, in-place state, and
   trace-time failure modes. *)

open Windtrap
open Rune_test_support.Support

let raises_jit_error f =
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

(* loss p = sum (w * w) + 3 * sum b. d/dw = 2w, d/db = 3, d/dscale = 0. *)
let quadratic p = Nx.add (Nx.sum (Nx.mul p.w p.w)) (Nx.mul_s (Nx.sum p.b) 3.0)

(* Basics *)

let test_elementwise_matches_eager () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_replay_reads_fresh_inputs () =
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  ignore (g (vec32 [| 1.0; 2.0; 3.0 |]));
  check_arr ~msg:"fresh data" [| 4.0; 9.0; 16.0 |]
    (g (vec32 [| 2.0; 3.0; 4.0 |]))

let test_retrace_on_new_shape () =
  let g = Rune.jit' (fun x -> Nx.sum x) in
  check_arr ~msg:"vector" [| 6.0 |] (g (vec32 [| 1.0; 2.0; 3.0 |]));
  check_arr ~msg:"matrix" [| 10.0 |]
    (g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]))

let test_closure_matmul () =
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.matmul x w in
  let g = Rune.jit' f in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"matmul" (to_arr (f x)) (g x)

let test_jit2_structured_output () =
  let f p =
    { w = Nx.mul p.w p.w; b = Nx.add p.b p.b; scale = Nx.mul_s p.scale 2.0 }
  in
  let g = Rune.jit2 (module Params) (module Params) f in
  let p = params () in
  let r = g p in
  let e = f p in
  check_arr ~msg:"w" (to_arr e.w) r.w;
  check_arr ~msg:"b" (to_arr e.b) r.b;
  check_arr ~msg:"scale (float64)" (to_arr e.scale) r.scale

(* Composition *)

let test_grad_inside_jit () =
  let step =
    Rune.jit2
      (module Params)
      (module Params)
      (fun p -> Rune.grad (module Params) quadratic p)
  in
  let g = step (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"db" [| 3.0 |] g.b;
  check_arr ~msg:"dscale" [| 0.0 |] g.scale;
  (* Replay computes gradients at the new point. *)
  let p2 = { (params ()) with w = vec32 [| 4.0; 5.0; 6.0 |] } in
  let g2 = step p2 in
  check_arr ~msg:"dw at new point" [| 8.0; 10.0; 12.0 |] g2.w

let test_jit_under_grad_is_transparent () =
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  let dx = Rune.grad' (fun x -> Nx.sum (g x)) (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"d(sum x^2)" [| 2.0; 4.0; 6.0 |] dx

let test_jit_under_vmap_is_transparent () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let y = Rune.vmap' g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]) in
  check_arr ~msg:"vmap over jit" [| 2.0; 4.0; 6.0; 8.0 |] y

let test_scan_unrolls_inside_jit () =
  let module C = struct
    type t = Nx.float32_t

    let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

    let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
  end in
  let cumsum xs =
    snd
      (Rune.scan
         (module C)
         ~f:(fun c x ->
           let c = Nx.add c x in
           (c, c))
         ~init:(Nx.scalar f32 0.0) xs)
  in
  let g = Rune.jit' cumsum in
  check_arr ~msg:"cumulative sum" [| 1.0; 3.0; 6.0 |]
    (g (vec32 [| 1.0; 2.0; 3.0 |]))

(* In-place state *)

let test_assign_writes_back_to_leaf () =
  let step =
    Rune.jit
      (module Params)
      (fun p ->
        Nx.blit (Nx.mul_s p.w 2.0) p.w;
        Nx.sum p.w)
  in
  let p = params () in
  let s = step p in
  check_arr ~msg:"sum of updated w" [| 4.0 |] s;
  check_arr ~msg:"w updated in place" [| 2.0; -4.0; 6.0 |] p.w;
  ignore (step p);
  check_arr ~msg:"second step compounds" [| 4.0; -8.0; 12.0 |] p.w

(* Buffer sharing: strided leaves must fall back to copies, views with an offset
   must read the right span, and each call must return tensors with their own
   storage. *)

let test_non_contiguous_input_matches_eager () =
  let f x = Nx.add (Nx.mul x x) x in
  let g = Rune.jit' f in
  let x =
    Nx.transpose (Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
  in
  check_arr ~msg:"transposed input" (to_arr (f x)) (g x)

let test_offset_view_input_matches_eager () =
  let f x = Nx.mul_s x 3.0 in
  let g = Rune.jit' f in
  let x =
    Nx.get [ 1 ] (Nx.create f32 [| 3; 4 |] (Array.init 12 float_of_int))
  in
  check_arr ~msg:"offset row view" (to_arr (f x)) (g x)

let test_outputs_have_their_own_storage () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let y1 = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  let _y2 = g (vec32 [| 10.0; 20.0; 30.0 |]) in
  check_arr ~msg:"first result unchanged by the second call" [| 2.0; 4.0; 6.0 |]
    y1

let test_closure_mutation_is_observed () =
  let c = vec32 [| 10.0; 20.0; 30.0 |] in
  let g = Rune.jit' (fun x -> Nx.add x c) in
  check_arr ~msg:"initial capture" [| 11.0; 21.0; 31.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]));
  Nx.blit (vec32 [| 0.0; 0.0; 0.0 |]) c;
  check_arr ~msg:"mutated capture is re-read" [| 1.0; 1.0; 1.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]))

(* Sliding windows *)

(* An asymmetric configuration so any axis-ordering mistake shows up: distinct
   kernel, stride, dilation, and padding per spatial dimension. *)
let window_config =
  ( [| 2; 3 |] (* kernel *),
    [| 2; 1 |] (* stride *),
    [| 1; 2 |] (* dilation *),
    [| (1, 0); (2, 1) |] (* padding *) )

let window_input () =
  Nx.create f32 [| 2; 3; 5; 6 |]
    (Array.init (2 * 3 * 5 * 6) (fun i -> float_of_int (i mod 17) -. 8.0))

let test_unfold_matches_eager () =
  let kernel_size, stride, dilation, padding = window_config in
  let f x = Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x in
  let g = Rune.jit' f in
  let x = window_input () in
  equal ~msg:"shape" (array int) (Nx.shape (f x)) (Nx.shape (g x));
  check_arr ~msg:"unfold" (to_arr (f x)) (g x)

let test_fold_matches_eager () =
  let kernel_size, stride, dilation, padding = window_config in
  let output_size = [| 5; 6 |] in
  let f x =
    Nx.combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding
      (Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x)
  in
  let g = Rune.jit' f in
  let x = window_input () in
  check_arr ~msg:"fold of unfold" (to_arr (f x)) (g x)

let test_correlate_matches_eager () =
  let kernel =
    Nx.create f32 [| 3; 3 |]
      [| 1.0; 0.0; -1.0; 2.0; 0.5; -2.0; 1.0; 0.0; -1.0 |]
  in
  let f x = Nx.correlate ~padding:`Same x kernel in
  let g = Rune.jit' f in
  let x = Nx.create f32 [| 6; 7 |] (Array.init 42 (fun i -> float_of_int i)) in
  check_arr ~msg:"correlate same" (to_arr (f x)) (g x)

(* Indexed access *)

(* Row 1 repeats an index so duplicate handling is pinned under jit: [`Set]
   keeps the last update, [`Add] accumulates both on top of [x]'s value. *)
let test_scatter_matches_eager () =
  let idx = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 1l |] in
  let f mode x =
    Nx.scatter ~mode ~axis:1 ~indices:idx
      ~values:(Nx.slice [ Nx.A; Nx.R (0, 2) ] x)
      x
  in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let g_set = Rune.jit' (f `Set) and g_add = Rune.jit' (f `Add) in
  check_arr ~msg:"set" (to_arr (f `Set x)) (g_set x);
  check_arr ~msg:"set replay" (to_arr (f `Set x)) (g_set x);
  check_arr ~msg:"add" (to_arr (f `Add x)) (g_add x)

(* Training-step integration: a two-layer MLP trained by a jitted step must
   follow the eager trajectory exactly. *)

type mlp = { w1 : Nx.float32_t; b1 : Nx.float32_t; w2 : Nx.float32_t }

module Mlp = struct
  type t = mlp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w1; b1; w2 } =
    { w1 = f w1; b1 = f b1; w2 = f w2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w1 = f p.w1 q.w1; b1 = f p.b1 q.b1; w2 = f p.w2 q.w2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w1; b1; w2 } =
    f w1;
    f b1;
    f w2
end

let test_jitted_training_matches_eager () =
  let xs =
    Nx.create f32 [| 8; 4 |]
      (Array.init 32 (fun i -> float_of_int (i mod 7) /. 7.0))
  in
  let ys =
    Nx.create f32 [| 8; 1 |]
      (Array.init 8 (fun i -> float_of_int (i mod 3) -. 1.0))
  in
  let init () =
    {
      w1 =
        Nx.create f32 [| 4; 5 |]
          (Array.init 20 (fun i -> (0.1 *. float_of_int (i mod 5)) -. 0.2));
      b1 = Nx.zeros f32 [| 5 |];
      w2 =
        Nx.create f32 [| 5; 1 |]
          (Array.init 5 (fun i -> 0.3 -. (0.1 *. float_of_int i)));
    }
  in
  let loss p =
    let h = Nx.tanh (Nx.add (Nx.matmul xs p.w1) p.b1) in
    let d = Nx.sub (Nx.matmul h p.w2) ys in
    Nx.mean (Nx.mul d d)
  in
  let update p =
    let g = Rune.grad (module Mlp) loss p in
    Mlp.map2 (fun w dw -> Nx.sub w (Nx.mul (scalar_like dw 0.1) dw)) p g
  in
  let step = Rune.jit2 (module Mlp) (module Mlp) update in
  let rec train f p n = if n = 0 then p else train f (f p) (n - 1) in
  let jitted = train step (init ()) 5 in
  let eager = train update (init ()) 5 in
  check_arr ~msg:"w1" (to_arr eager.w1) jitted.w1;
  check_arr ~msg:"b1" (to_arr eager.b1) jitted.b1;
  check_arr ~msg:"w2" (to_arr eager.w2) jitted.w2;
  let l0 = scalar (loss (init ())) and l5 = scalar (loss jitted) in
  is_true ~msg:"loss decreased" (l5 < l0)

(* Failure modes *)

let test_data_dependent_read_raises () =
  let g = Rune.jit' (fun x -> if Nx.item [ 0 ] x > 0.0 then x else Nx.neg x) in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0 |]))

let test_unsupported_op_raises () =
  let g = Rune.jit' (fun x -> Nx.cholesky x) in
  raises_jit_error (fun () ->
      g (Nx.create f32 [| 2; 2 |] [| 4.0; 2.0; 2.0; 3.0 |]))

let tests =
  [
    group "jit basics"
      [
        test "element-wise chain matches eager" test_elementwise_matches_eager;
        test "replay reads fresh input data" test_replay_reads_fresh_inputs;
        test "a new shape retraces" test_retrace_on_new_shape;
        test "closure-captured weights (matmul)" test_closure_matmul;
        test "jit2 returns structured outputs" test_jit2_structured_output;
      ];
    group "composition"
      [
        test "grad inside jit matches eager grad" test_grad_inside_jit;
        test "jit under grad runs eagerly" test_jit_under_grad_is_transparent;
        test "jit under vmap runs eagerly" test_jit_under_vmap_is_transparent;
        test "scan unrolls into the trace" test_scan_unrolls_inside_jit;
      ];
    group "sliding windows"
      [
        test "unfold matches eager" test_unfold_matches_eager;
        test "fold of unfold matches eager" test_fold_matches_eager;
        test "correlate matches eager" test_correlate_matches_eager;
      ];
    group "indexed access"
      [ test "scatter matches eager" test_scatter_matches_eager ];
    group "training"
      [
        test "jitted training follows the eager trajectory"
          test_jitted_training_matches_eager;
      ];
    group "state"
      [
        test "assign writes back to a leaf" test_assign_writes_back_to_leaf;
        test "closure mutation between calls is observed"
          test_closure_mutation_is_observed;
        test "non-contiguous inputs fall back to copies"
          test_non_contiguous_input_matches_eager;
        test "offset views read the right span"
          test_offset_view_input_matches_eager;
        test "outputs have their own storage"
          test_outputs_have_their_own_storage;
      ];
    group "errors"
      [
        test "reading a traced value raises" test_data_dependent_read_raises;
        test "unsupported operations raise" test_unsupported_op_raises;
      ];
  ]

let () = run "rune jit" tests
