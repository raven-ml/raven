(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Multi-device parallel jit: pmap numerics against jit, replicate-vs-shard
   placements, gradients (cross-device allreduce), device residency and
   gather-on-read, placement errors, and the under-transformation fallback.
   Runs on CPU device instances, so no GPU is needed. *)

open Windtrap
open Rune_test_support.Support

let devs2 = [ "CPU:1"; "CPU:2" ]
let devs4 = [ "CPU:1"; "CPU:2"; "CPU:3"; "CPU:4" ]

let raises_jit_error f =
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

(* A single-tensor Ptree.S instance. *)
module Single_f32 = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let arange n = Array.init n (fun i -> float_of_int (i + 1) /. 7.0)
let m46 () = Nx.create f32 [| 4; 6 |] (arange 24)
let m86 () = Nx.create f32 [| 8; 6 |] (arange 48)

(* An elementwise + matmul + reduce chain over one tensor. The matmul
   combines the batch-sharded value with its own transpose (mismatched shard
   axes), exercising the cross-device realignment path on top of the plain
   allreduce. *)
let chain x =
  let y = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let z = Nx.matmul y (Nx.transpose y) in
  Nx.sum z ~axes:[ 1 ]

(* Numerics vs jit *)

let test_matches_jit_2dev () =
  let x = m46 () in
  let expect = Rune.jit' chain x in
  let g = Rune.pmap ~devices:devs2 (module Single_f32) chain in
  check_arr ~msg:"first call" (to_arr expect) (g x);
  check_arr ~msg:"replay" (to_arr expect) (g x)

let test_matches_jit_4dev () =
  let x = m86 () in
  let expect = Rune.jit' chain x in
  let g = Rune.pmap ~devices:devs4 (module Single_f32) chain in
  check_arr ~msg:"4 devices" (to_arr expect) (g x)

(* No cross-device reduce: each device computes its shard independently, so
   the result is byte-equal to the single-device one. *)
let test_elementwise_byte_equal () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let x = m46 () in
  let expect = Rune.jit' f x in
  let g = Rune.pmap ~devices:devs2 (module Single_f32) f in
  check_arr ~eps:0.0 ~msg:"byte-equal" (to_arr expect) (g x)

let test_shard_axis_1 () =
  let f x = Nx.add (Nx.mul x x) x in
  let x = m46 () in
  let expect = Rune.jit' f x in
  let g = Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ] (module Single_f32) f in
  check_arr ~eps:0.0 ~msg:"axis 1 shards" (to_arr expect) (g x)

let test_retrace_on_new_shape () =
  let g = Rune.pmap ~devices:devs2 (module Single_f32) (fun x -> Nx.sum x) in
  check_arr ~msg:"first shape" [| 36.0 |]
    (g (vec32 (Array.init 8 (fun i -> float_of_int (i + 1)))));
  check_arr ~msg:"retraced shape" [| 10.0 |]
    (g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]))

(* Replicate-vs-shard: the data-parallel shape. Params are replicated, the
   batch is sharded, and the mean loss reduces over the sharded axis — a
   cross-device allreduce. *)

type dp = { w : Nx.float32_t; x : Nx.float32_t; t : Nx.float32_t }

module Dp = struct
  type t = dp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; x; t } =
    { w = f w; x = f x; t = f t }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; x = f p.x q.x; t = f p.t q.t }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; x; t } =
    f w;
    f x;
    f t
end

let dp_loss p =
  let d = Nx.sub (Nx.matmul p.x p.w) p.t in
  Nx.mean (Nx.mul d d)

let dp_axes = [ None; Some 0; Some 0 ]

let dp_input () =
  {
    w = Nx.create f32 [| 3; 2 |] (arange 6);
    x = Nx.create f32 [| 4; 3 |] (arange 12);
    t = Nx.create f32 [| 4; 2 |] (arange 8);
  }

let test_dp_loss_matches_jit () =
  let p = dp_input () in
  let expect = Rune.jit (module Dp) dp_loss p in
  let g = Rune.pmap ~devices:devs2 ~in_axes:dp_axes (module Dp) dp_loss in
  check_arr ~msg:"mean loss over sharded batch" (to_arr expect) (g p)

(* Gradients: value_and_grad inside the pmapped function. Differentiating a
   mean over the sharded batch makes every parameter gradient a cross-device
   allreduce, which multi_pm inserts automatically — the DDP path. *)

let test_grad_inside_pmap () =
  let grads p = snd (Rune.value_and_grad (module Dp) dp_loss p) in
  let p = dp_input () in
  let expect = Rune.jit2 (module Dp) (module Dp) grads p in
  let g = Rune.pmap2 ~devices:devs2 ~in_axes:dp_axes (module Dp) (module Dp) grads in
  let got = g p in
  check_arr ~msg:"dw (allreduced)" (to_arr expect.w) got.w;
  check_arr ~msg:"dx (sharded)" (to_arr expect.x) got.x;
  check_arr ~msg:"dt (sharded)" (to_arr expect.t) got.t

(* Gradients through keepdims reductions: reduce ~keepdims:true realizes a
   per-shard buffer whose broadcast back against the sharded operand is the
   softmax / layer-norm shape. Differentiated inside pmap, each gradient must
   match the single-device jit gradient. *)

let keepdims_input () =
  {
    w = Nx.create f32 [| 8; 8 |] (arange 64);
    x = Nx.create f32 [| 8; 8 |] (Array.init 64 (fun i -> sin (float_of_int i)));
    t = Nx.create f32 [| 8; 8 |] (arange 64);
  }

let check_keepdims_grad loss =
  let grads p = snd (Rune.value_and_grad (module Dp) loss p) in
  let p = keepdims_input () in
  let expect = Rune.jit2 (module Dp) (module Dp) grads p in
  let axes = [ None; Some 0; Some 0 ] in
  let g = Rune.pmap2 ~devices:devs2 ~in_axes:axes (module Dp) (module Dp) grads in
  let got = g p in
  check_arr ~msg:"dw" (to_arr expect.w) got.w;
  check_arr ~msg:"dx" (to_arr expect.x) got.x

let test_grad_max_keepdims () =
  check_keepdims_grad (fun s ->
      let h = Nx.add s.x s.w in
      Nx.mean (Nx.exp (Nx.sub h (Nx.max h ~axes:[ -1 ] ~keepdims:true))))

let test_grad_sum_keepdims () =
  check_keepdims_grad (fun s ->
      let h = Nx.add s.x s.w in
      Nx.mean (Nx.mul h (Nx.sum h ~axes:[ -1 ] ~keepdims:true)))

let test_grad_mean_keepdims () =
  check_keepdims_grad (fun s ->
      let h = Nx.add s.x s.w in
      let d = Nx.sub h (Nx.mean h ~axes:[ -1 ] ~keepdims:true) in
      Nx.mean (Nx.mul d d))

(* Residency: outputs stay per-device; feeding an unread output back into a
   matching placement moves no bytes; reading gathers shards correctly. *)

let test_feedback_moves_no_bytes () =
  let g = Rune.pmap ~devices:devs2 (module Single_f32) (fun x -> Nx.add x x) in
  let x = vec32 (Array.init 8 (fun i -> float_of_int i)) in
  let y1 = g x in
  Rune.reset_jit_stats ();
  let y2 = g y1 in
  let s = Rune.jit_stats () in
  equal ~msg:"feedback call moves no bytes to device" int 0 s.bytes_to_device;
  (* gather on read: shards reassemble to 4x *)
  check_arr ~eps:0.0 ~msg:"gathered result"
    (Array.init 8 (fun i -> 4.0 *. float_of_int i))
    y2

let test_replicated_feedback () =
  (* w -> w * 2 with w replicated: the replicated output seeds the replicated
     input directly on the next call. *)
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ None ]
      (module Single_f32)
      (fun w -> Nx.mul_s w 2.0)
  in
  let w = vec32 [| 1.0; 2.0; 3.0 |] in
  let w1 = g w in
  Rune.reset_jit_stats ();
  let w2 = g w1 in
  let s = Rune.jit_stats () in
  equal ~msg:"replicated feedback moves no bytes" int 0 s.bytes_to_device;
  check_arr ~eps:0.0 ~msg:"replicated read" [| 4.0; 8.0; 12.0 |] w2

let test_mismatched_placement_forces () =
  (* An output sharded on axis 0 fed into an axis-1 placement is forced to
     the host and re-split, not seeded. *)
  let f x = Nx.add x x in
  let g0 = Rune.pmap ~devices:devs2 ~in_axes:[ Some 0 ] (module Single_f32) f in
  let g1 = Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ] (module Single_f32) f in
  let x = m46 () in
  let y = g0 x in
  check_arr ~eps:0.0 ~msg:"re-split result matches"
    (to_arr (Nx.add (Nx.add x x) (Nx.add x x)))
    (g1 y)

let test_pass_through_output () =
  let g =
    Rune.pmap2 ~devices:devs2
      (module Single_f32)
      (module Single_f32)
      (fun x ->
        ignore (Nx.sum x);
        x)
  in
  let x = m46 () in
  check_arr ~eps:0.0 ~msg:"pass-through gathers the input" (to_arr x) (g x)

(* Donation: [donate:true] consumes a resident multi-device handle — every
   per-device shard buffer is released once the call completes — and the
   donated handle raises on read. A handle whose placement mismatches is
   forced to the host first, so donation does not apply to it. *)

let raises_donated f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg ->
          msg
          = "Rune.jit: this tensor was donated to a jitted call; read or copy \
             it before the call"
      | _ -> false)
    (fun () -> ignore (f ()))

let test_donate_sharded_state () =
  let n = 1024 in
  let g =
    Rune.pmap ~devices:devs2 ~donate:true
      (module Single_f32)
      (fun x -> Nx.add_s x 1.0)
  in
  let x = vec32 (Array.make n 0.0) in
  let base = (Rune.jit_stats ()).resident_bytes in
  let h1 = g x in
  let h2 = g h1 in
  let h3 = g h2 in
  (* Donation bounds the loop at two generations even though h1 and h2 stay
     reachable; only h3's shards remain resident. *)
  is_true ~msg:"sharded state loop holds at most two generations"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * n * 4);
  raises_donated (fun () -> to_arr h1);
  raises_donated (fun () -> to_arr h2);
  check_arr ~eps:0.0 ~msg:"the live generation reads correctly"
    (Array.make n 3.0) h3

let test_donate_replicated_releases_all_shards () =
  let n = 512 in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ None ] ~donate:true
      (module Single_f32)
      (fun w -> Nx.mul_s w 2.0)
  in
  let base = (Rune.jit_stats ()).resident_bytes in
  let w1 = g (vec32 (Array.make n 1.0)) in
  (* A replicated handle owns one full-size buffer per device. *)
  is_true ~msg:"one replicated generation is resident"
    ((Rune.jit_stats ()).resident_bytes - base >= 2 * n * 4);
  let w2 = g w1 in
  is_true ~msg:"donating releases every replica"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * n * 4);
  raises_donated (fun () -> to_arr w1);
  check_arr ~eps:0.0 ~msg:"value" (Array.make n 4.0) w2

let test_donate_mismatched_placement_not_consumed () =
  let f x = Nx.add x x in
  let g0 = Rune.pmap ~devices:devs2 ~in_axes:[ Some 0 ] (module Single_f32) f in
  let g1 =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ] ~donate:true
      (module Single_f32)
      f
  in
  let x = m46 () in
  let y = g0 x in
  (* The axis-1 call forces y to the host to re-split it: y is a plain host
     tensor afterwards, not donated. *)
  check_arr ~eps:0.0 ~msg:"re-split result matches"
    (to_arr (Nx.add (Nx.add x x) (Nx.add x x)))
    (g1 y);
  check_arr ~eps:0.0 ~msg:"the mismatched handle survives as a host tensor"
    (to_arr (Nx.add x x))
    y

(* Writebacks: replicated destinations are honored, sharded values are
   rejected at trace time. *)

let test_replicated_writeback () =
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ None ]
      (module Single_f32)
      (fun w ->
        Nx.blit (Nx.mul_s w 2.0) w;
        Nx.sum w)
  in
  let w = vec32 [| 1.0; 2.0; 3.0 |] in
  check_arr ~msg:"sum of updated value" [| 12.0 |] (g w);
  check_arr ~eps:0.0 ~msg:"writeback reached the host leaf"
    [| 2.0; 4.0; 6.0 |] w

let test_sharded_writeback_raises () =
  let g =
    Rune.pmap ~devices:devs2
      (module Single_f32)
      (fun x ->
        Nx.blit (Nx.mul x x) x;
        Nx.sum x)
  in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0; 3.0; 4.0 |]))

(* Errors *)

let test_empty_devices () =
  raises_invalid_arg (fun () ->
      let g = Rune.pmap ~devices:[] (module Single_f32) Fun.id in
      ignore (g (vec32 [| 1.0; 2.0 |])))

let test_mixed_backends () =
  raises_invalid_arg (fun () ->
      let g =
        Rune.pmap ~devices:[ "CPU:1"; "CUDA:0" ] (module Single_f32) Fun.id
      in
      ignore (g (vec32 [| 1.0; 2.0 |])))

let test_non_divisible_axis () =
  let g = Rune.pmap ~devices:devs2 (module Single_f32) Fun.id in
  raises_invalid_arg (fun () -> ignore (g (vec32 [| 1.0; 2.0; 3.0 |])))

let test_in_axes_arity () =
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 0; None ] (module Single_f32)
      Fun.id
  in
  raises_invalid_arg (fun () -> ignore (g (vec32 [| 1.0; 2.0 |])))

let test_axis_out_of_range () =
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 3 ] (module Single_f32) Fun.id
  in
  raises_invalid_arg (fun () -> ignore (g (vec32 [| 1.0; 2.0 |])))

(* Under an enclosing transformation the pmapped function runs eagerly, so
   grad over pmap differentiates the plain function. *)

let test_grad_over_pmap_runs_eagerly () =
  let g =
    Rune.pmap ~devices:devs2 (module Single_f32) (fun x ->
        Nx.sum (Nx.mul x x))
  in
  let x = vec32 [| 1.0; 2.0; 3.0; 4.0 |] in
  let dx = Rune.grad (module Single_f32) (fun x -> g x) x in
  check_arr ~msg:"grad over pmap" [| 2.0; 4.0; 6.0; 8.0 |] dx

(* The DP microbench: a 2-layer MLP train step (value_and_grad + SGD inside
   pmap2), params replicated, batch sharded over 2 devices. The 10-step loss
   trajectory matches single-device jit at the same effective batch. *)

type mlp = {
  w1 : Nx.float32_t;
  b1 : Nx.float32_t;
  w2 : Nx.float32_t;
  b2 : Nx.float32_t;
  xb : Nx.float32_t;
  yb : Nx.float32_t;
}

module Mlp = struct
  type t = mlp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    {
      w1 = f s.w1;
      b1 = f s.b1;
      w2 = f s.w2;
      b2 = f s.b2;
      xb = f s.xb;
      yb = f s.yb;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s t =
    {
      w1 = f s.w1 t.w1;
      b1 = f s.b1 t.b1;
      w2 = f s.w2 t.w2;
      b2 = f s.b2 t.b2;
      xb = f s.xb t.xb;
      yb = f s.yb t.yb;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    f s.w1;
    f s.b1;
    f s.w2;
    f s.b2;
    f s.xb;
    f s.yb
end

module Mlp_out = struct
  type t = mlp * Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (s, l) =
    (Mlp.map f s, f l)

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (s, l)
      (t, m) =
    (Mlp.map2 f s t, f l m)

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) (s, l) =
    Mlp.iter f s;
    f l
end

let mlp_loss s =
  let h = Nx.relu (Nx.add (Nx.matmul s.xb s.w1) s.b1) in
  let p = Nx.add (Nx.matmul h s.w2) s.b2 in
  let d = Nx.sub p s.yb in
  Nx.mean (Nx.mul d d)

let sgd (type a b) (w : (a, b) Nx.t) (g : (a, b) Nx.t) : (a, b) Nx.t =
  Nx.sub w (Nx.mul g (scalar_like w 0.05))

let mlp_step s =
  let l, g = Rune.value_and_grad (module Mlp) mlp_loss s in
  ({ (Mlp.map2 sgd s g) with xb = s.xb; yb = s.yb }, l)

let mlp_init () =
  let rng i n =
    Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.5)
  in
  {
    w1 = Nx.create f32 [| 4; 8 |] (rng 1 32);
    b1 = Nx.create f32 [| 8 |] (rng 2 8);
    w2 = Nx.create f32 [| 8; 2 |] (rng 3 16);
    b2 = Nx.create f32 [| 2 |] (rng 4 2);
    xb = Nx.create f32 [| 16; 4 |] (rng 5 64);
    yb = Nx.create f32 [| 16; 2 |] (rng 6 32);
  }

let trajectory step0 =
  let s = ref (mlp_init ()) in
  Array.init 10 (fun _ ->
      let s', l = step0 !s in
      s := s';
      scalar l)

let test_dp_training_matches_jit () =
  let jit_losses =
    trajectory (Rune.jit2 (module Mlp) (module Mlp_out) mlp_step)
  in
  let in_axes = [ None; None; None; None; Some 0; Some 0 ] in
  let pmap_losses =
    trajectory
      (Rune.pmap2 ~devices:devs2 ~in_axes (module Mlp) (module Mlp_out)
         mlp_step)
  in
  Array.iteri
    (fun i l ->
      equal
        ~msg:(Printf.sprintf "loss at step %d" i)
        (float 1e-6) l pmap_losses.(i))
    jit_losses;
  (* Late steps run entirely on resident state: nothing moves to the
     devices. *)
  Rune.reset_jit_stats ();
  let pstep = Rune.pmap2 ~devices:devs2 ~in_axes (module Mlp) (module Mlp_out) mlp_step in
  let s0 = mlp_init () in
  let s1, _ = pstep s0 in
  Rune.reset_jit_stats ();
  let s2, _ = pstep s1 in
  let stats = Rune.jit_stats () in
  equal ~msg:"resident training step moves no bytes to device" int 0
    stats.bytes_to_device;
  ignore (Sys.opaque_identity s2)

let tests =
  [
    group "numerics"
      [
        test "elementwise+matmul+reduce matches jit on 2 devices"
          test_matches_jit_2dev;
        test "elementwise+matmul+reduce matches jit on 4 devices"
          test_matches_jit_4dev;
        test "elementwise chain is byte-equal to jit"
          test_elementwise_byte_equal;
        test "sharding along axis 1" test_shard_axis_1;
        test "new shape retraces" test_retrace_on_new_shape;
      ];
    group "placement"
      [
        test "replicated params + sharded batch mean loss matches jit"
          test_dp_loss_matches_jit;
        test "grad inside pmap allreduces like jit" test_grad_inside_pmap;
        test "grad through max keepdims matches jit" test_grad_max_keepdims;
        test "grad through sum keepdims matches jit" test_grad_sum_keepdims;
        test "grad through mean keepdims matches jit" test_grad_mean_keepdims;
      ];
    group "residency"
      [
        test "feedback call moves no bytes" test_feedback_moves_no_bytes;
        test "replicated outputs feed back without transfer"
          test_replicated_feedback;
        test "mismatched placement forces and re-splits"
          test_mismatched_placement_forces;
        test "pass-through outputs gather on read" test_pass_through_output;
      ];
    group "donation"
      [
        test "sharded state loop is bounded at two generations"
          test_donate_sharded_state;
        test "replicated donation releases every replica"
          test_donate_replicated_releases_all_shards;
        test "mismatched placement forces instead of donating"
          test_donate_mismatched_placement_not_consumed;
      ];
    group "state"
      [
        test "replicated writeback reaches the host leaf"
          test_replicated_writeback;
        test "sharded writeback raises" test_sharded_writeback_raises;
      ];
    group "errors"
      [
        test "empty devices raises" test_empty_devices;
        test "mixed backends raise" test_mixed_backends;
        test "non-divisible shard axis raises" test_non_divisible_axis;
        test "in_axes arity mismatch raises" test_in_axes_arity;
        test "shard axis out of range raises" test_axis_out_of_range;
      ];
    group "composition"
      [
        test "grad over pmap runs eagerly" test_grad_over_pmap_runs_eagerly;
      ];
    group "training"
      [
        test "data-parallel MLP training follows the jit trajectory"
          test_dp_training_matches_jit;
      ];
  ]

let () = run "rune pmap" tests
