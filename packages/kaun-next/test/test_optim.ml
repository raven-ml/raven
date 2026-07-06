(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun_next

(* A single float64 tensor, for analytic trajectory checks. *)
module Vec = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* Two float32 leaves of different shapes, for structural pairing checks. *)
module Pair = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { a; b } =
    { a = f a; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { a; b } =
    f a;
    f b
end

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs

let pair a b =
  {
    Pair.a = Nx.create Nx.float32 [| Array.length a |] a;
    b = Nx.create Nx.float32 [| Array.length b |] b;
  }

let check_vec ?(eps = 1e-9) ?msg expected actual =
  equal ?msg (array (float eps)) expected (Nx.to_array actual)

(* Quadratic bowl over [Pair]: f p = ||p - target||^2, with analytic gradients,
   so tests exercise the optimizer alone. *)
let bowl_target = lazy (pair [| 1.5; -0.5 |] [| 2.0 |])
let bowl_start = lazy (pair [| 5.0; -3.0 |] [| -4.0 |])

let bowl_grads (params : Pair.t) =
  let target = Lazy.force bowl_target in
  {
    Pair.a = Nx.mul_s (Nx.sub params.a target.a) 2.0;
    b = Nx.mul_s (Nx.sub params.b target.b) 2.0;
  }

let bowl_distance params =
  Optim.global_norm
    (module Pair)
    (Pair.map2 Nx.sub params (Lazy.force bowl_target))

let descend ~steps ~step params =
  let rec loop k acc = if k = 0 then acc else loop (k - 1) (step acc) in
  loop steps params

(* Schedules *)

let test_constant () =
  let sched = Optim.constant 0.1 in
  equal (float 0.) 0.1 (sched 0);
  equal (float 0.) 0.1 (sched 1000)

let test_exponential_decay () =
  let sched = Optim.exponential_decay ~init:0.5 ~rate:0.1 ~steps:100 in
  equal (float 1e-12) 0.5 (sched 0);
  equal (float 1e-12) 0.05 (sched 100);
  equal (float 1e-12) 0.005 (sched 200)

let test_cosine_decay () =
  let sched = Optim.cosine_decay ~final:0.01 ~init:0.1 ~steps:100 () in
  equal (float 1e-12) 0.1 (sched 0);
  equal (float 1e-12) 0.055 (sched 50);
  equal (float 1e-12) 0.01 (sched 100);
  equal ~msg:"stays at final past steps" (float 1e-12) 0.01 (sched 250)

let test_warmup_cosine () =
  let sched = Optim.warmup_cosine ~peak:1.0 ~warmup:10 ~steps:110 () in
  equal (float 1e-12) 0.0 (sched 0);
  equal (float 1e-12) 0.5 (sched 5);
  equal (float 1e-12) 1.0 (sched 10);
  equal ~msg:"cosine midpoint" (float 1e-12) 0.5 (sched 60);
  equal (float 1e-12) 0.0 (sched 110)

let test_schedule_validation () =
  raises_invalid_arg "Optim.exponential_decay: steps <= 0" (fun () ->
      Optim.exponential_decay ~init:1.0 ~rate:0.5 ~steps:0);
  raises_invalid_arg "Optim.cosine_decay: steps <= 0" (fun () ->
      Optim.cosine_decay ~init:1.0 ~steps:(-1) ());
  raises_invalid_arg "Optim.warmup_cosine: warmup < 0" (fun () ->
      Optim.warmup_cosine ~peak:1.0 ~warmup:(-1) ~steps:10 ());
  raises_invalid_arg "Optim.warmup_cosine: steps <= warmup" (fun () ->
      Optim.warmup_cosine ~peak:1.0 ~warmup:10 ~steps:10 ())

(* Gradient transformations *)

let test_global_norm () =
  (* sqrt (3^2 + 0^2 + 4^2 + 12^2) = 13 *)
  let grads = pair [| 3.0; 0.0 |] [| 4.0; 12.0 |] in
  equal (float 1e-6) 13.0 (Optim.global_norm (module Pair) grads)

let test_clip_by_global_norm_rescales () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Optim.clip_by_global_norm (module Pair) ~max_norm:1.0 grads in
  equal ~msg:"norm is the bound" (float 1e-6) 1.0
    (Optim.global_norm (module Pair) clipped);
  check_vec ~eps:1e-6 ~msg:"direction preserved" [| 0.6; 0.0 |] clipped.a;
  check_vec ~eps:1e-6 [| 0.8 |] clipped.b

let test_clip_by_global_norm_small () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Optim.clip_by_global_norm (module Pair) ~max_norm:10.0 grads in
  check_vec ~eps:0. [| 3.0; 0.0 |] clipped.a;
  check_vec ~eps:0. [| 4.0 |] clipped.b;
  let zeros = pair [| 0.0; 0.0 |] [| 0.0 |] in
  let clipped = Optim.clip_by_global_norm (module Pair) ~max_norm:1.0 zeros in
  check_vec ~eps:0. ~msg:"zero gradients pass through" [| 0.0 |] clipped.b

let test_clip_by_value () =
  let grads = pair [| -3.0; 0.2 |] [| 5.0 |] in
  let clipped = Optim.clip_by_value (module Pair) ~max:1.0 grads in
  check_vec ~eps:1e-7 [| -1.0; 0.2 |] clipped.a;
  check_vec ~eps:0. [| 1.0 |] clipped.b

let test_clip_validation () =
  let grads = pair [| 1.0 |] [| 1.0 |] in
  raises_invalid_arg "Optim.clip_by_global_norm: max_norm <= 0" (fun () ->
      Optim.clip_by_global_norm (module Pair) ~max_norm:0.0 grads);
  raises_invalid_arg "Optim.clip_by_value: max <= 0" (fun () ->
      Optim.clip_by_value (module Pair) ~max:(-1.0) grads)

(* SGD *)

let test_sgd_first_step () =
  let params = vec [| 1.0; -2.0 |] in
  let grads = vec [| 0.5; -1.0 |] in
  let st = Optim.sgd_init (module Vec) params in
  (* Zero velocity: the first step is plain descent even with momentum. *)
  let params', st' =
    Optim.sgd_step (module Vec) ~lr:0.1 ~momentum:0.9 st ~params ~grads
  in
  check_vec [| 0.95; -1.9 |] params';
  check_vec ~msg:"velocity is the gradient" [| 0.5; -1.0 |] st'.velocity

let test_sgd_velocity_threads () =
  let params = vec [| 0.0 |] in
  let st = Optim.sgd_init (module Vec) params in
  let params, st =
    Optim.sgd_step
      (module Vec)
      ~lr:0.1 ~momentum:0.5 st ~params ~grads:(vec [| 1.0 |])
  in
  let _, st =
    Optim.sgd_step
      (module Vec)
      ~lr:0.1 ~momentum:0.5 st ~params ~grads:(vec [| 2.0 |])
  in
  (* v2 = 0.5 *. v1 +. g2 = 0.5 *. 1. +. 2. *)
  check_vec [| 2.5 |] st.velocity

let test_sgd_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Optim.sgd_step (module Pair) ~lr:0.1 st ~params ~grads
  in
  let params, _ =
    descend ~steps:100 ~step (params, Optim.sgd_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_momentum_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Optim.sgd_step (module Pair) ~lr:0.05 ~momentum:0.9 st ~params ~grads
  in
  let params, _ =
    descend ~steps:200 ~step (params, Optim.sgd_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_pairs_leaves_structurally () =
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let grads = pair [| 0.0; 0.0 |] [| 1.0 |] in
  let st = Optim.sgd_init (module Pair) params in
  let params', _ = Optim.sgd_step (module Pair) ~lr:0.5 st ~params ~grads in
  check_vec ~eps:0. ~msg:"zero-gradient leaf untouched" [| 1.0; 2.0 |] params'.a;
  check_vec ~eps:0. [| 2.5 |] params'.b

(* Adam *)

let test_adam_first_step () =
  let b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and lr = 0.1 in
  let g = [| 4.0; -0.5; 0.0 |] in
  let params = vec [| 1.0; -2.0; 3.0 |] in
  let st = Optim.adam_init (module Vec) params in
  let params', st' =
    Optim.adam_step (module Vec) ~lr st ~params ~grads:(vec g)
  in
  (* First step analytically: mu = (1-b1) g, nu = (1-b2) g^2, and the
     bias-corrected direction is g / (|g| + eps). *)
  let expected =
    Array.map2
      (fun p g -> p -. (lr *. g /. (Float.abs g +. eps)))
      (Nx.to_array params) g
  in
  check_vec expected params';
  check_vec ~msg:"mu" (Array.map (fun g -> (1. -. b1) *. g) g) st'.mu;
  check_vec ~msg:"nu" (Array.map (fun g -> (1. -. b2) *. g *. g) g) st'.nu;
  equal ~msg:"step count" int 1 st'.step

let test_adam_reference_trajectory () =
  let b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and lr = 0.05 in
  let grad p = 2.0 *. (p -. 1.0) in
  (* Scalar reference implementation in plain floats. *)
  let expected =
    let p = ref 3.0 and mu = ref 0.0 and nu = ref 0.0 in
    List.init 10 (fun i ->
        let t = i + 1 in
        let g = grad !p in
        mu := (b1 *. !mu) +. ((1.0 -. b1) *. g);
        nu := (b2 *. !nu) +. ((1.0 -. b2) *. g *. g);
        let mu_hat = !mu /. (1.0 -. (b1 ** float_of_int t)) in
        let nu_hat = !nu /. (1.0 -. (b2 ** float_of_int t)) in
        p := !p -. (lr *. mu_hat /. (Stdlib.sqrt nu_hat +. eps));
        !p)
  in
  let params = ref (vec [| 3.0 |]) in
  let st = ref (Optim.adam_init (module Vec) !params) in
  List.iteri
    (fun i e ->
      let grads = Nx.mul_s (Nx.sub_s !params 1.0) 2.0 in
      let params', st' =
        Optim.adam_step (module Vec) ~lr !st ~params:!params ~grads
      in
      params := params';
      st := st';
      check_vec ~msg:(Printf.sprintf "step %d" (i + 1)) [| e |] !params)
    expected

let test_adam_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Optim.adam_step (module Pair) ~lr:0.02 st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Optim.adam_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

let test_adam_with_schedule_converges () =
  let sched = Optim.cosine_decay ~init:0.1 ~steps:300 () in
  let params = Lazy.force bowl_start in
  let st = Optim.adam_init (module Pair) params in
  let state = ref (params, st) in
  for k = 0 to 299 do
    let params, st = !state in
    let grads = bowl_grads params in
    state := Optim.adam_step (module Pair) ~lr:(sched k) st ~params ~grads
  done;
  is_true ~msg:"decayed steps settle at the bottom"
    (bowl_distance (fst !state) < 0.02)

let test_adam_zero_grads () =
  let params = vec [| 1.0; -2.0 |] in
  let st = Optim.adam_init (module Vec) params in
  let params', st' =
    Optim.adam_step (module Vec) ~lr:0.1 st ~params ~grads:(vec [| 0.0; 0.0 |])
  in
  check_vec ~eps:0. ~msg:"parameters unchanged" [| 1.0; -2.0 |] params';
  equal ~msg:"step still advances" int 1 st'.step

let test_adam_step_is_pure () =
  let params = vec [| 3.0; -1.0 |] in
  let grads = vec [| 0.7; 0.3 |] in
  let st = Optim.adam_init (module Vec) params in
  let once, _ = Optim.adam_step (module Vec) ~lr:0.1 st ~params ~grads in
  let again, _ = Optim.adam_step (module Vec) ~lr:0.1 st ~params ~grads in
  check_vec ~eps:0. ~msg:"same state, same step" (Nx.to_array once) again

(* AdamW *)

let test_adamw_zero_decay_is_adam () =
  let grads_of params = Nx.mul_s (Nx.sub_s params 1.0) 2.0 in
  let run step =
    let params = ref (vec [| 3.0; -2.0 |]) in
    let st = ref (Optim.adam_init (module Vec) !params) in
    for _ = 1 to 5 do
      let params', st' = step !st ~params:!params ~grads:(grads_of !params) in
      params := params';
      st := st'
    done;
    !params
  in
  let adam =
    run (fun st ~params ~grads ->
        Optim.adam_step (module Vec) ~lr:0.1 st ~params ~grads)
  in
  let adamw =
    run (fun st ~params ~grads ->
        Optim.adamw_step
          (module Vec)
          ~lr:0.1 ~weight_decay:0.0 st ~params ~grads)
  in
  check_vec ~eps:0. (Nx.to_array adam) adamw

let test_adamw_decays_weights () =
  (* Zero gradients isolate the decay: p_k = p_0 (1 - lr wd)^k. A coupled (L2)
     decay would instead be distorted by the adaptive scaling. *)
  let lr = 0.1 and wd = 0.5 in
  let p0 = [| 2.0; -4.0 |] in
  let params = ref (vec p0) in
  let st = ref (Optim.adamw_init (module Vec) !params) in
  for _ = 1 to 3 do
    let params', st' =
      Optim.adamw_step
        (module Vec)
        ~lr ~weight_decay:wd !st ~params:!params
        ~grads:(vec [| 0.0; 0.0 |])
    in
    params := params';
    st := st'
  done;
  let c = (1.0 -. (lr *. wd)) ** 3.0 in
  check_vec (Array.map (fun p -> p *. c) p0) !params

let test_adamw_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Optim.adamw_step (module Pair) ~lr:0.02 ~weight_decay:1e-3 st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Optim.adamw_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

let tests =
  [
    group "schedules"
      [
        test "constant is constant" test_constant;
        test "exponential decay is geometric in steps" test_exponential_decay;
        test "cosine decay spans init to final" test_cosine_decay;
        test "warmup cosine ramps then decays" test_warmup_cosine;
        test "constructors reject bad step counts" test_schedule_validation;
      ];
    group "gradient transformations"
      [
        test "global norm spans all leaves" test_global_norm;
        test "clip by global norm rescales to the bound"
          test_clip_by_global_norm_rescales;
        test "clip by global norm passes small gradients through"
          test_clip_by_global_norm_small;
        test "clip by value clamps elementwise" test_clip_by_value;
        test "clipping rejects non-positive bounds" test_clip_validation;
      ];
    group "sgd"
      [
        test "first step is plain gradient descent" test_sgd_first_step;
        test "velocity threads across steps" test_sgd_velocity_threads;
        test "converges on a quadratic bowl" test_sgd_converges;
        test "momentum converges on a quadratic bowl"
          test_sgd_momentum_converges;
        test "pairs leaves structurally, not positionally"
          test_sgd_pairs_leaves_structurally;
      ];
    group "adam"
      [
        test "first step matches the analytic update" test_adam_first_step;
        test "follows the scalar reference trajectory"
          test_adam_reference_trajectory;
        test "converges on a quadratic bowl" test_adam_converges;
        test "converges under a cosine schedule"
          test_adam_with_schedule_converges;
        test "zero gradients leave parameters unchanged" test_adam_zero_grads;
        test "stepping is pure in the threaded state" test_adam_step_is_pure;
      ];
    group "adamw"
      [
        test "zero weight decay reduces to adam" test_adamw_zero_decay_is_adam;
        test "zero gradients decay weights geometrically"
          test_adamw_decays_weights;
        test "converges on a quadratic bowl" test_adamw_converges;
      ];
  ]

let () = run "kaun-next optim" tests
