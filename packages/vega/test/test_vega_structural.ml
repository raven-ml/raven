(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega's structural tier: optimizers over Nx.Ptree.S. *)

open Windtrap
module S = Vega.Schedule

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
  let t = if eps = 0. then float_exact else float eps in
  equal ?msg (array t) expected (Nx.to_array actual)

(* The float64 analytic tests need the rate exact at float64; [Vega.lr] is
   float32 (cast up, it would perturb the last digits). The [~lr] argument takes
   any float dtype. *)
let lr64 v = Nx.scalar Nx.float64 v

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
  Vega.global_norm
    (module Pair)
    (Pair.map2 Nx.sub params (Lazy.force bowl_target))

let descend ~steps ~step params =
  let rec loop k acc = if k = 0 then acc else loop (k - 1) (step acc) in
  loop steps params

(* Schedules. [S.eval] reads a schedule at a host counter; the values are
   float32, so expectations carry float32 tolerances. *)

let test_constant () =
  let sched = S.constant 0.1 in
  equal (float 1e-6) 0.1 (S.eval sched 0);
  equal (float 1e-6) 0.1 (S.eval sched 1000)

let test_exponential_decay () =
  let sched =
    S.exponential_decay ~init_value:0.5 ~decay_rate:0.1 ~decay_steps:100
  in
  equal (float 1e-6) 0.5 (S.eval sched 0);
  equal (float 1e-6) 0.05 (S.eval sched 100);
  equal (float 1e-6) 0.005 (S.eval sched 200)

let test_cosine_decay () =
  (* alpha = 0.1 makes the final value alpha * init_value = 0.01. *)
  let sched = S.cosine_decay ~init_value:0.1 ~decay_steps:100 ~alpha:0.1 () in
  equal (float 1e-6) 0.1 (S.eval sched 0);
  equal (float 1e-6) 0.055 (S.eval sched 50);
  equal (float 1e-6) 0.01 (S.eval sched 100);
  equal ~msg:"stays at final past steps" (float 1e-6) 0.01 (S.eval sched 250)

let test_warmup_cosine () =
  let sched =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
      ~decay_steps:100 ()
  in
  equal (float 1e-6) 0.0 (S.eval sched 0);
  equal (float 1e-6) 0.5 (S.eval sched 5);
  equal (float 1e-6) 1.0 (S.eval sched 10);
  equal ~msg:"cosine midpoint" (float 1e-6) 0.5 (S.eval sched 60);
  equal (float 1e-6) 0.0 (S.eval sched 110)

let test_schedule_validation () =
  raises
    (Invalid_argument "Schedule.exponential_decay: decay_steps must be positive")
    (fun () ->
      ignore
        (S.exponential_decay ~init_value:1.0 ~decay_rate:0.5 ~decay_steps:0
          : S.t));
  raises
    (Invalid_argument "Schedule.cosine_decay: decay_steps must be positive")
    (fun () ->
      ignore (S.cosine_decay ~init_value:1.0 ~decay_steps:(-1) () : S.t));
  raises
    (Invalid_argument
       "Schedule.warmup_cosine_decay: warmup_steps must be positive") (fun () ->
      ignore
        (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:0
           ~decay_steps:10 ()
          : S.t));
  raises
    (Invalid_argument
       "Schedule.warmup_cosine_decay: decay_steps must be positive") (fun () ->
      ignore
        (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
           ~decay_steps:0 ()
          : S.t))

(* Gradient transformations *)

let test_global_norm () =
  (* sqrt (3^2 + 0^2 + 4^2 + 12^2) = 13 *)
  let grads = pair [| 3.0; 0.0 |] [| 4.0; 12.0 |] in
  equal (float 1e-6) 13.0 (Vega.global_norm (module Pair) grads)

let test_clip_by_global_norm_rescales () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Vega.clip_by_global_norm (module Pair) ~max_norm:1.0 grads in
  equal ~msg:"norm is the bound" (float 1e-6) 1.0
    (Vega.global_norm (module Pair) clipped);
  check_vec ~eps:1e-6 ~msg:"direction preserved" [| 0.6; 0.0 |] clipped.a;
  check_vec ~eps:1e-6 [| 0.8 |] clipped.b

let test_clip_by_global_norm_small () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Vega.clip_by_global_norm (module Pair) ~max_norm:10.0 grads in
  check_vec ~eps:0. [| 3.0; 0.0 |] clipped.a;
  check_vec ~eps:0. [| 4.0 |] clipped.b;
  let zeros = pair [| 0.0; 0.0 |] [| 0.0 |] in
  let clipped = Vega.clip_by_global_norm (module Pair) ~max_norm:1.0 zeros in
  check_vec ~eps:0. ~msg:"zero gradients pass through" [| 0.0 |] clipped.b

let test_clip_by_value () =
  let grads = pair [| -3.0; 0.2 |] [| 5.0 |] in
  let clipped = Vega.clip_by_value (module Pair) ~max:1.0 grads in
  check_vec ~eps:1e-7 [| -1.0; 0.2 |] clipped.a;
  check_vec ~eps:0. [| 1.0 |] clipped.b

let test_clip_validation () =
  let grads = pair [| 1.0 |] [| 1.0 |] in
  raises
    (Invalid_argument "Vega.clip_by_global_norm: expected max_norm > 0.0, got 0")
    (fun () -> Vega.clip_by_global_norm (module Pair) ~max_norm:0.0 grads);
  raises (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got -1")
    (fun () -> Vega.clip_by_value (module Pair) ~max:(-1.0) grads)

(* SGD *)

let test_sgd_first_step () =
  let params = vec [| 1.0; -2.0 |] in
  let grads = vec [| 0.5; -1.0 |] in
  let st = Vega.sgd_init (module Vec) params in
  (* Zero velocity: the first step is plain descent even with momentum. *)
  let params', st' =
    Vega.sgd_step (module Vec) ~lr:(lr64 0.1) ~momentum:0.9 st ~params ~grads
  in
  check_vec [| 0.95; -1.9 |] params';
  check_vec ~msg:"velocity is the gradient" [| 0.5; -1.0 |] st'.velocity

let test_sgd_velocity_threads () =
  let params = vec [| 0.0 |] in
  let st = Vega.sgd_init (module Vec) params in
  let params, st =
    Vega.sgd_step
      (module Vec)
      ~lr:(lr64 0.1) ~momentum:0.5 st ~params ~grads:(vec [| 1.0 |])
  in
  let _, st =
    Vega.sgd_step
      (module Vec)
      ~lr:(lr64 0.1) ~momentum:0.5 st ~params ~grads:(vec [| 2.0 |])
  in
  (* v2 = 0.5 *. v1 +. g2 = 0.5 *. 1. +. 2. *)
  check_vec [| 2.5 |] st.velocity;
  equal ~msg:"counter reads 2" int 2 (Int32.to_int (Nx.item [] st.step))

let test_sgd_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.sgd_step (module Pair) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  let params, _ =
    descend ~steps:100 ~step (params, Vega.sgd_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_momentum_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.sgd_step
      (module Pair)
      ~lr:(Vega.lr 0.05) ~momentum:0.9 st ~params ~grads
  in
  let params, _ =
    descend ~steps:200 ~step (params, Vega.sgd_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_pairs_leaves_structurally () =
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let grads = pair [| 0.0; 0.0 |] [| 1.0 |] in
  let st = Vega.sgd_init (module Pair) params in
  let params', _ =
    Vega.sgd_step (module Pair) ~lr:(Vega.lr 0.5) st ~params ~grads
  in
  check_vec ~eps:0. ~msg:"zero-gradient leaf untouched" [| 1.0; 2.0 |] params'.a;
  check_vec ~eps:0. [| 2.5 |] params'.b

(* Adam *)

let test_adam_first_step () =
  let b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and lr = 0.1 in
  let g = [| 4.0; -0.5; 0.0 |] in
  let params = vec [| 1.0; -2.0; 3.0 |] in
  let st = Vega.adam_init (module Vec) params in
  let params', st' =
    Vega.adam_step (module Vec) ~lr:(lr64 lr) st ~params ~grads:(vec g)
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
  equal ~msg:"step count" int 1 (Int32.to_int (Nx.item [] st'.step))

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
  let st = ref (Vega.adam_init (module Vec) !params) in
  List.iteri
    (fun i e ->
      let grads = Nx.mul_s (Nx.sub_s !params 1.0) 2.0 in
      let params', st' =
        Vega.adam_step (module Vec) ~lr:(lr64 lr) !st ~params:!params ~grads
      in
      params := params';
      st := st';
      check_vec ~msg:(Printf.sprintf "step %d" (i + 1)) [| e |] !params)
    expected

let test_adam_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.adam_step (module Pair) ~lr:(Vega.lr 0.02) st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Vega.adam_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

let test_adam_with_schedule_converges () =
  (* The learning rate comes from the state's own step counter through the
     schedule — the jitted loop's shape, run eagerly. *)
  let sched = S.cosine_decay ~init_value:0.1 ~decay_steps:300 () in
  let params = Lazy.force bowl_start in
  let state = ref (params, Vega.adam_init (module Pair) params) in
  for _k = 1 to 300 do
    let params, st = !state in
    let grads = bowl_grads params in
    state := Vega.adam_step (module Pair) ~lr:(sched st.step) st ~params ~grads
  done;
  is_true ~msg:"decayed steps settle at the bottom"
    (bowl_distance (fst !state) < 0.02)

let test_adam_zero_grads () =
  let params = vec [| 1.0; -2.0 |] in
  let st = Vega.adam_init (module Vec) params in
  let params', st' =
    Vega.adam_step
      (module Vec)
      ~lr:(Vega.lr 0.1) st ~params
      ~grads:(vec [| 0.0; 0.0 |])
  in
  check_vec ~eps:0. ~msg:"parameters unchanged" [| 1.0; -2.0 |] params';
  equal ~msg:"step still advances" int 1 (Int32.to_int (Nx.item [] st'.step))

let test_adam_step_is_pure () =
  let params = vec [| 3.0; -1.0 |] in
  let grads = vec [| 0.7; 0.3 |] in
  let st = Vega.adam_init (module Vec) params in
  let once, _ =
    Vega.adam_step (module Vec) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  let again, _ =
    Vega.adam_step (module Vec) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  check_vec ~eps:0. ~msg:"same state, same step" (Nx.to_array once) again

(* AdamW *)

let test_adamw_zero_decay_is_adam () =
  let grads_of params = Nx.mul_s (Nx.sub_s params 1.0) 2.0 in
  let run step =
    let params = ref (vec [| 3.0; -2.0 |]) in
    let st = ref (Vega.adam_init (module Vec) !params) in
    for _ = 1 to 5 do
      let params', st' = step !st ~params:!params ~grads:(grads_of !params) in
      params := params';
      st := st'
    done;
    !params
  in
  let adam =
    run (fun st ~params ~grads ->
        Vega.adam_step (module Vec) ~lr:(Vega.lr 0.1) st ~params ~grads)
  in
  let adamw =
    run (fun st ~params ~grads ->
        Vega.adamw_step
          (module Vec)
          ~lr:(Vega.lr 0.1) ~weight_decay:0.0 st ~params ~grads)
  in
  check_vec ~eps:0. (Nx.to_array adam) adamw

let test_adamw_decays_weights () =
  (* Zero gradients isolate the decay: p_k = p_0 (1 - lr wd)^k. A coupled (L2)
     decay would instead be distorted by the adaptive scaling. *)
  let lr = 0.1 and wd = 0.5 in
  let p0 = [| 2.0; -4.0 |] in
  let params = ref (vec p0) in
  let st = ref (Vega.adamw_init (module Vec) !params) in
  for _ = 1 to 3 do
    let params', st' =
      Vega.adamw_step
        (module Vec)
        ~lr:(lr64 lr) ~weight_decay:wd !st ~params:!params
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
    Vega.adamw_step
      (module Pair)
      ~lr:(Vega.lr 0.02) ~weight_decay:1e-3 st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Vega.adamw_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

(* A parameter structure may carry leaves that are not parameters. The canonical
   one is an RNG key, which has to sit in the structure to reach a compiled step
   as an input but is not something to optimize. Rune leaves its gradient slot
   at zero; the optimizers must leave the value alone. Adam is where it would
   show: its direction runs each leaf through a square root and a division. *)
module Stepper = struct
  type t = { w : Nx.float64_t; key : Nx.Rng.key }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { w = f t.w; key = f t.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w; key = f a.key b.key }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    f t.w;
    f t.key
end

let test_optimizers_carry_a_non_parameter_leaf () =
  let params =
    Stepper.
      {
        w = Nx.create Nx.float64 [| 3 |] [| 1.0; -2.0; 3.0 |];
        key = Nx.Rng.key 7;
      }
  in
  let grads =
    Stepper.
      {
        w = Nx.create Nx.float64 [| 3 |] [| 2.0; -4.0; 6.0 |];
        key = Nx.zeros Nx.int32 [| 2 |];
      }
  in
  let key_before = Nx.to_array params.Stepper.key in
  let check name (updated : Stepper.t) =
    is_true
      ~msg:(name ^ " moves the weight")
      (Nx.to_array updated.Stepper.w <> Nx.to_array params.Stepper.w);
    equal
      ~msg:(name ^ " leaves the key alone")
      (array int32) key_before
      (Nx.to_array updated.Stepper.key)
  in
  let sgd, _ =
    Vega.sgd_step
      (module Stepper)
      ~lr:(Vega.lr 0.1)
      (Vega.sgd_init (module Stepper) params)
      ~params ~grads
  in
  check "sgd" sgd;
  let momentum, _ =
    Vega.sgd_step
      (module Stepper)
      ~lr:(Vega.lr 0.1) ~momentum:0.9
      (Vega.sgd_init (module Stepper) params)
      ~params ~grads
  in
  check "sgd with momentum" momentum;
  let adam, _ =
    Vega.adam_step
      (module Stepper)
      ~lr:(Vega.lr 0.1)
      (Vega.adam_init (module Stepper) params)
      ~params ~grads
  in
  check "adam" adam;
  let adamw, _ =
    Vega.adamw_step
      (module Stepper)
      ~lr:(Vega.lr 0.1)
      (Vega.adamw_init (module Stepper) params)
      ~params ~grads
  in
  check "adamw" adamw

(* L-BFGS *)

(* The bowl as an objective returning its value and analytic gradient, the form
   the L-BFGS steps take. *)
let bowl (params : Pair.t) =
  let diff = Pair.map2 Nx.sub params (Lazy.force bowl_target) in
  let value = Nx.add (Nx.sum (Nx.square diff.a)) (Nx.sum (Nx.square diff.b)) in
  (value, bowl_grads params)

(* Rosenbrock's valley in float64, minimized at (1, 1): the classic
   ill-conditioned test a first-order method crawls along. *)
let rosenbrock (v : Vec.t) =
  let x = Nx.item [ 0 ] v and y = Nx.item [ 1 ] v in
  let value = ((1.0 -. x) ** 2.0) +. (100.0 *. ((y -. (x *. x)) ** 2.0)) in
  let gx = (-2.0 *. (1.0 -. x)) -. (400.0 *. x *. (y -. (x *. x))) in
  let gy = 200.0 *. (y -. (x *. x)) in
  (Nx.scalar Nx.float64 value, vec [| gx; gy |])

let iterations (st : (_, _) Vega.lbfgs_state) =
  Int32.to_int (Nx.item [] st.step)

let test_global_dot () =
  let a = pair [| 1.0; 2.0 |] [| 3.0 |] and b = pair [| 4.0; 5.0 |] [| 6.0 |] in
  let d = Vega.global_dot (module Pair) Nx.float64 a b in
  equal ~msg:"spans all leaves" (float 1e-12) 32.0 (Nx.item [] d);
  equal ~msg:"accumulates at the requested dtype" (float 1e-12) 32.0
    (Nx.item [] (Vega.global_dot (module Pair) Nx.float32 a b))

let test_lbfgs_init () =
  let params = Lazy.force bowl_start in
  let st = Vega.lbfgs_init (module Pair) ~history:3 bowl params in
  let value, _ = bowl params in
  equal ~msg:"value is the objective at the start" (float 1e-6)
    (Nx.item [] value) (Nx.item [] st.value);
  check_vec ~msg:"gradient is the objective's" ~eps:1e-6 [| 7.0; -5.0 |]
    (Nx.cast Nx.float64 st.grads.a);
  equal ~msg:"memory stacks history on a new axis" (array int) [| 3; 2 |]
    (Nx.shape st.s.a);
  equal ~msg:"memory of the second leaf" (array int) [| 3; 1 |]
    (Nx.shape st.y.b);
  equal ~msg:"no pair carries weight" (array float_exact) [| 0.; 0.; 0. |]
    (Nx.to_array st.rho);
  equal ~msg:"counter starts at 0" int 0 (iterations st);
  raises (Invalid_argument "Vega.lbfgs_init: expected history >= 1, got 0")
    (fun () -> Vega.lbfgs_init (module Pair) ~history:0 bowl params)

let test_lbfgs_fixed_step () =
  (* With an empty memory the direction is the negated gradient, so a fixed rate
     of 1/2 on the bowl (gradient 2 (p - target)) lands on the target in one
     step. *)
  let params = Lazy.force bowl_start in
  let st = Vega.lbfgs_init (module Pair) bowl params in
  let st = Vega.lbfgs_step (module Pair) ~lr:(Vega.lr 0.5) bowl st in
  let target = Lazy.force bowl_target in
  check_vec ~msg:"a lands on the target" ~eps:1e-6 (Nx.to_array target.a)
    (Nx.cast Nx.float64 st.params.a);
  check_vec ~msg:"b lands on the target" ~eps:1e-6 (Nx.to_array target.b)
    (Nx.cast Nx.float64 st.params.b);
  equal ~msg:"counter advances" int 1 (iterations st);
  (* The pair the step produced sits on top of the memory with positive
     curvature; the rest is still empty. *)
  is_true ~msg:"newest pair has weight" (Nx.item [ 0 ] st.rho > 0.0);
  equal ~msg:"older slots are empty" float_exact 0.0 (Nx.item [ 1 ] st.rho);
  check_vec ~msg:"newest s is the step taken" ~eps:1e-6
    (Nx.to_array (Nx.cast Nx.float64 (Nx.sub target.a params.a)))
    (Nx.cast Nx.float64 (Nx.get [ 0 ] st.s.a));
  (* A second fixed step from the minimum: the gradient is zero, so the point
     stays and the new pair, with no curvature, gets no weight. *)
  let st' = Vega.lbfgs_step (module Pair) ~lr:(Vega.lr 0.5) bowl st in
  check_vec ~msg:"stays at the minimum" ~eps:1e-6 (Nx.to_array target.a)
    (Nx.cast Nx.float64 st'.params.a);
  equal ~msg:"zero-curvature pair has no weight" float_exact 0.0
    (Nx.item [ 0 ] st'.rho);
  is_true ~msg:"previous pair shifted down" (Nx.item [ 1 ] st'.rho > 0.0)

let test_lbfgs_bowl_converges () =
  let st, status = Vega.minimize (module Pair) bowl (Lazy.force bowl_start) in
  is_true ~msg:"converged" (status = Vega.Converged);
  is_true ~msg:"reaches the target" (bowl_distance st.params < 1e-3);
  is_true
    ~msg:(Printf.sprintf "few iterations on a quadratic (%d)" (iterations st))
    (iterations st <= 10)

let test_lbfgs_rosenbrock_converges () =
  let st, status =
    Vega.minimize (module Vec) ~gtol:1e-8 rosenbrock (vec [| -1.2; 1.0 |])
  in
  is_true ~msg:"converged" (status = Vega.Converged);
  check_vec ~msg:"reaches (1, 1)" ~eps:1e-5 [| 1.0; 1.0 |] st.params;
  is_true
    ~msg:
      (Printf.sprintf "far fewer iterations than descent (%d)" (iterations st))
    (iterations st < 100);
  (* One pair of memory still beats descent, if less decisively. *)
  let st, status =
    Vega.minimize
      (module Vec)
      ~history:1 ~gtol:1e-8 rosenbrock
      (vec [| -1.2; 1.0 |])
  in
  is_true ~msg:"converged with history 1" (status = Vega.Converged);
  check_vec ~msg:"reaches (1, 1) with history 1" ~eps:1e-4 [| 1.0; 1.0 |]
    st.params

let test_lbfgs_stops () =
  let start = vec [| -1.2; 1.0 |] in
  let st, status = Vega.minimize (module Vec) ~max_iter:3 rosenbrock start in
  is_true ~msg:"budget exhausted" (status = Vega.Max_iter_reached);
  equal ~msg:"took exactly the budget" int 3 (iterations st);
  (* A gradient that points away from the descent of its objective: no trial
     decreases the value, so the step returns its input and the driver reports
     the failure without moving. *)
  let inconsistent v = (Nx.sum (Nx.square v), Nx.mul_s v (-2.0)) in
  let st, status = Vega.minimize (module Vec) inconsistent (vec [| 1.0 |]) in
  is_true ~msg:"line search failed" (status = Vega.Line_search_failed);
  equal ~msg:"no step taken" int 0 (iterations st);
  check_vec ~msg:"point unchanged" [| 1.0 |] st.params;
  (* Already at a stationary point: converged before any step. *)
  let st, status = Vega.minimize (module Vec) rosenbrock (vec [| 1.0; 1.0 |]) in
  is_true ~msg:"converged at the minimum" (status = Vega.Converged);
  equal ~msg:"without stepping" int 0 (iterations st);
  raises (Invalid_argument "Vega.minimize: expected gtol >= 0.0, got -1")
    (fun () -> Vega.minimize (module Vec) ~gtol:(-1.0) rosenbrock start);
  raises
    (Invalid_argument
       "Vega.lbfgs_step: expected max_linesearch_steps >= 1, got 0") (fun () ->
      Vega.lbfgs_step
        (module Vec)
        ~max_linesearch_steps:0 rosenbrock
        (Vega.lbfgs_init (module Vec) rosenbrock start))

let test_lbfgs_rejects_negative_curvature () =
  (* A concave objective: along the descent direction the gradient difference
     opposes the step, so every pair has [y . s < 0]. Such a pair must get no
     weight, and the next direction must fall back to the scaled gradient. *)
  let concave v = (Nx.neg (Nx.sum (Nx.square v)), Nx.mul_s v (-2.0)) in
  let st = Vega.lbfgs_init (module Vec) ~history:2 concave (vec [| 1.0 |]) in
  let st = Vega.lbfgs_step (module Vec) ~lr:(lr64 0.1) concave st in
  check_vec ~msg:"first step is descent" [| 1.2 |] st.params;
  is_true ~msg:"the pair has negative curvature"
    (Nx.item []
       (Vega.global_dot
          (module Vec)
          Nx.float64 (Nx.get [ 0 ] st.y) (Nx.get [ 0 ] st.s))
    < 0.0);
  equal ~msg:"and no weight" float_exact 0.0 (Nx.item [ 0 ] st.rho);
  (* With no weighted pair the direction is [-g] with unit scaling: plain
     descent again. *)
  let st' = Vega.lbfgs_step (module Vec) ~lr:(lr64 0.1) concave st in
  check_vec ~msg:"second step is plain descent" [| 1.44 |] st'.params;
  is_true ~msg:"the value keeps decreasing"
    (Nx.item [] st'.value < Nx.item [] st.value)

let test_lbfgs_memory_evicts () =
  (* Two slots, three steps: the newest pair sits on top, the second newest
     below it, and the first pair is gone. *)
  let start = Lazy.force bowl_start in
  let st0 = Vega.lbfgs_init (module Pair) ~history:2 bowl start in
  let advance st = Vega.lbfgs_step (module Pair) ~lr:(Vega.lr 0.1) bowl st in
  let st1 = advance st0 in
  let st2 = advance st1 in
  let st3 = advance st2 in
  let top (st : (Pair.t, _) Vega.lbfgs_state) =
    Nx.to_array (Nx.get [ 0 ] st.s.a)
  in
  equal ~msg:"memory has two slots" (array int) [| 2; 2 |] (Nx.shape st3.s.a);
  equal ~msg:"top is the latest step"
    (array (float 1e-6))
    (Nx.to_array (Nx.sub st3.params.a st2.params.a))
    (top st3);
  equal ~msg:"below it the previous step"
    (array (float 1e-6))
    (top st2)
    (Nx.to_array (Nx.get [ 1 ] st3.s.a));
  is_true ~msg:"the first pair is gone"
    (top st1 <> top st3 && top st1 <> Nx.to_array (Nx.get [ 1 ] st3.s.a));
  is_true ~msg:"both slots carry weight"
    (Array.for_all (fun r -> r > 0.0) (Nx.to_array st3.rho))

let test_lbfgs_stops_on_ftol () =
  (* [x^4 + x] has its minimum at an irrational point, so the gradient never
     reads exactly zero and [gtol = 0] cannot stop the run; a loose [ftol] stops
     it as soon as a step gains less than a hundredth of the value. *)
  let quartic v =
    ( Nx.add (Nx.sum (Nx.pow_s v 4.0)) (Nx.sum v),
      Nx.add_s (Nx.mul_s (Nx.pow_s v 3.0) 4.0) 1.0 )
  in
  let st, status =
    Vega.minimize (module Vec) ~gtol:0.0 ~ftol:1e-2 quartic (vec [| 1.0 |])
  in
  is_true ~msg:"converged" (status = Vega.Converged);
  is_true ~msg:"on the value, not the gradient" (Nx.item [ 0 ] st.grads <> 0.0);
  is_true ~msg:"after a few steps" (iterations st > 0);
  is_true ~msg:"having made progress" (Nx.item [] st.value < 0.0)

let test_lbfgs_carries_a_non_parameter_leaf () =
  let params =
    Stepper.
      {
        w = Nx.create Nx.float64 [| 3 |] [| 1.0; -2.0; 3.0 |];
        key = Nx.Rng.key 7;
      }
  in
  let objective (p : Stepper.t) =
    ( Nx.sum (Nx.square p.w),
      Stepper.{ w = Nx.mul_s p.w 2.0; key = Nx.zeros Nx.int32 [| 2 |] } )
  in
  let st, status = Vega.minimize (module Stepper) objective params in
  is_true ~msg:"converged" (status = Vega.Converged);
  check_vec ~msg:"weight minimized" ~eps:1e-4 [| 0.0; 0.0; 0.0 |] st.params.w;
  equal ~msg:"key left alone" (array int32)
    (Nx.to_array params.Stepper.key)
    (Nx.to_array st.params.Stepper.key)

let test_lbfgs_state_is_a_ptree () =
  let module Opt =
    Vega.Lbfgs_state
      (Pair)
      (struct
        type t = Nx.float32_elt
      end) in
  let st = Vega.lbfgs_init (module Pair) bowl (Lazy.force bowl_start) in
  let n = ref 0 in
  Opt.iter (fun _ -> incr n) st;
  (* params 2 + value + grads 2 + s 2 + y 2 + rho + step. *)
  equal ~msg:"leaf count" int 11 !n;
  let roundtrip =
    Opt.map (fun t -> t) (Opt.map2 (fun _ r -> r) (Opt.map (fun t -> t) st) st)
  in
  let expected = Vega.lbfgs_step (module Pair) bowl st in
  let stepped = Vega.lbfgs_step (module Pair) bowl roundtrip in
  check_vec ~msg:"roundtrip state steps identically"
    (Nx.to_array (Nx.cast Nx.float64 expected.params.a))
    (Nx.cast Nx.float64 stepped.params.a)

(* Optimizer state as a parameter tree *)

let test_adam_counter_advances () =
  let params = vec [| 1.0 |] in
  let grads = vec [| 1.0 |] in
  let st = ref (Vega.adam_init (module Vec) params) in
  let lr = Vega.lr 0.1 in
  (* The counter is a tensor leaf, so it advances through the state alone — the
     shape a compiled loop relies on. The bias corrections derive from it inside
     each step (checked against the closed form by the reference trajectory
     above). *)
  for _ = 1 to 5 do
    let _, st' = Vega.adam_step (module Vec) ~lr !st ~params ~grads in
    st := st'
  done;
  equal ~msg:"counter reads 5 after 5 steps" int 5
    (Int32.to_int (Nx.item [] !st.step))

let test_state_traversals () =
  (* Both state functors are parameter trees: map/map2/iter walk every tensor
     leaf — payload leaves, then the counter — in a fixed order. *)
  let module A = Vega.Adam_state (Pair) in
  let module Sg = Vega.Sgd_state (Pair) in
  let double (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    Nx.cast (Nx.dtype t) (Nx.mul_s (Nx.cast Nx.float64 t) 2.0)
  in
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let st = Vega.adam_init (module Pair) params in
  (* map doubles everything; iter counts the leaves it visits. *)
  let doubled = A.map double st in
  check_vec ~msg:"mu doubled" [| 0.0; 0.0 |] doubled.mu.a;
  let n = ref 0 in
  A.iter (fun _ -> incr n) st;
  (* 2 mu leaves + 2 nu leaves + step. *)
  equal ~msg:"adam leaf count" int 5 !n;
  (* map2 merges leafwise: take the right state everywhere. *)
  let st' = A.map2 (fun _ r -> r) st doubled in
  check_vec ~msg:"merged mu" [| 0.0 |] st'.mu.b;
  equal ~msg:"merged step" int32 0l (Nx.item [] st'.step);
  (* The sgd state: 2 velocity leaves + step. *)
  let sst = Vega.sgd_init (module Pair) params in
  let n = ref 0 in
  Sg.iter (fun _ -> incr n) sst;
  equal ~msg:"sgd leaf count" int 3 !n

let test_state_functor_is_a_ptree () =
  (* [Vega.Adam_state (P)] is an Nx.Ptree.S: the state can sit inside another
     tree — the shape a jitted step's input record takes. *)
  let module Opt = Vega.Adam_state (Pair) in
  let params = pair [| 1.0 |] [| 2.0 |] in
  let grads = pair [| 0.5 |] [| -0.5 |] in
  let st = Vega.adam_init (module Pair) params in
  (* Run the step through the state's own walker: embedding the state in an
     outer record and mapping over it must reproduce the state exactly. *)
  let roundtrip =
    Opt.map (fun t -> t) (Opt.map2 (fun _ r -> r) (Opt.map (fun t -> t) st) st)
  in
  let params', _ =
    Vega.adam_step (module Pair) ~lr:(Vega.lr 0.1) roundtrip ~params ~grads
  in
  let expected, _ =
    Vega.adam_step (module Pair) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  check_vec ~msg:"roundtrip state steps identically" (Nx.to_array expected.a)
    params'.a

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
        test "the counter advances as a tensor" test_adam_counter_advances;
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
    group "optimizer state as a parameter tree"
      [
        test "state traversals walk every leaf" test_state_traversals;
        test "the state functor is a Ptree.S that steps identically"
          test_state_functor_is_a_ptree;
      ];
    group "non-parameter leaves"
      [
        test "every optimizer carries them unchanged"
          test_optimizers_carry_a_non_parameter_leaf;
      ];
    group "lbfgs"
      [
        test "global dot spans all leaves" test_global_dot;
        test "init evaluates the objective and empties the memory"
          test_lbfgs_init;
        test "a fixed rate preconditions the gradient" test_lbfgs_fixed_step;
        test "minimize converges on a quadratic" test_lbfgs_bowl_converges;
        test "minimize converges on Rosenbrock" test_lbfgs_rosenbrock_converges;
        test "minimize reports why it stopped" test_lbfgs_stops;
        test "a pair without positive curvature gets no weight"
          test_lbfgs_rejects_negative_curvature;
        test "the memory evicts its oldest pair" test_lbfgs_memory_evicts;
        test "minimize stops on the value tolerance" test_lbfgs_stops_on_ftol;
        test "the step carries a non-parameter leaf"
          test_lbfgs_carries_a_non_parameter_leaf;
        test "the state functor is a Ptree.S that steps identically"
          test_lbfgs_state_is_a_ptree;
      ];
  ]

let () = run "vega structural" tests
