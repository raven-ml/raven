(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega's optimizers over parameter structures. *)

open Windtrap
module S = Vega.Schedule

(* A single float64 tensor, for analytic trajectory checks. *)
module Vec = struct
  type t = Nx.float64_t

  let ptree : t Nx.Ptree.t = Nx.Ptree.tensor
end

(* Two float32 leaves of different shapes, for structural pairing checks. *)
module Pair = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t }

  module Walked = struct
    type nonrec _ t = t

    let walk c { a; b } =
      let open Nx.Ptree.Walk in
      let a = field c "a" tensor a in
      let b = field c "b" tensor b in
      { a; b }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
  let sub p q = Nx.Ptree.map2 ptree (fun _ -> Nx.sub) p q
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
  Vega.global_norm Pair.ptree (Pair.sub params (Lazy.force bowl_target))

let visit_lines s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let descend ~steps ~step params =
  let rec loop k acc = if k = 0 then acc else loop (k - 1) (step acc) in
  loop steps params

(* Gradient transformations *)

let test_global_norm () =
  (* sqrt (3^2 + 0^2 + 4^2 + 12^2) = 13 *)
  let grads = pair [| 3.0; 0.0 |] [| 4.0; 12.0 |] in
  equal (float 1e-6) 13.0 (Vega.global_norm Pair.ptree grads)

let test_clip_by_global_norm_rescales () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Vega.clip_by_global_norm Pair.ptree ~max_norm:1.0 grads in
  equal ~msg:"norm is the bound" (float 1e-6) 1.0
    (Vega.global_norm Pair.ptree clipped);
  check_vec ~eps:1e-6 ~msg:"direction preserved" [| 0.6; 0.0 |] clipped.a;
  check_vec ~eps:1e-6 [| 0.8 |] clipped.b

let test_clip_by_global_norm_small () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Vega.clip_by_global_norm Pair.ptree ~max_norm:10.0 grads in
  check_vec ~eps:0. [| 3.0; 0.0 |] clipped.a;
  check_vec ~eps:0. [| 4.0 |] clipped.b;
  let zeros = pair [| 0.0; 0.0 |] [| 0.0 |] in
  let clipped = Vega.clip_by_global_norm Pair.ptree ~max_norm:1.0 zeros in
  check_vec ~eps:0. ~msg:"zero gradients pass through" [| 0.0 |] clipped.b

let test_clip_by_value () =
  let grads = pair [| -3.0; 0.2 |] [| 5.0 |] in
  let clipped = Vega.clip_by_value Pair.ptree ~max:1.0 grads in
  check_vec ~eps:1e-7 [| -1.0; 0.2 |] clipped.a;
  check_vec ~eps:0. [| 1.0 |] clipped.b

let test_clip_validation () =
  let grads = pair [| 1.0 |] [| 1.0 |] in
  raises
    (Invalid_argument "Vega.clip_by_global_norm: expected max_norm > 0.0, got 0")
    (fun () -> Vega.clip_by_global_norm Pair.ptree ~max_norm:0.0 grads);
  raises (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got -1")
    (fun () -> Vega.clip_by_value Pair.ptree ~max:(-1.0) grads)

(* SGD *)

let test_sgd_first_step () =
  let params = vec [| 1.0; -2.0 |] in
  let grads = vec [| 0.5; -1.0 |] in
  let st = Vega.sgd_init Vec.ptree params in
  (* Zero velocity: the first step is plain descent even with momentum. *)
  let params', st' =
    Vega.sgd_step Vec.ptree ~lr:(lr64 0.1) ~momentum:0.9 st ~params ~grads
  in
  check_vec [| 0.95; -1.9 |] params';
  check_vec ~msg:"velocity is the gradient" [| 0.5; -1.0 |] st'.velocity

let test_sgd_velocity_threads () =
  let params = vec [| 0.0 |] in
  let st = Vega.sgd_init Vec.ptree params in
  let params, st =
    Vega.sgd_step Vec.ptree ~lr:(lr64 0.1) ~momentum:0.5 st ~params
      ~grads:(vec [| 1.0 |])
  in
  let _, st =
    Vega.sgd_step Vec.ptree ~lr:(lr64 0.1) ~momentum:0.5 st ~params
      ~grads:(vec [| 2.0 |])
  in
  (* v2 = 0.5 *. v1 +. g2 = 0.5 *. 1. +. 2. *)
  check_vec [| 2.5 |] st.velocity;
  equal ~msg:"counter reads 2" int 2 (Int32.to_int (Nx.item [] st.step))

let test_sgd_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.sgd_step Pair.ptree ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  let params, _ =
    descend ~steps:100 ~step (params, Vega.sgd_init Pair.ptree params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_momentum_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.sgd_step Pair.ptree ~lr:(Vega.lr 0.05) ~momentum:0.9 st ~params ~grads
  in
  let params, _ =
    descend ~steps:200 ~step (params, Vega.sgd_init Pair.ptree params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_pairs_leaves_structurally () =
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let grads = pair [| 0.0; 0.0 |] [| 1.0 |] in
  let st = Vega.sgd_init Pair.ptree params in
  let params', _ =
    Vega.sgd_step Pair.ptree ~lr:(Vega.lr 0.5) st ~params ~grads
  in
  check_vec ~eps:0. ~msg:"zero-gradient leaf untouched" [| 1.0; 2.0 |] params'.a;
  check_vec ~eps:0. [| 2.5 |] params'.b

(* Adam *)

let test_adam_first_step () =
  let b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and lr = 0.1 in
  let g = [| 4.0; -0.5; 0.0 |] in
  let params = vec [| 1.0; -2.0; 3.0 |] in
  let st = Vega.adam_init Vec.ptree params in
  let params', st' =
    Vega.adam_step Vec.ptree ~lr:(lr64 lr) st ~params ~grads:(vec g)
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
  let st = ref (Vega.adam_init Vec.ptree !params) in
  List.iteri
    (fun i e ->
      let grads = Nx.mul_s (Nx.sub_s !params 1.0) 2.0 in
      let params', st' =
        Vega.adam_step Vec.ptree ~lr:(lr64 lr) !st ~params:!params ~grads
      in
      params := params';
      st := st';
      check_vec ~msg:(Printf.sprintf "step %d" (i + 1)) [| e |] !params)
    expected

let test_adam_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.adam_step Pair.ptree ~lr:(Vega.lr 0.02) st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Vega.adam_init Pair.ptree params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

let test_adam_with_schedule_converges () =
  (* The learning rate comes from the state's own step counter through the
     schedule — the jitted loop's shape, run eagerly. *)
  let sched = S.cosine_decay ~init_value:0.1 ~decay_steps:300 () in
  let params = Lazy.force bowl_start in
  let state = ref (params, Vega.adam_init Pair.ptree params) in
  for _k = 1 to 300 do
    let params, st = !state in
    let grads = bowl_grads params in
    state := Vega.adam_step Pair.ptree ~lr:(sched st.step) st ~params ~grads
  done;
  is_true ~msg:"decayed steps settle at the bottom"
    (bowl_distance (fst !state) < 0.02)

let test_adam_zero_grads () =
  let params = vec [| 1.0; -2.0 |] in
  let st = Vega.adam_init Vec.ptree params in
  let params', st' =
    Vega.adam_step Vec.ptree ~lr:(Vega.lr 0.1) st ~params
      ~grads:(vec [| 0.0; 0.0 |])
  in
  check_vec ~eps:0. ~msg:"parameters unchanged" [| 1.0; -2.0 |] params';
  equal ~msg:"step still advances" int 1 (Int32.to_int (Nx.item [] st'.step))

let test_adam_step_is_pure () =
  let params = vec [| 3.0; -1.0 |] in
  let grads = vec [| 0.7; 0.3 |] in
  let st = Vega.adam_init Vec.ptree params in
  let once, _ = Vega.adam_step Vec.ptree ~lr:(Vega.lr 0.1) st ~params ~grads in
  let again, _ = Vega.adam_step Vec.ptree ~lr:(Vega.lr 0.1) st ~params ~grads in
  check_vec ~eps:0. ~msg:"same state, same step" (Nx.to_array once) again

(* AdamW *)

let test_adamw_zero_decay_is_adam () =
  let grads_of params = Nx.mul_s (Nx.sub_s params 1.0) 2.0 in
  let run step =
    let params = ref (vec [| 3.0; -2.0 |]) in
    let st = ref (Vega.adam_init Vec.ptree !params) in
    for _ = 1 to 5 do
      let params', st' = step !st ~params:!params ~grads:(grads_of !params) in
      params := params';
      st := st'
    done;
    !params
  in
  let adam =
    run (fun st ~params ~grads ->
        Vega.adam_step Vec.ptree ~lr:(Vega.lr 0.1) st ~params ~grads)
  in
  let adamw =
    run (fun st ~params ~grads ->
        Vega.adamw_step Vec.ptree ~lr:(Vega.lr 0.1) ~weight_decay:0.0 st ~params
          ~grads)
  in
  check_vec ~eps:0. (Nx.to_array adam) adamw

let test_adamw_decays_weights () =
  (* Zero gradients isolate the decay: p_k = p_0 (1 - lr wd)^k. A coupled (L2)
     decay would instead be distorted by the adaptive scaling. *)
  let lr = 0.1 and wd = 0.5 in
  let p0 = [| 2.0; -4.0 |] in
  let params = ref (vec p0) in
  let st = ref (Vega.adamw_init Vec.ptree !params) in
  for _ = 1 to 3 do
    let params', st' =
      Vega.adamw_step Vec.ptree ~lr:(lr64 lr) ~weight_decay:wd !st
        ~params:!params
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
    Vega.adamw_step Pair.ptree ~lr:(Vega.lr 0.02) ~weight_decay:1e-3 st ~params
      ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Vega.adamw_init Pair.ptree params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

let test_steps_validate () =
  let params = vec [| 1.0 |] in
  let lr = Vega.lr 0.1 and p = Vec.ptree in
  let rejects name f =
    raises_match ~msg:name Exn.invalid_arg (fun () ->
        ignore (f ~params ~grads:params : Vec.t * _))
  in
  rejects "sgd momentum"
    (Vega.sgd_step p ~lr ~momentum:1.0 (Vega.sgd_init p params));
  rejects "adam b1" (Vega.adam_step p ~lr ~b1:1.0 (Vega.adam_init p params));
  rejects "adam eps" (Vega.adam_step p ~lr ~eps:0.0 (Vega.adam_init p params));
  rejects "adamw weight_decay"
    (Vega.adamw_step p ~lr ~weight_decay:(-0.1) (Vega.adamw_init p params))

(* A parameter structure may carry leaves that are not parameters. The canonical
   one is an RNG key, which has to sit in the structure to reach a compiled step
   as an input but is not something to optimize. Rune leaves its gradient slot
   at zero; the optimizers must leave the value alone. Adam is where it would
   show: its direction runs each leaf through a square root and a division. *)
module Stepper = struct
  type t = { w : Nx.float64_t; key : Nx.Rng.key }

  module Walked = struct
    type nonrec _ t = t

    let walk c t =
      let open Nx.Ptree.Walk in
      let w = field c "w" tensor t.w in
      let key = field c "key" tensor t.key in
      { w; key }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
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
    Vega.sgd_step Stepper.ptree ~lr:(Vega.lr 0.1)
      (Vega.sgd_init Stepper.ptree params)
      ~params ~grads
  in
  check "sgd" sgd;
  let momentum, _ =
    Vega.sgd_step Stepper.ptree ~lr:(Vega.lr 0.1) ~momentum:0.9
      (Vega.sgd_init Stepper.ptree params)
      ~params ~grads
  in
  check "sgd with momentum" momentum;
  let adam, _ =
    Vega.adam_step Stepper.ptree ~lr:(Vega.lr 0.1)
      (Vega.adam_init Stepper.ptree params)
      ~params ~grads
  in
  check "adam" adam;
  let adamw, _ =
    Vega.adamw_step Stepper.ptree ~lr:(Vega.lr 0.1)
      (Vega.adamw_init Stepper.ptree params)
      ~params ~grads
  in
  check "adamw" adamw;
  let p = Stepper.ptree and lr = Vega.lr 0.1 in
  let step name (updated, _) = check name updated in
  step "lars" (Vega.lars_step p ~lr (Vega.lars_init p params) ~params ~grads);
  step "radam" (Vega.radam_step p ~lr (Vega.radam_init p params) ~params ~grads);
  step "lamb" (Vega.lamb_step p ~lr (Vega.lamb_init p params) ~params ~grads);
  step "rmsprop"
    (Vega.rmsprop_step p ~lr ~momentum:0.9
       (Vega.rmsprop_init p params)
       ~params ~grads);
  step "adagrad"
    (Vega.adagrad_step p ~lr (Vega.adagrad_init p params) ~params ~grads);
  step "adan" (Vega.adan_step p ~lr (Vega.adan_init p params) ~params ~grads);
  step "lion" (Vega.lion_step p ~lr (Vega.lion_init p params) ~params ~grads);
  step "adafactor"
    (Vega.adafactor_step p ~lr (Vega.adafactor_init p params) ~params ~grads)

(* Lion, RAdam, LAMB, LARS, Adafactor, Adan, RMSprop and Adagrad *)

(* A float64 matrix and vector: Adafactor factors the one and not the other, and
   LARS's and LAMB's trust ratios differ between them. *)
module Wb = struct
  type t = { w : Nx.float64_t; b : Nx.float64_t }

  module Walked = struct
    type nonrec _ t = t

    let walk c { w; b } =
      let open Nx.Ptree.Walk in
      let w = field c "w" tensor w in
      let b = field c "b" tensor b in
      { w; b }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
end

let wb w b =
  {
    Wb.w = Nx.create Nx.float64 [| 2; 3 |] w;
    b = Nx.create Nx.float64 [| 3 |] b;
  }

let wb_target =
  lazy (wb [| 0.1; 0.2; -0.3; 0.4; -0.5; 0.6 |] [| -0.7; 0.8; 0.9 |])

let wb_grads (x : Wb.t) =
  let t = Lazy.force wb_target in
  { Wb.w = Nx.mul_s (Nx.sub x.w t.w) 2.0; b = Nx.mul_s (Nx.sub x.b t.b) 2.0 }

(* [follows ~init ~step (w, b)] checks that eight steps from a fixed start on
   the gradient of [||p - wb_target||^2] end at [w] and [b]. Expected values:
   the per-tensor implementation at bfea66e15, run leaf by leaf. It took its
   rates and weight decays as float32 schedules, so rates here are [Vega.lr] and
   weight decays are [rounded] to float32. *)
let rounded x = Int32.float_of_bits (Int32.bits_of_float x)

let follows ~init ~step (w, b) () =
  let rec go k params st =
    if k = 0 then params
    else
      let params, st = step st ~params ~grads:(wb_grads params) in
      go (k - 1) params st
  in
  let params = wb [| 0.5; -1.2; 2.0; -0.3; 0.8; 1.5 |] [| 1.0; -2.0; 0.25 |] in
  let params = go 8 params (init Wb.ptree params) in
  check_vec ~eps:1e-12 ~msg:"w" w params.w;
  check_vec ~eps:1e-12 ~msg:"b" b params.b

let test_rmsprop_trajectory =
  follows ~init:Vega.rmsprop_init
    ~step:(fun st -> Vega.rmsprop_step Wb.ptree ~lr:(Vega.lr 0.01) st)
    ( [|
        0.36243613887935616;
        -1.0546103319516811;
        1.8535037661485867;
        -0.1575807154685458;
        0.65483149409076835;
        1.3562348676790772;
      |],
      [| 0.85410748659626345; -1.8532018103325221; 0.39194368077038816 |] )

let test_rmsprop_momentum_trajectory =
  follows ~init:Vega.rmsprop_init
    ~step:(fun st ->
      Vega.rmsprop_step Wb.ptree ~lr:(Vega.lr 0.01) ~momentum:0.9 st)
    ( [|
        0.019482040910134224;
        -0.63248913074282231;
        1.4221199669573561;
        0.23713084350163544;
        0.23462291139457309;
        0.9486450535003067;
      |],
      [| 0.42771408903600838; -1.4193780845760426; 0.78192829333036751 |] )

let test_adagrad_trajectory =
  follows ~init:Vega.adagrad_init
    ~step:(fun st -> Vega.adagrad_step Wb.ptree ~lr:(Vega.lr 0.1) st)
    ( [|
        0.18167555199967195;
        -0.79131715880629638;
        1.5795826597090736;
        0.075287325474801839;
        0.39371468960115386;
        1.1092891092927324;
      |],
      [| 0.58592940989034603; -1.576460256104508; 0.61975162231019465 |] )

let test_lion_trajectory =
  follows ~init:Vega.lion_init
    ~step:(fun st -> Vega.lion_step Wb.ptree ~lr:(Vega.lr 0.01) st)
    ( [|
        0.42000000178813934;
        -1.1200000017881393;
        1.9200000017881393;
        -0.22000000178813933;
        0.72000000178813939;
        1.4200000017881393;
      |],
      [| 0.92000000178813934; -1.9200000017881393; 0.32999999821186066 |] )

(* Eight steps cross from momentum steps to rectified ones: with [b2 = 0.999],
   [rho] is below 5 up to the fifth step and above from the sixth. *)
let test_radam_trajectory =
  follows ~init:Vega.radam_init
    ~step:(fun st -> Vega.radam_step Wb.ptree ~lr:(Vega.lr 0.01) st)
    ( [|
        0.4598985836034043;
        -1.0620572493260865;
        1.7740000784706591;
        -0.2305461712107803;
        0.67184137996033355;
        1.4109779052542408;
      |],
      [| 0.83270485829882479; -1.7250794297474281; 0.31456176161030008 |] )

let test_lamb_trajectory =
  follows ~init:Vega.lamb_init
    ~step:(fun st ->
      Vega.lamb_step Wb.ptree ~lr:(Vega.lr 0.01) ~weight_decay:(rounded 0.01) st)
    ( [|
        0.4076288911828635;
        -1.1064478293192996;
        1.9056372630040368;
        -0.20747395071242003;
        0.70683175651562802;
        1.4062769980333993;
      |],
      [| 0.89842985775586015; -1.8973554242093986; 0.35000475942674164 |] )

(* LARS's momentum accumulates the trust-scaled update, as in the paper. The
   per-tensor implementation's LARS took the trust ratio of the accumulated
   gradient instead, so with momentum the expected values are its decay, trust
   ratio, momentum and rate transforms chained in the paper's order, and without
   momentum its LARS itself. *)
let test_lars_trajectory =
  follows ~init:Vega.lars_init
    ~step:(fun st ->
      Vega.lars_step Wb.ptree ~lr:(Vega.lr 0.1) ~weight_decay:(rounded 0.01) st)
    ( [|
        -0.10311105473850804;
        0.90676805714082909;
        -1.4613330099492292;
        0.75113640978280594;
        -1.1539299762342272;
        0.14019060335384598;
      |],
      [| -1.1027388508197593; 1.4655109505693997; 1.0500890496290238 |] )

let test_lars_nesterov_trajectory =
  follows ~init:Vega.lars_init
    ~step:(fun st ->
      Vega.lars_step Wb.ptree ~lr:(Vega.lr 0.1) ~weight_decay:(rounded 0.01)
        ~nesterov:true st)
    ( [|
        0.0022904246280579807;
        0.5385830136729276;
        -0.85642017182001073;
        0.56743668858435314;
        -0.81245536976550736;
        0.37783493259799195;
      |],
      [| -1.1133674069043227; 1.4830278083976729; 1.0541331996352461 |] )

let test_lars_no_momentum_trajectory =
  follows ~init:Vega.lars_init
    ~step:(fun st ->
      Vega.lars_step Wb.ptree ~lr:(Vega.lr 0.1) ~weight_decay:(rounded 0.01)
        ~momentum:0.0 st)
    ( [|
        0.28644449346382289;
        -0.45401480198034716;
        0.77437709287124723;
        0.072196739996935169;
        0.10813321604287521;
        1.0185052865274873;
      |],
      [| 0.32932873062805601; -0.89467081119719283; 0.50518943463478394 |] )

let test_adan_trajectory =
  follows ~init:Vega.adan_init
    ~step:(fun st ->
      Vega.adan_step Wb.ptree ~lr:(Vega.lr 0.01) ~weight_decay:(rounded 0.02) st)
    ( [|
        0.45683347270888586;
        -1.1552658787847068;
        1.9539196198792232;
        -0.25687545160538033;
        0.75591644939640612;
        1.454890507770245;
      |],
      [| 0.95555272431779104; -1.9538996006246216; 0.29222581010000648 |] )

(* The per-tensor implementation's Adafactor built in a rate of [1e-3 / sqrt
   t]. *)
let adafactor_step (st : Wb.t Vega.adafactor_state) ~params ~grads =
  let t = Nx.add_s (Nx.cast Nx.float64 st.step) 1.0 in
  let lr = Nx.mul_s (Nx.rsqrt t) 1e-3 in
  Vega.adafactor_step Wb.ptree ~lr st ~params ~grads

let test_adafactor_trajectory =
  follows
    ~init:(fun p x -> Vega.adafactor_init p x)
    ~step:adafactor_step
    ( [|
        0.49762229479050468;
        -1.196488209408342;
        1.9955374424672774;
        -0.2934547708737284;
        0.79486616423488821;
        1.4972504760921053;
      |],
      [| 0.99563024375684173; -1.9956295831398083; 0.25436703443267278 |] )

let test_adafactor_unfactored_trajectory =
  follows
    ~init:(fun p x -> Vega.adafactor_init p ~factored:false x)
    ~step:adafactor_step
    ( [|
        0.49563572869059364;
        -1.1956306043108442;
        1.9956298050287165;
        -0.2956326503530885;
        0.79563076152346657;
        1.4956317403913488;
      |],
      [| 0.99563024375684173; -1.9956295831398083; 0.25436703443267278 |] )

let test_adafactor_factors_matrices () =
  let params = wb (Array.make 6 1.0) (Array.make 3 1.0) in
  let shapes (st : Wb.t Vega.adafactor_state) =
    List.map
      (fun (x : Wb.t) -> (Nx.shape x.w, Nx.shape x.b))
      [ st.nu_row; st.nu_col; st.nu ]
  in
  let shape = array int in
  let parts = list (Windtrap.pair shape shape) in
  equal ~msg:"factored" parts
    [ ([| 2; 1 |], [||]); ([| 1; 3 |], [||]); ([||], [| 3 |]) ]
    (shapes (Vega.adafactor_init Wb.ptree params));
  equal ~msg:"unfactored" parts
    [ ([||], [||]); ([||], [||]); ([| 2; 3 |], [| 3 |]) ]
    (shapes (Vega.adafactor_init Wb.ptree ~factored:false params))

let test_ported_steps_validate () =
  let params = vec [| 1.0 |] in
  let lr = Vega.lr 0.1 in
  let rejects name f =
    raises_match ~msg:name Exn.invalid_arg (fun () ->
        ignore (f ~params ~grads:params : Vec.t * _))
  in
  let p = Vec.ptree in
  rejects "lion b1" (Vega.lion_step p ~lr ~b1:1.0 (Vega.lion_init p params));
  rejects "radam b2"
    (Vega.radam_step p ~lr ~b2:(-0.1) (Vega.radam_init p params));
  rejects "lamb weight_decay"
    (Vega.lamb_step p ~lr ~weight_decay:(-1.0) (Vega.lamb_init p params));
  rejects "lars momentum"
    (Vega.lars_step p ~lr ~momentum:1.0 (Vega.lars_init p params));
  rejects "adafactor eps"
    (Vega.adafactor_step p ~lr ~eps:0.0 (Vega.adafactor_init p params));
  rejects "adan b3" (Vega.adan_step p ~lr ~b3:1.0 (Vega.adan_init p params));
  rejects "rmsprop decay"
    (Vega.rmsprop_step p ~lr ~decay:1.5 (Vega.rmsprop_init p params));
  rejects "adagrad eps"
    (Vega.adagrad_step p ~lr ~eps:(-1.0) (Vega.adagrad_init p params))

(* L-BFGS *)

(* The bowl as an objective returning its value and analytic gradient, the form
   the L-BFGS steps take. *)
let bowl (params : Pair.t) =
  let diff = Pair.sub params (Lazy.force bowl_target) in
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
  let d = Vega.global_dot Pair.ptree Nx.float64 a b in
  equal ~msg:"spans all leaves" (float 1e-12) 32.0 (Nx.item [] d);
  equal ~msg:"accumulates at the requested dtype" (float 1e-12) 32.0
    (Nx.item [] (Vega.global_dot Pair.ptree Nx.float32 a b))

let test_lbfgs_init () =
  let params = Lazy.force bowl_start in
  let st = Vega.lbfgs_init Pair.ptree ~history:3 bowl params in
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
    (fun () -> Vega.lbfgs_init Pair.ptree ~history:0 bowl params)

let test_lbfgs_fixed_step () =
  (* With an empty memory the direction is the negated gradient, so a fixed rate
     of 1/2 on the bowl (gradient 2 (p - target)) lands on the target in one
     step. *)
  let params = Lazy.force bowl_start in
  let st = Vega.lbfgs_init Pair.ptree bowl params in
  let st = Vega.lbfgs_step Pair.ptree ~lr:(Vega.lr 0.5) bowl st in
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
  let st' = Vega.lbfgs_step Pair.ptree ~lr:(Vega.lr 0.5) bowl st in
  check_vec ~msg:"stays at the minimum" ~eps:1e-6 (Nx.to_array target.a)
    (Nx.cast Nx.float64 st'.params.a);
  equal ~msg:"zero-curvature pair has no weight" float_exact 0.0
    (Nx.item [ 0 ] st'.rho);
  is_true ~msg:"previous pair shifted down" (Nx.item [ 1 ] st'.rho > 0.0)

let test_lbfgs_bowl_converges () =
  let st, status = Vega.minimize Pair.ptree bowl (Lazy.force bowl_start) in
  is_true ~msg:"converged" (status = Vega.Converged);
  is_true ~msg:"reaches the target" (bowl_distance st.params < 1e-3);
  is_true
    ~msg:(Printf.sprintf "few iterations on a quadratic (%d)" (iterations st))
    (iterations st <= 10)

let test_lbfgs_rosenbrock_converges () =
  let st, status =
    Vega.minimize Vec.ptree ~gtol:1e-8 rosenbrock (vec [| -1.2; 1.0 |])
  in
  is_true ~msg:"converged" (status = Vega.Converged);
  check_vec ~msg:"reaches (1, 1)" ~eps:1e-5 [| 1.0; 1.0 |] st.params;
  is_true
    ~msg:
      (Printf.sprintf "far fewer iterations than descent (%d)" (iterations st))
    (iterations st < 100);
  (* One pair of memory still beats descent, if less decisively. *)
  let st, status =
    Vega.minimize Vec.ptree ~history:1 ~gtol:1e-8 rosenbrock
      (vec [| -1.2; 1.0 |])
  in
  is_true ~msg:"converged with history 1" (status = Vega.Converged);
  check_vec ~msg:"reaches (1, 1) with history 1" ~eps:1e-4 [| 1.0; 1.0 |]
    st.params

let test_lbfgs_stops () =
  let start = vec [| -1.2; 1.0 |] in
  let st, status = Vega.minimize Vec.ptree ~max_iter:3 rosenbrock start in
  is_true ~msg:"budget exhausted" (status = Vega.Max_iter_reached);
  equal ~msg:"took exactly the budget" int 3 (iterations st);
  (* A gradient that points away from the descent of its objective: no trial
     decreases the value, so the step returns its input and the driver reports
     the failure without moving. *)
  let inconsistent v = (Nx.sum (Nx.square v), Nx.mul_s v (-2.0)) in
  let st, status = Vega.minimize Vec.ptree inconsistent (vec [| 1.0 |]) in
  is_true ~msg:"line search failed" (status = Vega.Line_search_failed);
  equal ~msg:"no step taken" int 0 (iterations st);
  check_vec ~msg:"point unchanged" [| 1.0 |] st.params;
  (* Already at a stationary point: converged before any step. *)
  let st, status = Vega.minimize Vec.ptree rosenbrock (vec [| 1.0; 1.0 |]) in
  is_true ~msg:"converged at the minimum" (status = Vega.Converged);
  equal ~msg:"without stepping" int 0 (iterations st);
  raises (Invalid_argument "Vega.minimize: expected gtol >= 0.0, got -1")
    (fun () -> Vega.minimize Vec.ptree ~gtol:(-1.0) rosenbrock start);
  raises
    (Invalid_argument
       "Vega.lbfgs_step: expected max_linesearch_steps >= 1, got 0") (fun () ->
      Vega.lbfgs_step Vec.ptree ~max_linesearch_steps:0 rosenbrock
        (Vega.lbfgs_init Vec.ptree rosenbrock start))

let test_lbfgs_rejects_negative_curvature () =
  (* A concave objective: along the descent direction the gradient difference
     opposes the step, so every pair has [y . s < 0]. Such a pair must get no
     weight, and the next direction must fall back to the scaled gradient. *)
  let concave v = (Nx.neg (Nx.sum (Nx.square v)), Nx.mul_s v (-2.0)) in
  let st = Vega.lbfgs_init Vec.ptree ~history:2 concave (vec [| 1.0 |]) in
  let st = Vega.lbfgs_step Vec.ptree ~lr:(lr64 0.1) concave st in
  check_vec ~msg:"first step is descent" [| 1.2 |] st.params;
  is_true ~msg:"the pair has negative curvature"
    (Nx.item []
       (Vega.global_dot Vec.ptree Nx.float64 (Nx.get [ 0 ] st.y)
          (Nx.get [ 0 ] st.s))
    < 0.0);
  equal ~msg:"and no weight" float_exact 0.0 (Nx.item [ 0 ] st.rho);
  (* With no weighted pair the direction is [-g] with unit scaling: plain
     descent again. *)
  let st' = Vega.lbfgs_step Vec.ptree ~lr:(lr64 0.1) concave st in
  check_vec ~msg:"second step is plain descent" [| 1.44 |] st'.params;
  is_true ~msg:"the value keeps decreasing"
    (Nx.item [] st'.value < Nx.item [] st.value)

let test_lbfgs_memory_evicts () =
  (* Two slots, three steps: the newest pair sits on top, the second newest
     below it, and the first pair is gone. *)
  let start = Lazy.force bowl_start in
  let st0 = Vega.lbfgs_init Pair.ptree ~history:2 bowl start in
  let advance st = Vega.lbfgs_step Pair.ptree ~lr:(Vega.lr 0.1) bowl st in
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
    Vega.minimize Vec.ptree ~gtol:0.0 ~ftol:1e-2 quartic (vec [| 1.0 |])
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
  let st, status = Vega.minimize Stepper.ptree objective params in
  is_true ~msg:"converged" (status = Vega.Converged);
  check_vec ~msg:"weight minimized" ~eps:1e-4 [| 0.0; 0.0; 0.0 |] st.params.w;
  equal ~msg:"key left alone" (array int32)
    (Nx.to_array params.Stepper.key)
    (Nx.to_array st.params.Stepper.key)

let test_lbfgs_state_is_a_ptree () =
  let opt = Vega.lbfgs_ptree Pair.ptree in
  let st = Vega.lbfgs_init Pair.ptree bowl (Lazy.force bowl_start) in
  equal ~msg:"visits" (list string)
    [
      "params.a: a leaf";
      "params.b: a leaf";
      "value: a leaf";
      "grads.a: a leaf";
      "grads.b: a leaf";
      "s.a: a leaf";
      "s.b: a leaf";
      "y.a: a leaf";
      "y.b: a leaf";
      "rho: a leaf";
      "step: a leaf";
    ]
    (visit_lines opt st);
  let roundtrip =
    Nx.Ptree.map opt (fun _ t -> t) (Nx.Ptree.map2 opt (fun _ _ r -> r) st st)
  in
  let expected = Vega.lbfgs_step Pair.ptree bowl st in
  let stepped = Vega.lbfgs_step Pair.ptree bowl roundtrip in
  check_vec ~msg:"roundtrip state steps identically"
    (Nx.to_array (Nx.cast Nx.float64 expected.params.a))
    (Nx.cast Nx.float64 stepped.params.a)

(* Optimizer state as a parameter tree *)

let test_adam_counter_advances () =
  let params = vec [| 1.0 |] in
  let grads = vec [| 1.0 |] in
  let st = ref (Vega.adam_init Vec.ptree params) in
  let lr = Vega.lr 0.1 in
  (* The counter is a tensor leaf, so it advances through the state alone — the
     shape a compiled loop relies on. The bias corrections derive from it inside
     each step (checked against the closed form by the reference trajectory
     above). *)
  for _ = 1 to 5 do
    let _, st' = Vega.adam_step Vec.ptree ~lr !st ~params ~grads in
    st := st'
  done;
  equal ~msg:"counter reads 5 after 5 steps" int 5
    (Int32.to_int (Nx.item [] !st.step))

let test_state_visits () =
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  equal ~msg:"adam" (list string)
    [
      "mu.a: a leaf";
      "mu.b: a leaf";
      "nu.a: a leaf";
      "nu.b: a leaf";
      "step: a leaf";
    ]
    (visit_lines
       (Vega.adam_ptree Pair.ptree)
       (Vega.adam_init Pair.ptree params));
  equal ~msg:"sgd" (list string)
    [ "velocity.a: a leaf"; "velocity.b: a leaf"; "step: a leaf" ]
    (visit_lines (Vega.sgd_ptree Pair.ptree) (Vega.sgd_init Pair.ptree params));
  let visits parts =
    List.concat_map (fun f -> [ f ^ ".a: a leaf"; f ^ ".b: a leaf" ]) parts
    @ [ "step: a leaf" ]
  in
  let p = Pair.ptree in
  equal ~msg:"rmsprop" (list string)
    (visits [ "nu"; "velocity" ])
    (visit_lines (Vega.rmsprop_ptree p) (Vega.rmsprop_init p params));
  equal ~msg:"adagrad" (list string)
    (visits [ "sum_of_squares" ])
    (visit_lines (Vega.adagrad_ptree p) (Vega.adagrad_init p params));
  equal ~msg:"adan" (list string)
    (visits [ "mu"; "delta"; "nu"; "prev_grads" ])
    (visit_lines (Vega.adan_ptree p) (Vega.adan_init p params));
  equal ~msg:"lion" (list string) (visits [ "mu" ])
    (visit_lines (Vega.lion_ptree p) (Vega.lion_init p params));
  equal ~msg:"adafactor" (list string)
    (visits [ "nu_row"; "nu_col"; "nu" ])
    (visit_lines (Vega.adafactor_ptree p) (Vega.adafactor_init p params));
  let nested = Nx.Ptree.list Vec.ptree in
  equal ~msg:"a state reports what its parameters report" (list string)
    [
      "mu: length 2";
      "mu.0: a leaf";
      "mu.1: a leaf";
      "nu: length 2";
      "nu.0: a leaf";
      "nu.1: a leaf";
      "step: a leaf";
    ]
    (visit_lines (Vega.adam_ptree nested)
       (Vega.adam_init nested [ vec [| 1.0 |]; vec [| 2.0 |] ]))

let test_state_is_a_ptree () =
  (* The state's structure serves Nx.Ptree's operations, and a state nested
     beside the parameters walks back to itself. *)
  let opt = Vega.adam_ptree Pair.ptree in
  let both = Nx.Ptree.pair Pair.ptree opt in
  let params = pair [| 1.0 |] [| 2.0 |] in
  let grads = pair [| 0.5 |] [| -0.5 |] in
  let st = Vega.adam_init Pair.ptree params in
  let double (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    Nx.cast (Nx.dtype t) (Nx.mul_s (Nx.cast Nx.float64 t) 2.0)
  in
  let _, st1 = Vega.adam_step Pair.ptree ~lr:(Vega.lr 0.1) st ~params ~grads in
  let doubled = Nx.Ptree.map opt (fun _ t -> double t) st1 in
  equal ~msg:"map reaches the counter" int32 2l (Nx.item [] doubled.step);
  let _, roundtrip =
    Nx.Ptree.map both
      (fun _ t -> t)
      (Nx.Ptree.map2 both (fun _ _ r -> r) (params, st) (params, st))
  in
  let params', _ =
    Vega.adam_step Pair.ptree ~lr:(Vega.lr 0.1) roundtrip ~params ~grads
  in
  let expected, _ =
    Vega.adam_step Pair.ptree ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  check_vec ~msg:"roundtrip state steps identically" (Nx.to_array expected.a)
    params'.a

(* A structure over packed tensors, whose leaves' dtypes differ between
   values. *)
module Any = struct
  type _ t = Nx.packed

  let walk c (Nx.P x) = Nx.P (Nx.Ptree.Walk.tensor c x)
end

(* A structure whose one leaf sits at the field ["a.b"] or at [a] then [b]: two
   paths that print alike. *)
module Split = struct
  type _ t = bool * Nx.float64_t

  let walk c (dotted, x) =
    let open Nx.Ptree.Walk in
    let x =
      if dotted then field c "a.b" tensor x
      else field c "a" (fun c -> field c "b" tensor) x
    in
    (dotted, x)
end

let test_steps_check_the_skeleton () =
  let vecs = Nx.Ptree.list Vec.ptree in
  let params = [ vec [| 1.0 |]; vec [| 2.0 |] ] in
  let st = Vega.adam_init vecs params in
  raises
    (Invalid_argument
       "Vega.adam_step: the root: length 1 in the gradients, length 2 in the \
        parameters") (fun () ->
      ignore
        (Vega.adam_step vecs ~lr:(Vega.lr 0.1) st ~params
           ~grads:[ vec [| 1.0 |] ]));
  raises
    (Invalid_argument
       "Vega.adamw_step: the root: length 1 in mu, length 2 in the parameters")
    (fun () ->
      ignore
        (Vega.adamw_step vecs ~lr:(Vega.lr 0.1)
           { st with mu = [ vec [| 0.0 |] ] }
           ~params ~grads:params));
  raises
    (Invalid_argument
       "Vega.sgd_step: the root: length 1 in the velocity, length 2 in the \
        parameters") (fun () ->
      ignore
        (Vega.sgd_step vecs ~lr:(Vega.lr 0.1) ~momentum:0.9
           { velocity = [ vec [| 0.0 |] ]; step = st.step }
           ~params ~grads:params));
  raises
    (Invalid_argument
       "Vega.sgd_step: the root: length 1 in the gradients, length 2 in the \
        parameters") (fun () ->
      ignore
        (Vega.sgd_step vecs ~lr:(Vega.lr 0.1)
           (Vega.sgd_init vecs params)
           ~params
           ~grads:[ vec [| 1.0 |] ]));
  raises
    (Invalid_argument
       "Vega.adan_step: the root: length 1 in prev_grads, length 2 in the \
        parameters") (fun () ->
      let st = Vega.adan_init vecs params in
      ignore
        (Vega.adan_step vecs ~lr:(Vega.lr 0.1)
           { st with prev_grads = [ vec [| 0.0 |] ] }
           ~params ~grads:params));
  let split : bool Split.t Nx.Ptree.t = Nx.Ptree.instantiate (module Split) in
  let params = (true, vec [| 1.0 |]) in
  raises
    (Invalid_argument
       "Vega.adam_step: [\"a\"; \"b\"]: a leaf in the gradients, a leaf at \
        [\"a.b\"] in the parameters") (fun () ->
      ignore
        (Vega.adam_step split ~lr:(Vega.lr 0.1)
           (Vega.adam_init split params)
           ~params
           ~grads:(false, vec [| 1.0 |])));
  let any : Nx.packed Nx.Ptree.t = Nx.Ptree.instantiate (module Any) in
  let params = Nx.P (vec [| 1.0 |]) in
  raises
    (Invalid_argument
       "Vega.adam_step: the root: float32 in the gradients, float64 in the \
        parameters") (fun () ->
      ignore
        (Vega.adam_step any ~lr:(Vega.lr 0.1)
           (Vega.adam_init any params)
           ~params
           ~grads:(Nx.P (Nx.create Nx.float32 [| 1 |] [| 1.0 |]))))

let tests =
  [
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
        test "sgd, adam and adamw reject bad hyperparameters"
          test_steps_validate;
      ];
    group "trajectories"
      [
        test "rmsprop follows its trajectory" test_rmsprop_trajectory;
        test "rmsprop with momentum follows its trajectory"
          test_rmsprop_momentum_trajectory;
        test "adagrad follows its trajectory" test_adagrad_trajectory;
        test "lion follows its trajectory" test_lion_trajectory;
        test "radam follows its trajectory" test_radam_trajectory;
        test "lamb follows its trajectory" test_lamb_trajectory;
        test "lars follows its trajectory" test_lars_trajectory;
        test "lars with nesterov follows its trajectory"
          test_lars_nesterov_trajectory;
        test "lars without momentum follows its trajectory"
          test_lars_no_momentum_trajectory;
        test "adan follows its trajectory" test_adan_trajectory;
        test "adafactor follows its trajectory" test_adafactor_trajectory;
        test "unfactored adafactor follows its trajectory"
          test_adafactor_unfactored_trajectory;
        test "adafactor factors the leaves of two axes"
          test_adafactor_factors_matrices;
        test "steps reject bad hyperparameters" test_ported_steps_validate;
      ];
    group "optimizer state as a structure"
      [
        test "states visit their leaves at their paths" test_state_visits;
        test "a state's structure walks it back to itself" test_state_is_a_ptree;
        test "steps check that their values share one skeleton"
          test_steps_check_the_skeleton;
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
        test "the state's structure visits its leaves and walks it back"
          test_lbfgs_state_is_a_ptree;
      ];
  ]

let () = run "vega" tests
