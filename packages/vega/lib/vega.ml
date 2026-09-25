(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Schedule = Schedule
module Dtype = Nx_core.Dtype

(* Helpers *)

let scalar (type a b) (dt : (a, b) Dtype.t) x =
  Nx.scalar dt (Dtype.of_float dt x)

(* A parameter structure may carry leaves that are not parameters: an RNG key
   threaded through a compiled step, a step counter, a batch of indices. Rune
   does not differentiate them — their slot in the gradient structure holds
   zeros — and an optimizer must not update them either. Adam's square root over
   an integer leaf is meaningless, and even plain descent would round its step
   into the value. Carry them instead. *)
let updates (type a b) (p : (a, b) Nx.t) = Dtype.is_float (Nx.dtype p)

let float_of_scalar (type a b) (dt : (a, b) Dtype.t) (v : a) : float =
  match dt with
  | Dtype.Float16 -> (v : float)
  | Dtype.Float32 -> (v : float)
  | Dtype.Float64 -> (v : float)
  | Dtype.BFloat16 -> (v : float)
  | Dtype.Float8_e4m3 -> (v : float)
  | Dtype.Float8_e5m2 -> (v : float)
  | _ -> invalid_arg "Vega: expected floating-point dtype"

let randn (type a b) (dt : (a, b) Dtype.t) shape : (a, b) Nx.t =
  match dt with
  | Dtype.Float16 -> Nx.randn Dtype.Float16 shape
  | Dtype.Float32 -> Nx.randn Dtype.Float32 shape
  | Dtype.Float64 -> Nx.randn Dtype.Float64 shape
  | Dtype.BFloat16 -> Nx.randn Dtype.BFloat16 shape
  | Dtype.Float8_e4m3 -> Nx.randn Dtype.Float8_e4m3 shape
  | Dtype.Float8_e5m2 -> Nx.randn Dtype.Float8_e5m2 shape
  | _ -> invalid_arg "Vega.add_noise: expected floating-point dtype"

(* Validation *)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let validate_positive ctx name value =
  if value <= 0.0 then
    invalid_argf "%s: expected %s > 0.0, got %g" ctx name value

let validate_non_negative ctx name value =
  if value < 0.0 then
    invalid_argf "%s: expected %s >= 0.0, got %g" ctx name value

let validate_unit_interval ctx name value =
  if value < 0.0 || value >= 1.0 then
    invalid_argf "%s: expected 0.0 <= %s < 1.0, got %g" ctx name value

(* Primitive: a single composable gradient transformation *)

type prim = {
  n_tensors : int;
  prim_init : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t array;
  prim_update :
    'a 'b.
    int ->
    ('a, 'b) Nx.t array ->
    ('a, 'b) Nx.t ->
    ('a, 'b) Nx.t ->
    ('a, 'b) Nx.t * ('a, 'b) Nx.t array;
      (* count -> sub_state -> updates -> param -> (new_updates,
         new_sub_state) *)
}

type t = prim list

type ('a, 'b) state = {
  prims : prim array;
  count : int;
  tensors : ('a, 'b) Nx.t array;
}

(* Core *)

let chain ts = List.concat ts
let n_tensors tx = List.fold_left (fun acc p -> acc + p.n_tensors) 0 tx

let init tx param =
  let prims = Array.of_list tx in
  let tensors =
    Array.concat (Array.to_list (Array.map (fun p -> p.prim_init param) prims))
  in
  { prims; count = 0; tensors }

let update st ~grad ~param =
  let count = st.count + 1 in
  let n_prims = Array.length st.prims in
  let all_tensors = Array.copy st.tensors in
  let offset = ref 0 in
  let updates = ref grad in
  for i = 0 to n_prims - 1 do
    let p = st.prims.(i) in
    let sub_state = Array.sub st.tensors !offset p.n_tensors in
    let new_updates, new_sub_state =
      p.prim_update count sub_state !updates param
    in
    Array.blit new_sub_state 0 all_tensors !offset p.n_tensors;
    updates := new_updates;
    offset := !offset + p.n_tensors
  done;
  (!updates, { prims = st.prims; count; tensors = all_tensors })

let apply_updates ~param ~updates = Nx.add param updates

let step st ~grad ~param =
  let updates, st = update st ~grad ~param in
  (apply_updates ~param ~updates, st)

(* Scaling transforms *)

(* A chain evaluates its schedule once per parameter per update, and the count
   only moves once per step: remembering the last evaluation turns N tensor
   evaluations per step into one. *)
let memo_eval sched =
  let last = ref (-1, 0.0) in
  fun count ->
    let c, v = !last in
    if c = count then v
    else
      let v = Schedule.eval sched count in
      last := (count, v);
      v

let scale s =
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun _count _st updates _param ->
          let dt = Nx.dtype updates in
          (Nx.mul updates (scalar dt s), [||]));
    };
  ]

let scale_by_schedule sched =
  let sched = memo_eval sched in
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun count _st updates _param ->
          let dt = Nx.dtype updates in
          let s = sched count in
          (Nx.mul updates (scalar dt s), [||]));
    };
  ]

let scale_by_learning_rate lr =
  let lr = memo_eval lr in
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun count _st updates _param ->
          let dt = Nx.dtype updates in
          let s = -.lr count in
          (Nx.mul updates (scalar dt s), [||]));
    };
  ]

(* Adaptive scaling transforms *)

let scale_by_adam ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?(nesterov = false)
    ?(amsgrad = false) () =
  validate_unit_interval "Vega.scale_by_adam" "b1" b1;
  validate_unit_interval "Vega.scale_by_adam" "b2" b2;
  validate_positive "Vega.scale_by_adam" "eps" eps;
  let n_tensors = if amsgrad then 3 else 2 in
  [
    {
      n_tensors;
      prim_init =
        (fun param ->
          if amsgrad then
            [| Nx.zeros_like param; Nx.zeros_like param; Nx.zeros_like param |]
          else [| Nx.zeros_like param; Nx.zeros_like param |]);
      prim_update =
        (fun count st updates _param ->
          let mu = st.(0) and nu = st.(1) in
          let dt = Nx.dtype updates in
          let new_mu =
            Nx.add
              (Nx.mul mu (scalar dt b1))
              (Nx.mul updates (scalar dt (1. -. b1)))
          in
          let new_nu =
            Nx.add
              (Nx.mul nu (scalar dt b2))
              (Nx.mul (Nx.mul updates updates) (scalar dt (1. -. b2)))
          in
          let bc1 = 1. -. (b1 ** float_of_int count) in
          let bc2 = 1. -. (b2 ** float_of_int count) in
          let m_hat = Nx.div new_mu (scalar dt bc1) in
          let v_hat, new_st =
            if amsgrad then
              let v_max = Nx.maximum st.(2) new_nu in
              (Nx.div v_max (scalar dt bc2), [| new_mu; new_nu; v_max |])
            else (Nx.div new_nu (scalar dt bc2), [| new_mu; new_nu |])
          in
          let out =
            if nesterov then
              let m_hat_nesterov =
                Nx.add
                  (Nx.mul (scalar dt (b1 /. bc1)) new_mu)
                  (Nx.mul (scalar dt ((1. -. b1) /. bc1)) updates)
              in
              Nx.div m_hat_nesterov (Nx.add (Nx.sqrt v_hat) (scalar dt eps))
            else Nx.div m_hat (Nx.add (Nx.sqrt v_hat) (scalar dt eps))
          in
          (out, new_st));
    };
  ]

let scale_by_rms ?(decay = 0.9) ?(eps = 1e-8) () =
  validate_unit_interval "Vega.scale_by_rms" "decay" decay;
  validate_positive "Vega.scale_by_rms" "eps" eps;
  [
    {
      n_tensors = 1;
      prim_init = (fun param -> [| Nx.zeros_like param |]);
      prim_update =
        (fun _count st updates _param ->
          let nu = st.(0) in
          let dt = Nx.dtype updates in
          let new_nu =
            Nx.add
              (Nx.mul nu (scalar dt decay))
              (Nx.mul (Nx.mul updates updates) (scalar dt (1. -. decay)))
          in
          let out = Nx.div updates (Nx.add (Nx.sqrt new_nu) (scalar dt eps)) in
          (out, [| new_nu |]));
    };
  ]

let scale_by_adagrad ?(eps = 1e-8) () =
  validate_positive "Vega.scale_by_adagrad" "eps" eps;
  [
    {
      n_tensors = 1;
      prim_init = (fun param -> [| Nx.zeros_like param |]);
      prim_update =
        (fun _count st updates _param ->
          let accum = st.(0) in
          let dt = Nx.dtype updates in
          let new_accum = Nx.add accum (Nx.mul updates updates) in
          let out =
            Nx.div updates (Nx.add (Nx.sqrt new_accum) (scalar dt eps))
          in
          (out, [| new_accum |]));
    };
  ]

let scale_by_lion ?(b1 = 0.9) ?(b2 = 0.99) () =
  validate_unit_interval "Vega.scale_by_lion" "b1" b1;
  validate_unit_interval "Vega.scale_by_lion" "b2" b2;
  [
    {
      n_tensors = 1;
      prim_init = (fun param -> [| Nx.zeros_like param |]);
      prim_update =
        (fun _count st updates _param ->
          let mu = st.(0) in
          let dt = Nx.dtype updates in
          (* Update direction: sign of interpolation with b1 *)
          let interp =
            Nx.add
              (Nx.mul mu (scalar dt b1))
              (Nx.mul updates (scalar dt (1. -. b1)))
          in
          let out = Nx.sign interp in
          (* Momentum state: EMA with b2 *)
          let new_mu =
            Nx.add
              (Nx.mul mu (scalar dt b2))
              (Nx.mul updates (scalar dt (1. -. b2)))
          in
          (out, [| new_mu |]));
    };
  ]

let scale_by_radam ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  validate_unit_interval "Vega.scale_by_radam" "b1" b1;
  validate_unit_interval "Vega.scale_by_radam" "b2" b2;
  validate_positive "Vega.scale_by_radam" "eps" eps;
  let rho_inf = (2. /. (1. -. b2)) -. 1. in
  [
    {
      n_tensors = 2;
      prim_init = (fun param -> [| Nx.zeros_like param; Nx.zeros_like param |]);
      prim_update =
        (fun count st updates _param ->
          let mu = st.(0) and nu = st.(1) in
          let dt = Nx.dtype updates in
          let new_mu =
            Nx.add
              (Nx.mul mu (scalar dt b1))
              (Nx.mul updates (scalar dt (1. -. b1)))
          in
          let new_nu =
            Nx.add
              (Nx.mul nu (scalar dt b2))
              (Nx.mul (Nx.mul updates updates) (scalar dt (1. -. b2)))
          in
          let bc1 = 1. -. (b1 ** float_of_int count) in
          let m_hat = Nx.div new_mu (scalar dt bc1) in
          let b2_t = b2 ** float_of_int count in
          let rho_t =
            rho_inf -. (2. *. float_of_int count *. b2_t /. (1. -. b2_t))
          in
          let out =
            if rho_t > 5. then begin
              let bc2 = 1. -. b2_t in
              let v_hat = Nx.div new_nu (scalar dt bc2) in
              let rect =
                sqrt
                  ((rho_t -. 4.) *. (rho_t -. 2.) *. rho_inf
                  /. ((rho_inf -. 4.) *. (rho_inf -. 2.) *. rho_t))
              in
              Nx.mul (scalar dt rect)
                (Nx.div m_hat (Nx.add (Nx.sqrt v_hat) (scalar dt eps)))
            end
            else m_hat
          in
          (out, [| new_mu; new_nu |]));
    };
  ]

let scale_by_trust_ratio ?(eps = 1e-6) () =
  validate_positive "Vega.scale_by_trust_ratio" "eps" eps;
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun _count _st updates param ->
          let dt = Nx.dtype updates in
          let param_norm =
            float_of_scalar dt
              (Nx.item [] (Nx.sqrt (Nx.sum (Nx.mul param param))))
          in
          let update_norm =
            float_of_scalar dt
              (Nx.item [] (Nx.sqrt (Nx.sum (Nx.mul updates updates))))
          in
          let ratio =
            if param_norm > 0. && update_norm > 0. then
              param_norm /. (update_norm +. eps)
            else 1.
          in
          (Nx.mul updates (scalar dt ratio), [||]));
    };
  ]

let scale_by_adafactor ?(b2_decay = `Rms) ?(eps = 1e-30) ?(eps_scale = 1e-3)
    ?(factored = true) ?(clipping_threshold = 1.0) () =
  validate_positive "Vega.scale_by_adafactor" "eps" eps;
  validate_positive "Vega.scale_by_adafactor" "eps_scale" eps_scale;
  validate_positive "Vega.scale_by_adafactor" "clipping_threshold"
    clipping_threshold;
  let rms_clip (type a b) (dt : (a, b) Dtype.t) (u : (a, b) Nx.t) =
    if Float.is_finite clipping_threshold then
      let rms =
        float_of_scalar dt (Nx.item [] (Nx.sqrt (Nx.mean (Nx.mul u u))))
      in
      let scale =
        if rms > 0. then Float.min 1. (clipping_threshold /. rms) else 1.
      in
      if scale < 1. then Nx.mul u (scalar dt scale) else u
    else u
  in
  [
    {
      n_tensors = 2;
      prim_init =
        (fun param ->
          let shape = Nx.shape param in
          let ndim = Array.length shape in
          if factored && ndim >= 2 then (
            let row_shape = Array.copy shape in
            row_shape.(ndim - 1) <- 1;
            let col_shape = Array.copy shape in
            col_shape.(ndim - 2) <- 1;
            [|
              Nx.zeros (Nx.dtype param) row_shape;
              Nx.zeros (Nx.dtype param) col_shape;
            |])
          else
            [|
              Nx.zeros_like param;
              Nx.scalar (Nx.dtype param) (Dtype.of_float (Nx.dtype param) 0.);
            |]);
      prim_update =
        (fun count st updates _param ->
          let dt = Nx.dtype updates in
          let shape = Nx.shape updates in
          let ndim = Array.length shape in
          let rho =
            match b2_decay with
            | `Constant rho -> rho
            | `Rms ->
                let t = float_of_int (max count 1) in
                1. -. (t ** -0.8)
          in
          let lr = -.eps_scale /. sqrt (float_of_int (max count 1)) in
          let g_sq = Nx.mul updates updates in
          if factored && ndim >= 2 then begin
            let row_ax = ndim - 1 in
            let col_ax = ndim - 2 in
            let row_mean = Nx.mean ~axes:[ row_ax ] ~keepdims:true g_sq in
            let col_mean = Nx.mean ~axes:[ col_ax ] ~keepdims:true g_sq in
            let new_rf =
              Nx.add
                (Nx.mul st.(0) (scalar dt rho))
                (Nx.mul row_mean (scalar dt (1. -. rho)))
            in
            let new_cf =
              Nx.add
                (Nx.mul st.(1) (scalar dt rho))
                (Nx.mul col_mean (scalar dt (1. -. rho)))
            in
            let rf_mean = Nx.mean ~axes:[ col_ax ] ~keepdims:true new_rf in
            let v_est =
              Nx.div (Nx.mul new_rf new_cf) (Nx.add rf_mean (scalar dt eps))
            in
            let u = Nx.div updates (Nx.add (Nx.sqrt v_est) (scalar dt eps)) in
            let out = rms_clip dt u in
            (Nx.mul out (scalar dt lr), [| new_rf; new_cf |])
          end
          else begin
            let new_nu =
              Nx.add
                (Nx.mul st.(0) (scalar dt rho))
                (Nx.mul g_sq (scalar dt (1. -. rho)))
            in
            let u = Nx.div updates (Nx.add (Nx.sqrt new_nu) (scalar dt eps)) in
            let out = rms_clip dt u in
            (Nx.mul out (scalar dt lr), [| new_nu; st.(1) |])
          end);
    };
  ]

let scale_by_adan ?(b1 = 0.98) ?(b2 = 0.92) ?(b3 = 0.99) ?(eps = 1e-8) () =
  validate_unit_interval "Vega.scale_by_adan" "b1" b1;
  validate_unit_interval "Vega.scale_by_adan" "b2" b2;
  validate_unit_interval "Vega.scale_by_adan" "b3" b3;
  validate_positive "Vega.scale_by_adan" "eps" eps;
  [
    {
      n_tensors = 4;
      prim_init =
        (fun param ->
          [|
            Nx.zeros_like param;
            Nx.zeros_like param;
            Nx.zeros_like param;
            Nx.zeros_like param;
          |]);
      prim_update =
        (fun _count st updates _param ->
          let m = st.(0) and v = st.(1) and n = st.(2) and prev_g = st.(3) in
          let dt = Nx.dtype updates in
          let diff = Nx.sub updates prev_g in
          let new_m =
            Nx.add
              (Nx.mul m (scalar dt b1))
              (Nx.mul updates (scalar dt (1. -. b1)))
          in
          let new_v =
            Nx.add
              (Nx.mul v (scalar dt b2))
              (Nx.mul diff (scalar dt (1. -. b2)))
          in
          let nesterov_g = Nx.add updates (Nx.mul diff (scalar dt b2)) in
          let new_n =
            Nx.add
              (Nx.mul n (scalar dt b3))
              (Nx.mul (Nx.mul nesterov_g nesterov_g) (scalar dt (1. -. b3)))
          in
          let out =
            Nx.div
              (Nx.add new_m (Nx.mul new_v (scalar dt b2)))
              (Nx.add (Nx.sqrt new_n) (scalar dt eps))
          in
          (out, [| new_m; new_v; new_n; updates |]));
    };
  ]

(* Accumulation transforms *)

let trace ?(decay = 0.9) ?(nesterov = false) () =
  validate_unit_interval "Vega.trace" "decay" decay;
  [
    {
      n_tensors = 1;
      prim_init = (fun param -> [| Nx.zeros_like param |]);
      prim_update =
        (fun _count st updates _param ->
          let vel = st.(0) in
          let dt = Nx.dtype updates in
          let new_vel = Nx.add (Nx.mul vel (scalar dt decay)) updates in
          let out =
            if nesterov then Nx.add updates (Nx.mul new_vel (scalar dt decay))
            else new_vel
          in
          (out, [| new_vel |]));
    };
  ]

(* Regularization transforms *)

let add_decayed_weights ?(rate = Schedule.constant 0.01) () =
  let rate = memo_eval rate in
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun count _st updates param ->
          let dt = Nx.dtype updates in
          let r = rate count in
          (Nx.add updates (Nx.mul param (scalar dt r)), [||]));
    };
  ]

(* Clipping transforms *)

let clip delta =
  validate_positive "Vega.clip" "delta" delta;
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun _count _st updates _param ->
          let dt = Nx.dtype updates in
          let min_v = Dtype.of_float dt (-.delta) in
          let max_v = Dtype.of_float dt delta in
          (Nx.clamp updates ~min:min_v ~max:max_v, [||]));
    };
  ]

let clip_by_norm max_norm =
  validate_positive "Vega.clip_by_norm" "max_norm" max_norm;
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun _count _st updates _param ->
          let dt = Nx.dtype updates in
          let norm =
            float_of_scalar dt
              (Nx.item [] (Nx.sqrt (Nx.sum (Nx.mul updates updates))))
          in
          if norm <= max_norm then (updates, [||])
          else
            let s = max_norm /. norm in
            (Nx.mul updates (scalar dt s), [||]));
    };
  ]

(* Gradient processing *)

let centralize =
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun _count _st updates _param ->
          let ndim = Array.length (Nx.shape updates) in
          if ndim < 2 then (updates, [||])
          else
            let axes = List.init (ndim - 1) (fun i -> i + 1) in
            let mean = Nx.mean ~axes ~keepdims:true updates in
            (Nx.sub updates mean, [||]));
    };
  ]

let add_noise ~eta ?(gamma = 0.55) () =
  let eta = memo_eval eta in
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun count _st updates _param ->
          let dt = Nx.dtype updates in
          let variance =
            eta count /. Float.pow (1. +. float_of_int count) gamma
          in
          let noise =
            Nx.mul (randn dt (Nx.shape updates)) (scalar dt (sqrt variance))
          in
          (Nx.add updates noise, [||]));
    };
  ]

(* Robustness *)

let apply_if_finite tx =
  let inner_prims = Array.of_list tx in
  let inner_n =
    Array.fold_left (fun acc p -> acc + p.n_tensors) 0 inner_prims
  in
  [
    {
      n_tensors = inner_n + 1;
      prim_init =
        (fun param ->
          let inner_st =
            Array.concat
              (Array.to_list
                 (Array.map (fun p -> p.prim_init param) inner_prims))
          in
          let counter =
            Nx.scalar (Nx.dtype param) (Dtype.of_float (Nx.dtype param) 0.)
          in
          Array.append inner_st [| counter |]);
      prim_update =
        (fun count st updates param ->
          let dt = Nx.dtype updates in
          let inner_st = Array.sub st 0 inner_n in
          (* Run the inner chain *)
          let offset = ref 0 in
          let upd = ref updates in
          let new_inner = Array.copy inner_st in
          for i = 0 to Array.length inner_prims - 1 do
            let p = inner_prims.(i) in
            let sub = Array.sub inner_st !offset p.n_tensors in
            let new_upd, new_sub = p.prim_update count sub !upd param in
            Array.blit new_sub 0 new_inner !offset p.n_tensors;
            upd := new_upd;
            offset := !offset + p.n_tensors
          done;
          (* Check if result is finite *)
          let is_finite =
            let fin = Nx.isfinite !upd in
            let all_fin = Nx.all fin in
            Nx.item [] all_fin
          in
          if is_finite then
            let new_st =
              Array.append new_inner [| Nx.scalar dt (Dtype.of_float dt 0.) |]
            in
            (!upd, new_st)
          else
            let counter = st.(inner_n) in
            let new_counter =
              Nx.add counter (Nx.scalar dt (Dtype.of_float dt 1.))
            in
            let new_st = Array.append inner_st [| new_counter |] in
            (Nx.zeros_like updates, new_st));
    };
  ]

(* Optimizer aliases *)

let sgd ?(momentum = 0.) ?(nesterov = false) lr =
  validate_unit_interval "Vega.sgd" "momentum" momentum;
  if momentum > 0. then
    chain [ trace ~decay:momentum ~nesterov (); scale_by_learning_rate lr ]
  else chain [ scale_by_learning_rate lr ]

let adam ?b1 ?b2 ?eps lr =
  chain [ scale_by_adam ?b1 ?b2 ?eps (); scale_by_learning_rate lr ]

let adamw ?b1 ?b2 ?eps ?(weight_decay = 0.01) lr =
  validate_non_negative "Vega.adamw" "weight_decay" weight_decay;
  chain
    [
      scale_by_adam ?b1 ?b2 ?eps ();
      add_decayed_weights ~rate:(Schedule.constant weight_decay) ();
      scale_by_learning_rate lr;
    ]

let rmsprop ?decay ?eps ?(momentum = 0.) lr =
  validate_unit_interval "Vega.rmsprop" "momentum" momentum;
  let base = scale_by_rms ?decay ?eps () in
  if momentum > 0. then
    chain [ base; trace ~decay:momentum (); scale_by_learning_rate lr ]
  else chain [ base; scale_by_learning_rate lr ]

let adagrad ?eps lr =
  chain [ scale_by_adagrad ?eps (); scale_by_learning_rate lr ]

let lamb ?b1 ?b2 ?eps ?(weight_decay = 0.01) lr =
  chain
    [
      scale_by_adam ?b1 ?b2 ?eps ();
      add_decayed_weights ~rate:(Schedule.constant weight_decay) ();
      scale_by_trust_ratio ();
      scale_by_learning_rate lr;
    ]

let lion ?b1 ?b2 lr =
  chain [ scale_by_lion ?b1 ?b2 (); scale_by_learning_rate lr ]

let radam ?b1 ?b2 ?eps lr =
  chain [ scale_by_radam ?b1 ?b2 ?eps (); scale_by_learning_rate lr ]

let lars ?(momentum = 0.9) ?(weight_decay = 0.01) ?(nesterov = false) lr =
  chain
    [
      trace ~decay:momentum ~nesterov ();
      add_decayed_weights ~rate:(Schedule.constant weight_decay) ();
      scale_by_trust_ratio ();
      scale_by_learning_rate lr;
    ]

let adan ?b1 ?b2 ?b3 ?eps ?(weight_decay = 0.02) lr =
  validate_non_negative "Vega.adan" "weight_decay" weight_decay;
  chain
    [
      scale_by_adan ?b1 ?b2 ?b3 ?eps ();
      add_decayed_weights ~rate:(Schedule.constant weight_decay) ();
      scale_by_learning_rate lr;
    ]

let adafactor ?b2_decay () = chain [ scale_by_adafactor ?b2_decay () ]

(* Serialization *)

let state_to_tensors st = (st.count, st.tensors)

let state_of_tensors tx ~count tensors =
  let prims = Array.of_list tx in
  let expected = Array.fold_left (fun acc p -> acc + p.n_tensors) 0 prims in
  let got = Array.length tensors in
  if got <> expected then
    invalid_arg
      (Printf.sprintf "Vega.state_of_tensors: expected %d tensors, got %d"
         expected got);
  { prims; count; tensors }

(* Structural optimizers over parameter structures (Nx.Ptree.t). Optimizer state
   is itself parameter-shaped, and every scalar that changes across steps — the
   bias corrections, the step counter — is a tensor leaf, so a training step is
   a pure function of (params, state) that traces under Rune.jit. *)

(* Values that share the parameters' skeleton. A step flattens the parameters
   and every other value once, checks that each has the parameters' skeleton,
   then walks the parameters, taking one tensor of each other value per leaf. *)

let describe path =
  match Nx.Ptree.Path.segments path with
  | [] -> "the root"
  | _ -> Nx.Ptree.Path.to_string path

let aligned fn p skeleton name x =
  let leaves, k = Nx.Ptree.flatten p x in
  match
    Nx.Ptree.Skeleton.diff ~this:("in " ^ name) k ~that:"in the parameters"
      skeleton
  with
  | None -> leaves
  | Some m -> invalid_arg (fn ^ ": " ^ m)

let take (type a b) fn name path (x : (a, b) Nx.t) rest : (a, b) Nx.t =
  match !rest with
  | [] -> invalid_arg (fn ^ ": the structure's walk visited one value two ways")
  | Nx.P y :: tail -> (
      rest := tail;
      match Dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
      | Some Equal -> y
      | None ->
          invalid_argf "%s: %s: %s in %s, %s in the parameters" fn
            (describe path)
            (Dtype.to_string (Nx.dtype y))
            name
            (Dtype.to_string (Nx.dtype x)))

(* [leafwise fn p ~params ~grads parts ~f] steps [params] leaf by leaf. [parts]
   are the state's values of the parameters' skeleton, each with its name for
   errors. At a float leaf, [f x g s] takes the parameter, its gradient and its
   leaf of each part, in order, and returns the new parameter and the parts' new
   leaves; other leaves pass through unchanged. The result is the new parameters
   and the new parts. *)
let leafwise fn p ~params ~grads parts
    ~(f :
       'a 'b.
       ('a, 'b) Nx.t ->
       ('a, 'b) Nx.t ->
       ('a, 'b) Nx.t array ->
       ('a, 'b) Nx.t * ('a, 'b) Nx.t array) =
  let skeleton = snd (Nx.Ptree.flatten p params) in
  let aligned name x = ref (aligned fn p skeleton name x) in
  let grads = aligned "the gradients" grads in
  let parts = Array.of_list parts in
  let leaves = Array.map (fun (name, x) -> (name, aligned name x)) parts in
  let parts' = Array.map (fun _ -> ref []) parts in
  let update path x =
    let g = take fn "the gradients" path x grads in
    let s = Array.map (fun (name, rest) -> take fn name path x rest) leaves in
    let x, s = if updates x then f x g s else (x, s) in
    Array.iteri (fun i y -> parts'.(i) := Nx.P y :: !(parts'.(i))) s;
    x
  in
  let params = Nx.Ptree.map p update params in
  let rebuild i (_, like) = Nx.Ptree.rebuild p ~like (List.rev !(parts'.(i))) in
  (params, Array.mapi rebuild parts)

(* Gradient transformations *)

let global_norm p grads =
  let sum =
    Nx.Ptree.fold p
      (fun _ g acc ->
        acc +. Nx.item [] (Nx.sum (Nx.square (Nx.cast Nx.float64 g))))
      grads 0.0
  in
  Stdlib.sqrt sum

let clip_by_global_norm p ~max_norm grads =
  validate_positive "Vega.clip_by_global_norm" "max_norm" max_norm;
  (* The norm and the scale factor stay in tensor arithmetic — no [Nx.item] — so
     the transform traces under jit. The accumulation is float32: every device
     computes it, unlike [global_norm]'s float64 host read. *)
  let sq =
    Nx.Ptree.fold p
      (fun _ g acc -> Nx.add acc (Nx.sum (Nx.square (Nx.cast Nx.float32 g))))
      grads (Nx.scalar Nx.float32 0.0)
  in
  let norm = Nx.sqrt sq in
  let factor =
    Nx.where
      (Nx.greater_s norm max_norm)
      (Nx.rdiv_s max_norm norm) (Nx.scalar Nx.float32 1.0)
  in
  Nx.Ptree.map p (fun _ g -> Nx.mul g (Nx.cast (Nx.dtype g) factor)) grads

let clip_by_value p ~max grads =
  validate_positive "Vega.clip_by_value" "max" max;
  Nx.Ptree.map p
    (fun _ g ->
      let of_float = Dtype.of_float (Nx.dtype g) in
      Nx.clamp ~min:(of_float (-.max)) ~max:(of_float max) g)
    grads

(* The leafwise walk is a [map2] whose result is dropped, so the leaves of [a]
   are returned untouched. *)
let global_dot p (dt : (float, 'v) Nx.dtype) a b : (float, 'v) Nx.t =
  let acc = ref (Nx.scalar dt 0.0) in
  ignore
    (Nx.Ptree.map2 p
       (fun _ x y ->
         if updates x then acc := Nx.add !acc (Nx.cast dt (Nx.sum (Nx.mul x y)));
         x)
       a b);
  !acc

(* Loss scaling *)

module Loss_scale = struct
  type t = { scale : Nx.float32_t; good_steps : Nx.int32_t }

  (* Static scales are marked by [good_steps = -1]: the mark is itself a tensor,
     so [adjust] can pass them through with [Nx.where] arithmetic instead of
     control flow — under [jit] the state is an ordinary input, not a trace-time
     constant. *)

  let static v =
    validate_positive "Vega.Loss_scale.static" "scale" v;
    { scale = Nx.scalar Nx.float32 v; good_steps = Nx.scalar Nx.int32 (-1l) }

  let dynamic ?(init = 32768.0) () =
    validate_positive "Vega.Loss_scale.dynamic" "init" init;
    { scale = Nx.scalar Nx.float32 init; good_steps = Nx.scalar Nx.int32 0l }

  module Walked = struct
    type nonrec _ t = t

    let walk c { scale; good_steps } =
      let open Nx.Ptree.Walk in
      let scale = field c "scale" tensor scale in
      let good_steps = field c "good_steps" tensor good_steps in
      { scale; good_steps }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
  let scale t x = Nx.mul x (Nx.cast (Nx.dtype x) t.scale)

  let unscale p t grads =
    Nx.Ptree.map p (fun _ g -> Nx.div g (Nx.cast (Nx.dtype g) t.scale)) grads

  let grads_finite p grads =
    Nx.Ptree.fold p
      (fun _ g acc -> Nx.logical_and acc (Nx.all (Nx.isfinite g)))
      grads (Nx.scalar Nx.bool true)

  let adjust ?(growth_interval = 2000) ?(growth_factor = 2.0)
      ?(backoff_factor = 0.5) t ~finite =
    if growth_interval <= 0 then
      invalid_argf
        "Vega.Loss_scale.adjust: expected growth_interval > 0, got %d"
        growth_interval;
    validate_positive "Vega.Loss_scale.adjust" "growth_factor" growth_factor;
    validate_positive "Vega.Loss_scale.adjust" "backoff_factor" backoff_factor;
    let dynamic = Nx.greater_equal_s t.good_steps 0l in
    let good = Nx.add_s t.good_steps 1l in
    let grow = Nx.greater_equal_s good (Int32.of_int growth_interval) in
    let scale_fin = Nx.where grow (Nx.mul_s t.scale growth_factor) t.scale in
    let good_fin = Nx.where grow (Nx.zeros_like good) good in
    let scale' = Nx.where finite scale_fin (Nx.mul_s t.scale backoff_factor) in
    let good' = Nx.where finite good_fin (Nx.zeros_like good) in
    {
      scale = Nx.where dynamic scale' t.scale;
      good_steps = Nx.where dynamic good' t.good_steps;
    }
end

(* Learning rates *)

let lr v = Nx.scalar Nx.float32 v

(* Shared arithmetic of the steps *)

(* [zeros p x] is [x] with every tensor zeroed. *)
let zeros p x = Nx.Ptree.map p (fun _ t -> Nx.zeros_like t) x

(* [descend ~lr x d] is [x - lr * d], the rate cast to [x]'s dtype. *)
let descend ~lr x d = Nx.sub x (Nx.mul d (Nx.cast (Nx.dtype x) lr))

(* [ema b m x] is [b * m + (1 - b) * x]: the moving average [m] after [x]. *)
let ema b m x =
  let dt = Nx.dtype x in
  Nx.add (Nx.mul m (scalar dt b)) (Nx.mul x (scalar dt (1.0 -. b)))

(* The layer-wise trust ratio of LARS and LAMB: [|x| / (|u| + 1e-6)] for the
   leaf [x] and its update [u], or [1] when either norm is zero, as a scalar at
   [x]'s dtype. *)
let trust_ratio x u =
  let dt = Nx.dtype x in
  let norm t = Nx.sqrt (Nx.sum (Nx.mul t t)) in
  let xn = norm x and un = norm u in
  let zero = scalar dt 0.0 in
  Nx.where
    (Nx.logical_and (Nx.greater xn zero) (Nx.greater un zero))
    (Nx.div xn (Nx.add un (scalar dt 1e-6)))
    (scalar dt 1.0)

(* SGD and LARS *)

type 'p sgd_state = { velocity : 'p; step : Nx.int32_t }

module Sgd_state = struct
  type 'p t = 'p sgd_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let velocity = field c "velocity" leaf st.velocity in
    let step = field c "step" tensor st.step in
    { velocity; step }
end

let sgd_ptree p = Nx.Ptree.nest (module Sgd_state) p

let sgd_init p params =
  { velocity = zeros p params; step = Nx.scalar Nx.int32 0l }

let sgd_step p ~lr ?(momentum = 0.0) st ~params ~grads =
  let fn = "Vega.sgd_step" in
  let step = Nx.add_s st.step 1l in
  if momentum = 0.0 then
    (* Plain gradient descent: the velocity is exactly the gradient. Skipping
       the [momentum * v + g] arithmetic avoids touching (and, under [jit],
       capturing) the velocity tensors at all. *)
    let f x g _ = (descend ~lr x g, [||]) in
    let params, _ = leafwise fn p ~params ~grads [] ~f in
    (params, { velocity = grads; step })
  else
    let f x g s =
      let v = Nx.add (Nx.mul s.(0) (scalar (Nx.dtype x) momentum)) g in
      (descend ~lr x v, [| v |])
    in
    let params, parts =
      leafwise fn p ~params ~grads [ ("the velocity", st.velocity) ] ~f
    in
    (params, { velocity = parts.(0); step })

let lars_init = sgd_init

let lars_step p ~lr ?(momentum = 0.9) ?(weight_decay = 0.01) ?(nesterov = false)
    st ~params ~grads =
  let fn = "Vega.lars_step" in
  validate_unit_interval fn "momentum" momentum;
  validate_non_negative fn "weight_decay" weight_decay;
  let f x g s =
    let dt = Nx.dtype x in
    let u = Nx.add g (Nx.mul x (scalar dt weight_decay)) in
    let u = Nx.mul u (trust_ratio x u) in
    let v = Nx.add (Nx.mul s.(0) (scalar dt momentum)) u in
    let d = if nesterov then Nx.add u (Nx.mul v (scalar dt momentum)) else v in
    (descend ~lr x d, [| v |])
  in
  let params, parts =
    leafwise fn p ~params ~grads [ ("the velocity", st.velocity) ] ~f
  in
  (params, { velocity = parts.(0); step = Nx.add_s st.step 1l })

(* The Adam family: Adam, AdamW, RAdam and LAMB *)

type 'p adam_state = { mu : 'p; nu : 'p; step : Nx.int32_t }

module Adam_state = struct
  type 'p t = 'p adam_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf st.mu in
    let nu = field c "nu" leaf st.nu in
    let step = field c "step" tensor st.step in
    { mu; nu; step }
end

let adam_ptree p = Nx.Ptree.nest (module Adam_state) p

let adam_init p params =
  { mu = zeros p params; nu = zeros p params; step = Nx.scalar Nx.int32 0l }

(* One step of Adam's moments over every leaf, shared by the family: [apply t x
   mu_hat nu_hat] is the new parameter from the old one and the bias-corrected
   moments at step [t]. [t] and the bias corrections [1 - b^t] are derived from
   the counter per leaf, at the leaf's dtype like every other scalar in the step
   — tensor arithmetic with a constant base, which compiles to [exp2] on every
   device — so the whole step traces under jit and the state carries nothing the
   counter does not already determine. *)
let adam_update fn p ~b1 ~b2
    ~(apply :
       'a 'b.
       ('a, 'b) Nx.t ->
       ('a, 'b) Nx.t ->
       ('a, 'b) Nx.t ->
       ('a, 'b) Nx.t ->
       ('a, 'b) Nx.t) st ~params ~grads =
  let step = Nx.add_s st.step 1l in
  let f x g s =
    let dt = Nx.dtype x in
    let m = ema b1 s.(0) g and n = ema b2 s.(1) (Nx.mul g g) in
    let t = Nx.cast dt step in
    let c1 = Nx.sub (scalar dt 1.0) (Nx.pow (scalar dt b1) t) in
    let c2 = Nx.sub (scalar dt 1.0) (Nx.pow (scalar dt b2) t) in
    (apply t x (Nx.div m c1) (Nx.div n c2), [| m; n |])
  in
  let params, parts =
    leafwise fn p ~params ~grads [ ("mu", st.mu); ("nu", st.nu) ] ~f
  in
  (params, { mu = parts.(0); nu = parts.(1); step })

(* Adam's direction from the bias-corrected moments. *)
let adam_direction ~eps mu_hat nu_hat =
  Nx.div mu_hat (Nx.add (Nx.sqrt nu_hat) (scalar (Nx.dtype mu_hat) eps))

let adam_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) st ~params ~grads =
  let apply _ x mu_hat nu_hat =
    descend ~lr x (adam_direction ~eps mu_hat nu_hat)
  in
  adam_update "Vega.adam_step" p ~b1 ~b2 ~apply st ~params ~grads

let adamw_init = adam_init

let adamw_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8)
    ?(weight_decay = 0.01) st ~params ~grads =
  let apply _ x mu_hat nu_hat =
    let d = adam_direction ~eps mu_hat nu_hat in
    descend ~lr x (Nx.add d (Nx.mul x (scalar (Nx.dtype x) weight_decay)))
  in
  adam_update "Vega.adamw_step" p ~b1 ~b2 ~apply st ~params ~grads

let validate_adam fn ~b1 ~b2 ~eps =
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  validate_positive fn "eps" eps

let radam_init = adam_init

(* Like the bias corrections, every scalar of the rectification derives from
   [b2] at the leaf's dtype, [rho_inf] included, so that [rho], the small
   difference of two terms near [rho_inf], is taken between terms of one
   [b2]. *)
let radam_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.radam_step" in
  validate_adam fn ~b1 ~b2 ~eps;
  let apply t x mu_hat nu_hat =
    let dt = Nx.dtype x in
    let c v = scalar dt v in
    let b2 = c b2 in
    let rho_inf = Nx.sub (Nx.div (c 2.0) (Nx.sub (c 1.0) b2)) (c 1.0) in
    let b2t = Nx.pow b2 t in
    let rho =
      Nx.sub rho_inf
        (Nx.div (Nx.mul (Nx.mul (c 2.0) t) b2t) (Nx.sub (c 1.0) b2t))
    in
    let r =
      Nx.sqrt
        (Nx.div
           (Nx.mul (Nx.mul (Nx.sub rho (c 4.0)) (Nx.sub rho (c 2.0))) rho_inf)
           (Nx.mul
              (Nx.mul (Nx.sub rho_inf (c 4.0)) (Nx.sub rho_inf (c 2.0)))
              rho))
    in
    let d =
      Nx.where
        (Nx.greater rho (c 5.0))
        (Nx.mul r (adam_direction ~eps mu_hat nu_hat))
        mu_hat
    in
    descend ~lr x d
  in
  adam_update fn p ~b1 ~b2 ~apply st ~params ~grads

let lamb_init = adam_init

let lamb_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8)
    ?(weight_decay = 0.01) st ~params ~grads =
  let fn = "Vega.lamb_step" in
  validate_adam fn ~b1 ~b2 ~eps;
  validate_non_negative fn "weight_decay" weight_decay;
  let apply _ x mu_hat nu_hat =
    let d = adam_direction ~eps mu_hat nu_hat in
    let u = Nx.add d (Nx.mul x (scalar (Nx.dtype x) weight_decay)) in
    descend ~lr x (Nx.mul u (trust_ratio x u))
  in
  adam_update fn p ~b1 ~b2 ~apply st ~params ~grads

(* RMSprop *)

type 'p rmsprop_state = { nu : 'p; velocity : 'p; step : Nx.int32_t }

module Rmsprop_state = struct
  type 'p t = 'p rmsprop_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let nu = field c "nu" leaf st.nu in
    let velocity = field c "velocity" leaf st.velocity in
    let step = field c "step" tensor st.step in
    { nu; velocity; step }
end

let rmsprop_ptree p = Nx.Ptree.nest (module Rmsprop_state) p

let rmsprop_init p params =
  {
    nu = zeros p params;
    velocity = zeros p params;
    step = Nx.scalar Nx.int32 0l;
  }

let rmsprop_step p ~lr ?(decay = 0.9) ?(eps = 1e-8) ?(momentum = 0.0) st ~params
    ~grads =
  let fn = "Vega.rmsprop_step" in
  validate_unit_interval fn "decay" decay;
  validate_positive fn "eps" eps;
  validate_unit_interval fn "momentum" momentum;
  let f x g s =
    let dt = Nx.dtype x in
    let nu = ema decay s.(0) (Nx.mul g g) in
    let u = Nx.div g (Nx.add (Nx.sqrt nu) (scalar dt eps)) in
    let v =
      if momentum = 0.0 then u else Nx.add (Nx.mul s.(1) (scalar dt momentum)) u
    in
    (descend ~lr x v, [| nu; v |])
  in
  let parts = [ ("nu", st.nu); ("the velocity", st.velocity) ] in
  let params, parts = leafwise fn p ~params ~grads parts ~f in
  (params, { nu = parts.(0); velocity = parts.(1); step = Nx.add_s st.step 1l })

(* Adagrad *)

type 'p adagrad_state = { sum_of_squares : 'p; step : Nx.int32_t }

module Adagrad_state = struct
  type 'p t = 'p adagrad_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let sum_of_squares = field c "sum_of_squares" leaf st.sum_of_squares in
    let step = field c "step" tensor st.step in
    { sum_of_squares; step }
end

let adagrad_ptree p = Nx.Ptree.nest (module Adagrad_state) p

let adagrad_init p params =
  { sum_of_squares = zeros p params; step = Nx.scalar Nx.int32 0l }

let adagrad_step p ~lr ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.adagrad_step" in
  validate_positive fn "eps" eps;
  let f x g s =
    let s = Nx.add s.(0) (Nx.mul g g) in
    let d = Nx.div g (Nx.add (Nx.sqrt s) (scalar (Nx.dtype x) eps)) in
    (descend ~lr x d, [| s |])
  in
  let parts = [ ("sum_of_squares", st.sum_of_squares) ] in
  let params, parts = leafwise fn p ~params ~grads parts ~f in
  (params, { sum_of_squares = parts.(0); step = Nx.add_s st.step 1l })

(* Adan *)

type 'p adan_state = {
  mu : 'p;
  delta : 'p;
  nu : 'p;
  prev_grads : 'p;
  step : Nx.int32_t;
}

module Adan_state = struct
  type 'p t = 'p adan_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf st.mu in
    let delta = field c "delta" leaf st.delta in
    let nu = field c "nu" leaf st.nu in
    let prev_grads = field c "prev_grads" leaf st.prev_grads in
    let step = field c "step" tensor st.step in
    { mu; delta; nu; prev_grads; step }
end

let adan_ptree p = Nx.Ptree.nest (module Adan_state) p

let adan_init p params =
  {
    mu = zeros p params;
    delta = zeros p params;
    nu = zeros p params;
    prev_grads = zeros p params;
    step = Nx.scalar Nx.int32 0l;
  }

let adan_step p ~lr ?(b1 = 0.98) ?(b2 = 0.92) ?(b3 = 0.99) ?(eps = 1e-8)
    ?(weight_decay = 0.02) st ~params ~grads =
  let fn = "Vega.adan_step" in
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  validate_unit_interval fn "b3" b3;
  validate_positive fn "eps" eps;
  validate_non_negative fn "weight_decay" weight_decay;
  let f x g s =
    let dt = Nx.dtype x in
    let dg = Nx.sub g s.(3) in
    let mu = ema b1 s.(0) g and delta = ema b2 s.(1) dg in
    let ahead = Nx.add g (Nx.mul dg (scalar dt b2)) in
    let nu = ema b3 s.(2) (Nx.mul ahead ahead) in
    let d =
      Nx.div
        (Nx.add mu (Nx.mul delta (scalar dt b2)))
        (Nx.add (Nx.sqrt nu) (scalar dt eps))
    in
    let d = Nx.add d (Nx.mul x (scalar dt weight_decay)) in
    (descend ~lr x d, [| mu; delta; nu; g |])
  in
  let parts =
    [
      ("mu", st.mu);
      ("delta", st.delta);
      ("nu", st.nu);
      ("prev_grads", st.prev_grads);
    ]
  in
  let params, parts = leafwise fn p ~params ~grads parts ~f in
  let step = Nx.add_s st.step 1l in
  ( params,
    {
      mu = parts.(0);
      delta = parts.(1);
      nu = parts.(2);
      prev_grads = parts.(3);
      step;
    } )

(* Lion *)

type 'p lion_state = { mu : 'p; step : Nx.int32_t }

module Lion_state = struct
  type 'p t = 'p lion_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf st.mu in
    let step = field c "step" tensor st.step in
    { mu; step }
end

let lion_ptree p = Nx.Ptree.nest (module Lion_state) p
let lion_init p params = { mu = zeros p params; step = Nx.scalar Nx.int32 0l }

let lion_step p ~lr ?(b1 = 0.9) ?(b2 = 0.99) st ~params ~grads =
  let fn = "Vega.lion_step" in
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  let f x g s =
    (descend ~lr x (Nx.sign (ema b1 s.(0) g)), [| ema b2 s.(0) g |])
  in
  let params, parts = leafwise fn p ~params ~grads [ ("mu", st.mu) ] ~f in
  (params, { mu = parts.(0); step = Nx.add_s st.step 1l })

(* Adafactor *)

type 'p adafactor_state = {
  nu_row : 'p;
  nu_col : 'p;
  nu : 'p;
  step : Nx.int32_t;
}

module Adafactor_state = struct
  type 'p t = 'p adafactor_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let nu_row = field c "nu_row" leaf st.nu_row in
    let nu_col = field c "nu_col" leaf st.nu_col in
    let nu = field c "nu" leaf st.nu in
    let step = field c "step" tensor st.step in
    { nu_row; nu_col; nu; step }
end

let adafactor_ptree p = Nx.Ptree.nest (module Adafactor_state) p

(* A factored leaf keeps its statistics in [nu_row] and [nu_col], any other in
   [nu], and the parts a leaf does not use hold a scalar zero. So the state
   records which leaves are factored: those whose [nu] has fewer axes than the
   leaf itself. *)
let adafactor_init p ?(factored = true) params =
  let factors x = factored && Nx.ndim x >= 2 in
  let unused x = Nx.zeros (Nx.dtype x) [||] in
  let factor axis x =
    if factors x then (
      let shape = Array.copy (Nx.shape x) in
      shape.(Array.length shape - axis) <- 1;
      Nx.zeros (Nx.dtype x) shape)
    else unused x
  in
  {
    nu_row = Nx.Ptree.map p (fun _ x -> factor 1 x) params;
    nu_col = Nx.Ptree.map p (fun _ x -> factor 2 x) params;
    nu =
      Nx.Ptree.map p
        (fun _ x -> if factors x then unused x else Nx.zeros_like x)
        params;
    step = Nx.scalar Nx.int32 0l;
  }

let adafactor_step p ~lr ?(decay_rate = 0.8) ?(eps = 1e-30)
    ?(clipping_threshold = 1.0) st ~params ~grads =
  let fn = "Vega.adafactor_step" in
  validate_positive fn "decay_rate" decay_rate;
  validate_positive fn "eps" eps;
  validate_positive fn "clipping_threshold" clipping_threshold;
  let step = Nx.add_s st.step 1l in
  let f x g s =
    let dt = Nx.dtype x in
    let one = scalar dt 1.0 and eps = scalar dt eps in
    (* [t^-decay_rate] as [exp (-decay_rate * log t)]: compiled code lowers
       [pow] only for a constant base or an integer or half-integer exponent. *)
    let decay = Nx.mul (Nx.log (Nx.cast dt step)) (scalar dt (-.decay_rate)) in
    let b = Nx.sub one (Nx.exp decay) in
    let average m v = Nx.add (Nx.mul m b) (Nx.mul v (Nx.sub one b)) in
    let g2 = Nx.mul g g in
    let n = Nx.ndim x in
    let u, s =
      if Nx.ndim s.(2) < n then
        let row = average s.(0) (Nx.mean ~axes:[ n - 1 ] ~keepdims:true g2) in
        let col = average s.(1) (Nx.mean ~axes:[ n - 2 ] ~keepdims:true g2) in
        let row_mean = Nx.mean ~axes:[ n - 2 ] ~keepdims:true row in
        let nu = Nx.div (Nx.mul row col) (Nx.add row_mean eps) in
        (Nx.div g (Nx.add (Nx.sqrt nu) eps), [| row; col; s.(2) |])
      else
        let nu = average s.(2) g2 in
        (Nx.div g (Nx.add (Nx.sqrt nu) eps), [| s.(0); s.(1); nu |])
    in
    let rms = Nx.sqrt (Nx.mean (Nx.mul u u)) in
    let clip = Nx.minimum one (Nx.div (scalar dt clipping_threshold) rms) in
    (descend ~lr x (Nx.mul u clip), s)
  in
  let parts = [ ("nu_row", st.nu_row); ("nu_col", st.nu_col); ("nu", st.nu) ] in
  let params, parts = leafwise fn p ~params ~grads parts ~f in
  (params, { nu_row = parts.(0); nu_col = parts.(1); nu = parts.(2); step })

(* L-BFGS *)

type ('p, 'v) lbfgs_state = {
  params : 'p;
  value : (float, 'v) Nx.t;
  grads : 'p;
  s : 'p;
  y : 'p;
  rho : (float, 'v) Nx.t;
  step : Nx.int32_t;
}

let lbfgs_ptree (type v) p : (_, v) lbfgs_state Nx.Ptree.t =
  let module Walked = struct
    type 'p t = ('p, v) lbfgs_state

    let walk c st =
      let open Nx.Ptree.Walk in
      let params = field c "params" leaf st.params in
      let value = field c "value" tensor st.value in
      let grads = field c "grads" leaf st.grads in
      let s = field c "s" leaf st.s in
      let y = field c "y" leaf st.y in
      let rho = field c "rho" tensor st.rho in
      let step = field c "step" tensor st.step in
      { params; value; grads; s; y; rho; step }
  end in
  Nx.Ptree.nest (module Walked) p

let lbfgs_init p ?(history = 10) f params =
  if history < 1 then
    invalid_argf "Vega.lbfgs_init: expected history >= 1, got %d" history;
  let value, grads = f params in
  let memory () =
    Nx.Ptree.map p
      (fun _ leaf ->
        Nx.zeros (Nx.dtype leaf) (Array.append [| history |] (Nx.shape leaf)))
      params
  in
  {
    params;
    value;
    grads;
    s = memory ();
    y = memory ();
    rho = Nx.zeros (Nx.dtype value) [| history |];
    step = Nx.scalar Nx.int32 0l;
  }

(* A leaf's memory is its pairs stacked along axis 0, newest first: [slot i]
   views the [i]-th, [push] puts a new one on top and drops the oldest. *)
let slot i memory = Nx.get [ i ] memory

let push x memory =
  let n = (Nx.shape memory).(0) in
  let x = Nx.unsqueeze ~axes:[ 0 ] x in
  if n = 1 then x
  else Nx.concatenate ~axis:0 [ x; Nx.slice [ Nx.R (0, n - 1) ] memory ]

(* [move params d a] is [params + a * d] on the float leaves, [a] a scalar
   tensor cast to each leaf's dtype; [axpy a x y] is [y + a * x] likewise. *)
let move p params d a =
  Nx.Ptree.map2 p
    (fun _ x d ->
      if updates x then Nx.add x (Nx.mul d (Nx.cast (Nx.dtype x) a)) else x)
    params d

let axpy p a x y =
  Nx.Ptree.map2 p
    (fun _ x y ->
      if updates x then Nx.add y (Nx.mul x (Nx.cast (Nx.dtype x) a)) else y)
    x y

(* The two-loop recursion (Nocedal, 1980): [-H g] for the inverse Hessian the
   stored pairs define, scaled initially by [(s . y) / (y . y)] of the newest
   pair. Pairs of weight [0] — empty slots, rejected curvature — contribute
   nothing to either loop, so no fill count is kept. Every scalar is a tensor at
   the objective's dtype and every index is static, so the direction traces
   under jit. *)
let lbfgs_direction p st =
  let dt = Nx.dtype st.value in
  let m = (Nx.shape st.rho).(0) in
  let dot = global_dot p dt in
  let pair i =
    ( Nx.Ptree.map p (fun _ m -> slot i m) st.s,
      Nx.Ptree.map p (fun _ m -> slot i m) st.y,
      Nx.get [ i ] st.rho )
  in
  let alphas = Array.make m (Nx.scalar dt 0.0) in
  let q = ref st.grads in
  for i = 0 to m - 1 do
    let s, y, rho = pair i in
    let alpha = Nx.mul rho (dot s !q) in
    alphas.(i) <- alpha;
    q := axpy p (Nx.neg alpha) y !q
  done;
  let y0 = Nx.Ptree.map p (fun _ m -> slot 0 m) st.y
  and rho0 = Nx.get [ 0 ] st.rho in
  let gamma =
    Nx.where (Nx.greater_s rho0 0.0)
      (Nx.div (Nx.scalar dt 1.0) (Nx.mul rho0 (dot y0 y0)))
      (Nx.scalar dt 1.0)
  in
  let r =
    ref
      (Nx.Ptree.map p
         (fun _ q ->
           if updates q then Nx.mul q (Nx.cast (Nx.dtype q) gamma) else q)
         !q)
  in
  for i = m - 1 downto 0 do
    let s, y, rho = pair i in
    let beta = Nx.mul rho (dot y !r) in
    r := axpy p (Nx.sub alphas.(i) beta) s !r
  done;
  Nx.Ptree.map p (fun _ r -> if updates r then Nx.neg r else r) !r

(* A point of the line search: [params + alpha * d], the objective and gradient
   there, and [phi alpha], [phi' alpha] read to the host. *)
type ('p, 'v) trial = {
  point : 'p;
  objective : (float, 'v) Nx.t;
  gradient : 'p;
  alpha : float;
  phi : float;
  dphi : float;
}

(* A strong-Wolfe line search along [d] from [st] (Nocedal and Wright, 2006,
   algorithms 3.5 and 3.6): bracket from a unit step, doubling until the
   sufficient-decrease condition fails or the slope turns, then zoom into the
   bracket by safeguarded quadratic interpolation. Returns the accepted trial;
   when [budget] evaluations are spent, the lowest trial that decreased the
   value, or [None] if none did. A [nan] objective fails every acceptance test,
   so a trial that overflowed only shrinks the bracket. *)
let line_search p ~budget f st d =
  let dt = Nx.dtype st.value in
  let dot = global_dot p dt in
  let c1 = 1e-4 and c2 = 0.9 in
  let origin =
    {
      point = st.params;
      objective = st.value;
      gradient = st.grads;
      alpha = 0.0;
      phi = Nx.item [] st.value;
      dphi = Nx.item [] (dot st.grads d);
    }
  in
  let armijo t = t.phi <= origin.phi +. (c1 *. t.alpha *. origin.dphi) in
  let curvature t = Float.abs t.dphi <= -.c2 *. origin.dphi in
  let probe alpha =
    let point = move p st.params d (Nx.scalar dt alpha) in
    let objective, gradient = f point in
    let phi = Nx.item [] objective and dphi = Nx.item [] (dot gradient d) in
    { point; objective; gradient; alpha; phi; dphi }
  in
  let lower best t =
    match best with
    | Some b when not (t.phi < b.phi) -> best
    | _ -> if t.phi < origin.phi then Some t else best
  in
  let rec zoom lo hi best budget =
    if budget = 0 then best
    else
      let alpha =
        (* The minimizer of the quadratic through [phi lo], [phi' lo] and [phi
           hi], kept to the middle 80% of the bracket; bisection when the
           interpolation lands outside it or is undefined. *)
        let w = hi.alpha -. lo.alpha in
        let denom = 2.0 *. (hi.phi -. lo.phi -. (lo.dphi *. w)) in
        let a = lo.alpha -. (lo.dphi *. w *. w /. denom) in
        let near = lo.alpha +. (0.1 *. w) and far = hi.alpha -. (0.1 *. w) in
        let inside =
          if w > 0.0 then near <= a && a <= far else far <= a && a <= near
        in
        if inside then a else lo.alpha +. (0.5 *. w)
      in
      let t = probe alpha in
      let best = lower best t in
      if (not (armijo t)) || t.phi >= lo.phi then zoom lo t best (budget - 1)
      else if curvature t then Some t
      else if t.dphi *. (hi.alpha -. lo.alpha) >= 0.0 then
        zoom t lo best (budget - 1)
      else zoom t hi best (budget - 1)
  in
  let rec bracket prev alpha best budget =
    if budget = 0 then best
    else
      let t = probe alpha in
      let best = lower best t in
      if (not (armijo t)) || t.phi >= prev.phi then zoom prev t best (budget - 1)
      else if curvature t then Some t
      else if t.dphi >= 0.0 then zoom t prev best (budget - 1)
      else bracket t (2.0 *. alpha) best (budget - 1)
  in
  if origin.dphi >= 0.0 then None else bracket origin 1.0 None budget

let lbfgs_step p ?lr ?(max_linesearch_steps = 20) f st =
  if max_linesearch_steps < 1 then
    invalid_argf "Vega.lbfgs_step: expected max_linesearch_steps >= 1, got %d"
      max_linesearch_steps;
  let dt = Nx.dtype st.value in
  let d = lbfgs_direction p st in
  let advance point objective gradient =
    let difference =
      Nx.Ptree.map2 p (fun _ a b -> if updates a then Nx.sub a b else a)
    in
    let s = difference point st.params and y = difference gradient st.grads in
    let ys = global_dot p dt y s in
    let rho =
      Nx.where (Nx.greater_s ys 0.0) (Nx.rdiv_s 1.0 ys) (Nx.scalar dt 0.0)
    in
    {
      params = point;
      value = objective;
      grads = gradient;
      s = Nx.Ptree.map2 p (fun _ x m -> push x m) s st.s;
      y = Nx.Ptree.map2 p (fun _ x m -> push x m) y st.y;
      rho = push rho st.rho;
      step = Nx.add_s st.step 1l;
    }
  in
  match lr with
  | Some lr ->
      let point = move p st.params d lr in
      let objective, gradient = f point in
      advance point objective gradient
  | None -> (
      match line_search p ~budget:max_linesearch_steps f st d with
      | Some t -> advance t.point t.objective t.gradient
      | None -> st)

type status = Converged | Max_iter_reached | Line_search_failed

let minimize p ?history ?(max_iter = 1000) ?(gtol = 1e-5) ?(ftol = 1e-9)
    ?max_linesearch_steps f params =
  if max_iter < 0 then
    invalid_argf "Vega.minimize: expected max_iter >= 0, got %d" max_iter;
  validate_non_negative "Vega.minimize" "gtol" gtol;
  validate_non_negative "Vega.minimize" "ftol" ftol;
  let grad_max st =
    Nx.Ptree.fold p
      (fun _ g acc ->
        if updates g then
          let dt = Nx.dtype g in
          Float.max acc (float_of_scalar dt (Nx.item [] (Nx.max (Nx.abs g))))
        else acc)
      st.grads 0.0
  in
  let rec loop st k =
    if grad_max st <= gtol then (st, Converged)
    else if k = max_iter then (st, Max_iter_reached)
    else
      let st' = lbfgs_step p ?max_linesearch_steps f st in
      if Nx.item [] st'.step = Nx.item [] st.step then (st, Line_search_failed)
      else
        let before = Nx.item [] st.value and after = Nx.item [] st'.value in
        let scale =
          Float.max 1.0 (Float.max (Float.abs before) (Float.abs after))
        in
        if before -. after <= ftol *. scale then (st', Converged)
        else loop st' (k + 1)
  in
  loop (lbfgs_init p ?history f params) 0
