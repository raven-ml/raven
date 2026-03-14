(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Schedule = Schedule
module Dtype = Nx_core.Dtype

(* Helpers *)

let scalar (type a b) (dt : (a, b) Dtype.t) x =
  Nx.scalar dt (Dtype.of_float dt x)

let float_of_scalar (type a b) (dt : (a, b) Dtype.t) (v : a) : float =
  match dt with
  | Dtype.Float16 -> (v : float)
  | Dtype.Float32 -> (v : float)
  | Dtype.Float64 -> (v : float)
  | Dtype.BFloat16 -> (v : float)
  | Dtype.Float8_e4m3 -> (v : float)
  | Dtype.Float8_e5m2 -> (v : float)
  | _ -> invalid_arg "Vega: expected floating-point dtype"

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

let clip_by_value delta =
  validate_positive "Vega.clip_by_value" "delta" delta;
  [
    {
      n_tensors = 0;
      prim_init = (fun _ -> [||]);
      prim_update =
        (fun _count _st updates _param ->
          let dt = Nx.dtype updates in
          let min_v = Dtype.of_float dt (-.delta) in
          let max_v = Dtype.of_float dt delta in
          (Nx.clip updates ~min:min_v ~max:max_v, [||]));
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
            Nx.mul (Nx.randn dt (Nx.shape updates)) (scalar dt (sqrt variance))
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
