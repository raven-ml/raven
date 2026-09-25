(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Schedule = Schedule

(* Helpers *)

let scalar (type a b) (dt : (a, b) Nx_dtype.t) x =
  Nx.scalar dt (Nx_dtype.of_float dt x)

(* A parameter structure may carry leaves that are not parameters: an RNG key
   threaded through a compiled step, a step counter, a batch of indices. Rune
   does not differentiate them — their slot in the gradient structure holds
   zeros — and an optimizer must not update them either. Adam's square root over
   an integer leaf is meaningless, and even plain descent would round its step
   into the value. Carry them instead. *)
let updates (type a b) (p : (a, b) Nx.t) = Nx_dtype.is_float (Nx.dtype p)

let float_of_scalar (type a b) (dt : (a, b) Nx_dtype.t) (v : a) : float =
  match dt with
  | Nx_dtype.Float16 -> (v : float)
  | Nx_dtype.Float32 -> (v : float)
  | Nx_dtype.Float64 -> (v : float)
  | Nx_dtype.BFloat16 -> (v : float)
  | Nx_dtype.Float8_e4m3 -> (v : float)
  | Nx_dtype.Float8_e5m2 -> (v : float)
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

(* Optimizers over parameter structures (Nx.Ptree.t). Optimizer state is itself
   parameter-shaped, and every scalar that changes across steps — the bias
   corrections, the step counter — is a tensor leaf, so a training step is a
   pure function of (params, state) that traces under Rune.jit. *)

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
      match Nx_dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
      | Some Equal -> y
      | None ->
          invalid_argf "%s: %s: %s in %s, %s in the parameters" fn
            (describe path)
            (Nx_dtype.to_string (Nx.dtype y))
            name
            (Nx_dtype.to_string (Nx.dtype x)))

(* A step computes at float32, or at float64 for a float64 leaf, and returns
   each leaf at its own dtype. Float16 and bfloat16 leaves round once, when the
   result is stored: at their dtype, Adam's [eps = 1e-8] rounds to zero in
   float16 and [b2 = 0.999] to one in bfloat16. *)
type 'w compute = (float, 'w) Nx.t

(* The compute dtypes, as a witness a step's arithmetic is written against. *)
type _ wide = F32 : Nx.float32_elt wide | F64 : Nx.float64_elt wide

let dtype : type w. w wide -> (float, w) Nx.dtype = function
  | F32 -> Nx.float32
  | F64 -> Nx.float64

(* [leafwise fn p ~params ~grads parts ~f] steps [params] leaf by leaf. [parts]
   are the state's values of the parameters' skeleton, each with its name for
   errors. At a float leaf, [f w x g s] takes the parameter, its gradient and
   its leaf of each part, in order, cast to the compute dtype [w], and returns
   the new parameter and the parts' new leaves, which are cast back; other
   leaves pass through unchanged. The result is the new parameters and the new
   parts. *)
let leafwise fn p ~params ~grads parts
    ~(f :
       'w.
       'w wide ->
       'w compute ->
       'w compute ->
       'w compute array ->
       'w compute * 'w compute array) =
  let skeleton = snd (Nx.Ptree.flatten p params) in
  let aligned name x = ref (aligned fn p skeleton name x) in
  let grads = aligned "the gradients" grads in
  let parts = Array.of_list parts in
  let leaves = Array.map (fun (name, x) -> (name, aligned name x)) parts in
  let parts' = Array.map (fun _ -> ref []) parts in
  let step (type a b) (x : (a, b) Nx.t) g s =
    let dt = Nx.dtype x in
    let run (type w) (w : w wide) =
      let wide = dtype w in
      let x, s =
        f w (Nx.cast wide x) (Nx.cast wide g) (Array.map (Nx.cast wide) s)
      in
      (Nx.cast dt x, Array.map (Nx.cast dt) s)
    in
    match dt with Nx_dtype.Float64 -> run F64 | _ -> run F32
  in
  let update path x =
    let g = take fn "the gradients" path x grads in
    let s = Array.map (fun (name, rest) -> take fn name path x rest) leaves in
    let x, s = if updates x then step x g s else (x, s) in
    Array.iteri (fun i y -> parts'.(i) := Nx.P y :: !(parts'.(i))) s;
    x
  in
  let params = Nx.Ptree.map p update params in
  let rebuild i (_, like) = Nx.Ptree.rebuild p ~like (List.rev !(parts'.(i))) in
  (params, Array.mapi rebuild parts)

(* A scalar of the step counter, computed once per step at each compute dtype
   that asks for it. *)
type counted = { f32 : Nx.float32_t Lazy.t; f64 : Nx.float64_t Lazy.t }

let counted (f : 'w. (float, 'w) Nx.dtype -> 'w compute) =
  { f32 = lazy (f Nx.float32); f64 = lazy (f Nx.float64) }

let at : type w. counted -> w wide -> w compute =
 fun c -> function F32 -> Lazy.force c.f32 | F64 -> Lazy.force c.f64

(* [one_minus_exp x] is [1 - exp x] for [x <= 0] without the cancellation of
   subtracting from one: [-expm1 x], with [expm1] written with [exp] and [log]
   as [(u - 1) x / log u] for [u = exp x], whose rounding errors cancel. It is
   [x] where [u] rounds to one, and [u - 1] below [-1], which loses nothing and
   stays defined where [u] underflows to zero. *)
let one_minus_exp x =
  let dt = Nx.dtype x in
  let one = scalar dt 1.0 in
  let u = Nx.exp x in
  let near = Nx.div (Nx.mul (Nx.sub u one) x) (Nx.log u) in
  let expm1 = Nx.where (Nx.equal u one) x near in
  Nx.neg (Nx.where (Nx.less x (scalar dt (-1.0))) (Nx.sub u one) expm1)

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
      let of_float = Nx_dtype.of_float (Nx.dtype g) in
      Nx.clamp ~min:(of_float (-.max)) ~max:(of_float max) g)
    grads

(* The leafwise walk is a [map2] whose result is dropped, so the leaves of [a]
   are returned untouched. *)
let global_dot p (dt : (float, 'v) Nx.dtype) a b : (float, 'v) Nx.t =
  let acc = ref (Nx.scalar dt 0.0) in
  ignore
    (Nx.Ptree.map2 p
       (fun _ x y ->
         if updates x then acc := Nx.add !acc (Nx.cast dt (Nx.vdot x y));
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
  validate_unit_interval fn "momentum" momentum;
  let step = Nx.add_s st.step 1l in
  if momentum = 0.0 then
    (* Plain gradient descent: the velocity is exactly the gradient. Skipping
       the [momentum * v + g] arithmetic avoids touching (and, under [jit],
       capturing) the velocity tensors at all. *)
    let f _ x g _ = (descend ~lr x g, [||]) in
    let params, _ = leafwise fn p ~params ~grads [] ~f in
    (params, { velocity = grads; step })
  else
    let f _ x g s =
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
  let f _ x g s =
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

(* [bias_correction step b] is [1 - b^t] at the counter [t = step], as [-expm1
   (t ln b)]: [1 - pow b t] would lose the digits that matter to cancellation,
   since [b^t] is within [1e-3] of one for Adam's [b2] and small [t]. *)
let bias_correction step b =
  counted (fun dt ->
      one_minus_exp (Nx.mul (Nx.cast dt step) (scalar dt (Float.log b))))

(* One step of Adam's moments over every leaf, shared by the family: [apply x
   mu_hat nu_hat] is the new parameter from the old one and the bias-corrected
   moments. The corrections derive from the state's counter inside the step, so
   the whole step traces under jit and the state carries nothing the counter
   does not already determine. *)
let adam_update fn p ~b1 ~b2
    ~(apply :
       'w. 'w wide -> 'w compute -> 'w compute -> 'w compute -> 'w compute) st
    ~params ~grads =
  let step = Nx.add_s st.step 1l in
  let c1 = bias_correction step b1 and c2 = bias_correction step b2 in
  let f w x g s =
    let m = ema b1 s.(0) g and n = ema b2 s.(1) (Nx.mul g g) in
    (apply w x (Nx.div m (at c1 w)) (Nx.div n (at c2 w)), [| m; n |])
  in
  let params, parts =
    leafwise fn p ~params ~grads [ ("mu", st.mu); ("nu", st.nu) ] ~f
  in
  (params, { mu = parts.(0); nu = parts.(1); step })

let validate_adam fn ~b1 ~b2 ~eps =
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  validate_positive fn "eps" eps

(* Adam's direction from the bias-corrected moments. *)
let adam_direction ~eps mu_hat nu_hat =
  Nx.div mu_hat (Nx.add (Nx.sqrt nu_hat) (scalar (Nx.dtype mu_hat) eps))

let adam_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.adam_step" in
  validate_adam fn ~b1 ~b2 ~eps;
  let apply _ x mu_hat nu_hat =
    descend ~lr x (adam_direction ~eps mu_hat nu_hat)
  in
  adam_update fn p ~b1 ~b2 ~apply st ~params ~grads

let adamw_init = adam_init

let adamw_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8)
    ?(weight_decay = 0.01) st ~params ~grads =
  let fn = "Vega.adamw_step" in
  validate_adam fn ~b1 ~b2 ~eps;
  validate_non_negative fn "weight_decay" weight_decay;
  let apply _ x mu_hat nu_hat =
    let d = adam_direction ~eps mu_hat nu_hat in
    descend ~lr x (Nx.add d (Nx.mul x (scalar (Nx.dtype x) weight_decay)))
  in
  adam_update fn p ~b1 ~b2 ~apply st ~params ~grads

let radam_init = adam_init

(* The first step at which [rho] reaches 5, where RAdam starts rectifying. It
   depends on [b2] alone, so it is found on the host in float64 and the step
   compares it with the integer counter, exactly whatever the leaves' dtype.
   [rho] rises with [t] towards [rho_inf], so the search ends; it never reaches
   5 when [rho_inf] does not exceed it. *)
let radam_switch b2 =
  let rho_inf = (2.0 /. (1.0 -. b2)) -. 1.0 in
  let rho t =
    let x = t *. Float.log b2 in
    rho_inf -. (2.0 *. t *. Float.exp x /. -.Float.expm1 x)
  in
  let rec first t =
    if t >= Int32.to_int Int32.max_int then Int32.max_int
    else if rho (float t) >= 5.0 then Int32.of_int t
    else first (t + 1)
  in
  if rho_inf <= 5.0 then Int32.max_int else first 1

let radam_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.radam_step" in
  validate_adam fn ~b1 ~b2 ~eps;
  let rho_inf = (2.0 /. (1.0 -. b2)) -. 1.0 in
  let step = Nx.add_s st.step 1l in
  let rectified = Nx.greater_equal_s step (radam_switch b2) in
  let r =
    counted (fun dt ->
        let c v = scalar dt v in
        let t = Nx.cast dt step in
        let x = Nx.mul t (c (Float.log b2)) in
        let tail =
          Nx.div (Nx.mul (Nx.mul (c 2.0) t) (Nx.exp x)) (one_minus_exp x)
        in
        let rho = Nx.sub (c rho_inf) tail in
        let k = rho_inf /. ((rho_inf -. 4.0) *. (rho_inf -. 2.0)) in
        Nx.sqrt
          (Nx.div
             (Nx.mul (Nx.mul (Nx.sub rho (c 4.0)) (Nx.sub rho (c 2.0))) (c k))
             rho))
  in
  let apply w x mu_hat nu_hat =
    let rectification = Nx.mul (at r w) (adam_direction ~eps mu_hat nu_hat) in
    descend ~lr x (Nx.where rectified rectification mu_hat)
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
  let f _ x g s =
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
  let f _ x g s =
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
  let f _ x g s =
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
  let f _ x g s =
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
   leaf itself. A leaf the step does not update is never factored and holds
   zeros of its own shape in every part, as it does in every other state. *)
let adafactor_init p ?(factored = true) params =
  let factors x = factored && updates x && Nx.ndim x >= 2 in
  let unused x =
    if updates x then Nx.zeros (Nx.dtype x) [||] else Nx.zeros_like x
  in
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
  (* An average keeps [1 - t^-decay_rate] of its old value and takes
     [t^-decay_rate] of the new one, [exp e] for [e = -decay_rate log t]. *)
  let exponent dt =
    Nx.mul (Nx.log (Nx.cast dt step)) (scalar dt (-.decay_rate))
  in
  let keep = counted (fun dt -> one_minus_exp (exponent dt)) in
  let fresh = counted (fun dt -> Nx.exp (exponent dt)) in
  let f w x g s =
    let dt = Nx.dtype x in
    let eps = scalar dt eps in
    let average m v = Nx.add (Nx.mul m (at keep w)) (Nx.mul v (at fresh w)) in
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
    let clip =
      Nx.minimum (scalar dt 1.0) (Nx.div (scalar dt clipping_threshold) rms)
    in
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
