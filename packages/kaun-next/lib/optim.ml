(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Optimizer state is itself parameter-shaped: updates are pure map2
   compositions, fully typed, with no positional pairing. *)

(* Schedules *)

type schedule = int -> float

let constant lr _ = lr

let exponential_decay ~init ~rate ~steps =
  if steps <= 0 then invalid_arg "Optim.exponential_decay: steps <= 0";
  fun k -> init *. (rate ** (float_of_int k /. float_of_int steps))

let cosine_decay ?(final = 0.) ~init ~steps () =
  if steps <= 0 then invalid_arg "Optim.cosine_decay: steps <= 0";
  fun k ->
    let t = Float.min 1.0 (float_of_int k /. float_of_int steps) in
    final +. ((init -. final) *. 0.5 *. (1.0 +. Stdlib.cos (Float.pi *. t)))

let warmup_cosine ?(final = 0.) ~peak ~warmup ~steps () =
  if warmup < 0 then invalid_arg "Optim.warmup_cosine: warmup < 0";
  if steps <= warmup then invalid_arg "Optim.warmup_cosine: steps <= warmup";
  let decay = cosine_decay ~final ~init:peak ~steps:(steps - warmup) () in
  fun k ->
    if k < warmup then peak *. float_of_int k /. float_of_int warmup
    else decay (k - warmup)

(* Gradient transformations *)

(* A scalar constant in [leaf]'s dtype, broadcastable against it. *)
let scalar leaf v =
  Nx.full (Nx.dtype leaf) [||] (Nx_core.Dtype.of_float (Nx.dtype leaf) v)

let global_norm (module P : Nx.Ptree.S) (grads : P.t) : float =
  let acc = ref 0.0 in
  P.iter
    (fun g ->
      acc := !acc +. Nx.item [] (Nx.sum (Nx.square (Nx.cast Nx.float64 g))))
    grads;
  Stdlib.sqrt !acc

let clip_by_global_norm (module P : Nx.Ptree.S) ~max_norm (grads : P.t) : P.t =
  if not (max_norm > 0.) then
    invalid_arg "Optim.clip_by_global_norm: max_norm <= 0";
  let norm = global_norm (module P) grads in
  if norm <= max_norm then grads
  else
    let c = max_norm /. norm in
    P.map (fun g -> Nx.mul g (scalar g c)) grads

let clip_by_value (module P : Nx.Ptree.S) ~max (grads : P.t) : P.t =
  if not (max > 0.) then invalid_arg "Optim.clip_by_value: max <= 0";
  P.map
    (fun g ->
      let of_float = Nx_core.Dtype.of_float (Nx.dtype g) in
      Nx.clip ~min:(of_float (-.max)) ~max:(of_float max) g)
    grads

(* SGD *)

type 'p sgd_state = { velocity : 'p }

let sgd_init (module P : Nx.Ptree.S) (params : P.t) : P.t sgd_state =
  { velocity = P.map (fun leaf -> Nx.zeros_like leaf) params }

let sgd_step (module P : Nx.Ptree.S) ~lr ?(momentum = 0.0) (st : P.t sgd_state)
    ~(params : P.t) ~(grads : P.t) : P.t * P.t sgd_state =
  let velocity =
    P.map2
      (fun v g -> Nx.add (Nx.mul v (scalar v momentum)) g)
      st.velocity grads
  in
  let params =
    P.map2 (fun p v -> Nx.sub p (Nx.mul v (scalar v lr))) params velocity
  in
  (params, { velocity })

(* Adam and AdamW *)

type 'p adam_state = { mu : 'p; nu : 'p; step : int }

let adam_init (module P : Nx.Ptree.S) (params : P.t) : P.t adam_state =
  let zeros () = P.map (fun leaf -> Nx.zeros_like leaf) params in
  { mu = zeros (); nu = zeros (); step = 0 }

(* Advances the moments and computes the bias-corrected update direction shared
   by [adam_step] and [adamw_step]. *)
let adam_direction (module P : Nx.Ptree.S) ~b1 ~b2 ~eps (st : P.t adam_state)
    ~(grads : P.t) : P.t * P.t adam_state =
  let step = st.step + 1 in
  let mu =
    P.map2
      (fun m g ->
        Nx.add (Nx.mul m (scalar m b1)) (Nx.mul g (scalar g (1.0 -. b1))))
      st.mu grads
  in
  let nu =
    P.map2
      (fun n g ->
        Nx.add
          (Nx.mul n (scalar n b2))
          (Nx.mul (Nx.mul g g) (scalar g (1.0 -. b2))))
      st.nu grads
  in
  let c1 = 1.0 -. (b1 ** float_of_int step) in
  let c2 = 1.0 -. (b2 ** float_of_int step) in
  let direction =
    P.map2
      (fun m n ->
        let mu_hat = Nx.div m (scalar m c1) in
        let nu_hat = Nx.div n (scalar n c2) in
        Nx.div mu_hat (Nx.add (Nx.sqrt nu_hat) (scalar nu_hat eps)))
      mu nu
  in
  (direction, { mu; nu; step })

let adam_step (module P : Nx.Ptree.S) ~lr ?(b1 = 0.9) ?(b2 = 0.999)
    ?(eps = 1e-8) (st : P.t adam_state) ~(params : P.t) ~(grads : P.t) :
    P.t * P.t adam_state =
  let direction, st = adam_direction (module P) ~b1 ~b2 ~eps st ~grads in
  let params =
    P.map2 (fun p d -> Nx.sub p (Nx.mul d (scalar p lr))) params direction
  in
  (params, st)

let adamw_init (module P : Nx.Ptree.S) (params : P.t) : P.t adam_state =
  adam_init (module P) params

let adamw_step (module P : Nx.Ptree.S) ~lr ?(b1 = 0.9) ?(b2 = 0.999)
    ?(eps = 1e-8) ?(weight_decay = 0.01) (st : P.t adam_state) ~(params : P.t)
    ~(grads : P.t) : P.t * P.t adam_state =
  let direction, st = adam_direction (module P) ~b1 ~b2 ~eps st ~grads in
  let params =
    P.map2
      (fun p d ->
        let decayed = Nx.add d (Nx.mul p (scalar p weight_decay)) in
        Nx.sub p (Nx.mul decayed (scalar p lr)))
      params direction
  in
  (params, st)
