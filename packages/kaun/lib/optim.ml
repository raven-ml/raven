(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype

(* Errors and helpers *)

let err_expected_float_dtype = "Optim: expected floating-point dtype"
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

let float_of_scalar (type a b) (dtype : (a, b) Dtype.t) (value : a) : float =
  match dtype with
  | Dtype.Float16 ->
      let value : float = value in
      value
  | Dtype.Float32 ->
      let value : float = value in
      value
  | Dtype.Float64 ->
      let value : float = value in
      value
  | Dtype.BFloat16 ->
      let value : float = value in
      value
  | Dtype.Float8_e4m3 ->
      let value : float = value in
      value
  | Dtype.Float8_e5m2 ->
      let value : float = value in
      value
  | _ -> invalid_arg err_expected_float_dtype

let scalar dt x = Nx.scalar dt (Dtype.of_float dt x)

let tensor_sum_sq (Ptree.P t) =
  let dtype = Nx.dtype t in
  let sq = Nx.mul t t in
  float_of_scalar dtype (Nx.item [] (Nx.sum sq))

let zeros_like t = Ptree.map { run = Nx.zeros_like } t

(* Schedules *)

module Schedule = struct
  type t = int -> float

  let constant value _ = value

  let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () step =
    if step >= decay_steps then alpha *. init_value
    else
      let ratio = float_of_int step /. float_of_int decay_steps in
      let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
      (((1. -. alpha) *. cosine_val) +. alpha) *. init_value

  let warmup_cosine ~init_value ~peak_value ~warmup_steps step =
    if step >= warmup_steps then peak_value
    else
      let ratio = float_of_int step /. float_of_int warmup_steps in
      let cosine_val = 0.5 *. (1. -. Stdlib.cos (Float.pi *. ratio)) in
      init_value +. ((peak_value -. init_value) *. cosine_val)

  let exponential_decay ~init_value ~decay_rate ~decay_steps step =
    let ratio = float_of_int step /. float_of_int decay_steps in
    init_value *. (decay_rate ** ratio)

  let warmup_linear ~init_value ~peak_value ~warmup_steps step =
    if step >= warmup_steps then peak_value
    else
      let ratio = float_of_int step /. float_of_int warmup_steps in
      init_value +. ((peak_value -. init_value) *. ratio)
end

(* Optimizer core *)

type 'a spec = {
  init : Ptree.t -> 'a;
  update : step:int -> lr:float -> 'a -> Ptree.t -> Ptree.t -> Ptree.t * 'a;
  to_trees : 'a -> Ptree.t list;
  of_trees : Ptree.t list -> 'a;
}

type algorithm = A : { schedule : Schedule.t; spec : 'a spec } -> algorithm
type state = S : { count : int; data : 'a; spec : 'a spec } -> state

let init (A { spec; _ }) params = S { count = 0; data = spec.init params; spec }

let step (A { schedule; _ }) (S s) params grads =
  let count = s.count + 1 in
  let lr = schedule count in
  let updates, data = s.spec.update ~step:count ~lr s.data params grads in
  (updates, S { count; data; spec = s.spec })

let apply_updates params updates =
  Ptree.map2 { run = (fun p u -> Nx.add p u) } params updates

let update algo st params grads =
  let updates, st' = step algo st params grads in
  (apply_updates params updates, st')

(* Gradient utilities *)

let global_norm t =
  let sum_sq = Ptree.fold (fun acc p -> acc +. tensor_sum_sq p) 0. t in
  sqrt sum_sq

let err_trees_length name expected got =
  invalid_argf "Optim.state_of_trees: %s expects %d trees, got %d" name expected
    got

let state_to_trees (S s) = (s.count, s.spec.to_trees s.data)

let state_of_trees (A { spec; _ }) ~count trees =
  S { count; data = spec.of_trees trees; spec }

let clip_by_global_norm max_norm grads =
  let norm = global_norm grads in
  if norm <= max_norm then grads
  else
    let scale = max_norm /. norm in
    Ptree.map
      {
        run =
          (fun t ->
            let dt = Nx.dtype t in
            Nx.mul t (scalar dt scale));
      }
      grads

(* SGD *)

let sgd ~lr ?(momentum = 0.) ?(nesterov = false) () =
  validate_unit_interval "Optim.sgd" "momentum" momentum;
  if momentum = 0. then
    let spec =
      {
        init = (fun _ -> ());
        update =
          (fun ~step:_ ~lr () _params grads ->
            let updates =
              Ptree.map
                {
                  run =
                    (fun g ->
                      let dt = Nx.dtype g in
                      Nx.mul (scalar dt (-.lr)) g);
                }
                grads
            in
            (updates, ()));
        to_trees = (fun () -> []);
        of_trees =
          (fun l -> if l <> [] then err_trees_length "sgd" 0 (List.length l));
      }
    in
    A { schedule = lr; spec }
  else
    let spec =
      {
        init = (fun params -> zeros_like params);
        update =
          (fun ~step:_ ~lr vel _params grads ->
            let new_vel =
              Ptree.map2
                {
                  run =
                    (fun v g ->
                      let dt = Nx.dtype g in
                      Nx.add (Nx.mul v (scalar dt momentum)) g);
                }
                vel grads
            in
            let updates =
              if nesterov then
                Ptree.map2
                  {
                    run =
                      (fun g v ->
                        let dt = Nx.dtype g in
                        Nx.mul (scalar dt (-.lr))
                          (Nx.add g (Nx.mul v (scalar dt momentum))));
                  }
                  grads new_vel
              else
                Ptree.map
                  {
                    run =
                      (fun v ->
                        let dt = Nx.dtype v in
                        Nx.mul (scalar dt (-.lr)) v);
                  }
                  new_vel
            in
            (updates, new_vel));
        to_trees = (fun vel -> [ vel ]);
        of_trees =
          (function
          | [ vel ] -> vel
          | l -> err_trees_length "sgd" 1 (List.length l));
      }
    in
    A { schedule = lr; spec }

(* Adam *)

let adam ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  validate_unit_interval "Optim.adam" "b1" b1;
  validate_unit_interval "Optim.adam" "b2" b2;
  validate_positive "Optim.adam" "eps" eps;
  let spec =
    {
      init =
        (fun params ->
          let z = zeros_like params in
          (z, z));
      update =
        (fun ~step ~lr (mu, nu) _params grads ->
          let new_mu =
            Ptree.map2
              {
                run =
                  (fun m g ->
                    let dt = Nx.dtype g in
                    Nx.add
                      (Nx.mul m (scalar dt b1))
                      (Nx.mul g (scalar dt (1. -. b1))));
              }
              mu grads
          in
          let new_nu =
            Ptree.map2
              {
                run =
                  (fun v g ->
                    let dt = Nx.dtype g in
                    Nx.add
                      (Nx.mul v (scalar dt b2))
                      (Nx.mul (Nx.mul g g) (scalar dt (1. -. b2))));
              }
              nu grads
          in
          let bc1 = 1. -. (b1 ** float_of_int step) in
          let bc2 = 1. -. (b2 ** float_of_int step) in
          let updates =
            Ptree.map2
              {
                run =
                  (fun m v ->
                    let dt = Nx.dtype m in
                    let m_hat = Nx.div m (scalar dt bc1) in
                    let v_hat = Nx.div v (scalar dt bc2) in
                    Nx.mul (scalar dt (-.lr))
                      (Nx.div m_hat (Nx.add (Nx.sqrt v_hat) (scalar dt eps))));
              }
              new_mu new_nu
          in
          (updates, (new_mu, new_nu)));
      to_trees = (fun (mu, nu) -> [ mu; nu ]);
      of_trees =
        (function
        | [ mu; nu ] -> (mu, nu)
        | l -> err_trees_length "adam" 2 (List.length l));
    }
  in
  A { schedule = lr; spec }

(* AdamW *)

let adamw ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.01) ()
    =
  validate_unit_interval "Optim.adamw" "b1" b1;
  validate_unit_interval "Optim.adamw" "b2" b2;
  validate_positive "Optim.adamw" "eps" eps;
  validate_non_negative "Optim.adamw" "weight_decay" weight_decay;
  let spec =
    {
      init =
        (fun params ->
          let z = zeros_like params in
          (z, z));
      update =
        (fun ~step ~lr (mu, nu) params grads ->
          let new_mu =
            Ptree.map2
              {
                run =
                  (fun m g ->
                    let dt = Nx.dtype g in
                    Nx.add
                      (Nx.mul m (scalar dt b1))
                      (Nx.mul g (scalar dt (1. -. b1))));
              }
              mu grads
          in
          let new_nu =
            Ptree.map2
              {
                run =
                  (fun v g ->
                    let dt = Nx.dtype g in
                    Nx.add
                      (Nx.mul v (scalar dt b2))
                      (Nx.mul (Nx.mul g g) (scalar dt (1. -. b2))));
              }
              nu grads
          in
          let bc1 = 1. -. (b1 ** float_of_int step) in
          let bc2 = 1. -. (b2 ** float_of_int step) in
          let adam_updates =
            Ptree.map2
              {
                run =
                  (fun m v ->
                    let dt = Nx.dtype m in
                    let m_hat = Nx.div m (scalar dt bc1) in
                    let v_hat = Nx.div v (scalar dt bc2) in
                    Nx.mul (scalar dt (-.lr))
                      (Nx.div m_hat (Nx.add (Nx.sqrt v_hat) (scalar dt eps))));
              }
              new_mu new_nu
          in
          let decay_updates =
            Ptree.map
              {
                run =
                  (fun p ->
                    let dt = Nx.dtype p in
                    Nx.mul p (scalar dt (-.lr *. weight_decay)));
              }
              params
          in
          let updates =
            Ptree.map2
              { run = (fun adam decay -> Nx.add adam decay) }
              adam_updates decay_updates
          in
          (updates, (new_mu, new_nu)));
      to_trees = (fun (mu, nu) -> [ mu; nu ]);
      of_trees =
        (function
        | [ mu; nu ] -> (mu, nu)
        | l -> err_trees_length "adamw" 2 (List.length l));
    }
  in
  A { schedule = lr; spec }

(* RMSprop *)

let rmsprop ~lr ?(decay = 0.9) ?(eps = 1e-8) ?(momentum = 0.) () =
  validate_unit_interval "Optim.rmsprop" "decay" decay;
  validate_positive "Optim.rmsprop" "eps" eps;
  validate_unit_interval "Optim.rmsprop" "momentum" momentum;
  if momentum = 0. then
    let spec =
      {
        init = (fun params -> zeros_like params);
        update =
          (fun ~step:_ ~lr nu _params grads ->
            let new_nu =
              Ptree.map2
                {
                  run =
                    (fun v g ->
                      let dt = Nx.dtype g in
                      Nx.add
                        (Nx.mul v (scalar dt decay))
                        (Nx.mul (Nx.mul g g) (scalar dt (1. -. decay))));
                }
                nu grads
            in
            let updates =
              Ptree.map2
                {
                  run =
                    (fun g v ->
                      let dt = Nx.dtype g in
                      Nx.mul (scalar dt (-.lr))
                        (Nx.div g (Nx.add (Nx.sqrt v) (scalar dt eps))));
                }
                grads new_nu
            in
            (updates, new_nu));
        to_trees = (fun nu -> [ nu ]);
        of_trees =
          (function
          | [ nu ] -> nu
          | l -> err_trees_length "rmsprop" 1 (List.length l));
      }
    in
    A { schedule = lr; spec }
  else
    let spec =
      {
        init =
          (fun params ->
            let z = zeros_like params in
            (z, z));
        update =
          (fun ~step:_ ~lr (nu, vel) _params grads ->
            let new_nu =
              Ptree.map2
                {
                  run =
                    (fun v g ->
                      let dt = Nx.dtype g in
                      Nx.add
                        (Nx.mul v (scalar dt decay))
                        (Nx.mul (Nx.mul g g) (scalar dt (1. -. decay))));
                }
                nu grads
            in
            let scaled =
              Ptree.map2
                {
                  run =
                    (fun g v ->
                      let dt = Nx.dtype g in
                      Nx.div g (Nx.add (Nx.sqrt v) (scalar dt eps)));
                }
                grads new_nu
            in
            let new_vel =
              Ptree.map2
                {
                  run =
                    (fun v s ->
                      let dt = Nx.dtype v in
                      Nx.add (Nx.mul v (scalar dt momentum)) s);
                }
                vel scaled
            in
            let updates =
              Ptree.map
                {
                  run =
                    (fun v ->
                      let dt = Nx.dtype v in
                      Nx.mul (scalar dt (-.lr)) v);
                }
                new_vel
            in
            (updates, (new_nu, new_vel)));
        to_trees = (fun (nu, vel) -> [ nu; vel ]);
        of_trees =
          (function
          | [ nu; vel ] -> (nu, vel)
          | l -> err_trees_length "rmsprop" 2 (List.length l));
      }
    in
    A { schedule = lr; spec }

(* Adagrad *)

let adagrad ~lr ?(eps = 1e-8) () =
  validate_positive "Optim.adagrad" "eps" eps;
  let spec =
    {
      init = (fun params -> zeros_like params);
      update =
        (fun ~step:_ ~lr accum _params grads ->
          let new_accum =
            Ptree.map2
              { run = (fun acc g -> Nx.add acc (Nx.mul g g)) }
              accum grads
          in
          let updates =
            Ptree.map2
              {
                run =
                  (fun g acc ->
                    let dt = Nx.dtype g in
                    Nx.mul (scalar dt (-.lr))
                      (Nx.div g (Nx.add (Nx.sqrt acc) (scalar dt eps))));
              }
              grads new_accum
          in
          (updates, new_accum));
      to_trees = (fun accum -> [ accum ]);
      of_trees =
        (function
        | [ accum ] -> accum
        | l -> err_trees_length "adagrad" 1 (List.length l));
    }
  in
  A { schedule = lr; spec }
