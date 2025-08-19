type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

type ('layout, 'dev) params =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of (string * ('layout, 'dev) params) list

type ('layout, 'dev) opt_state = State : 'a -> ('layout, 'dev) opt_state

type ('layout, 'dev) gradient_transformation = {
  init : ('layout, 'dev) params -> ('layout, 'dev) opt_state;
  update :
    ('layout, 'dev) opt_state ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params * ('layout, 'dev) opt_state;
}

(* Utility functions *)
let rec map_params : type a b.
    (a, b) params -> ((a, b) tensor -> (a, b) tensor) -> (a, b) params =
 fun params f ->
  match params with
  | Tensor t -> Tensor (f t)
  | List ps -> List (List.map (fun p -> map_params p f) ps)
  | Record fields ->
      Record (List.map (fun (name, p) -> (name, map_params p f)) fields)

let rec map_params2 : type a b.
    (a, b) params ->
    (a, b) params ->
    ((a, b) tensor -> (a, b) tensor -> (a, b) tensor) ->
    (a, b) params =
 fun p1 p2 f ->
  match (p1, p2) with
  | Tensor t1, Tensor t2 -> Tensor (f t1 t2)
  | List ps1, List ps2 ->
      List (List.map2 (fun p1 p2 -> map_params2 p1 p2 f) ps1 ps2)
  | Record fields1, Record fields2 ->
      let sorted1 =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) fields1
      in
      let sorted2 =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) fields2
      in
      Record
        (List.map2
           (fun (k1, p1) (k2, p2) ->
             assert (k1 = k2);
             (k1, map_params2 p1 p2 f))
           sorted1 sorted2)
  | _ -> failwith "map_params2: parameter structure mismatch"

let rec flatten_params : type a b. (a, b) params -> (a, b) tensor list =
 fun params ->
  match params with
  | Tensor t -> [ t ]
  | List ps -> List.concat_map flatten_params ps
  | Record fields -> List.concat_map (fun (_, p) -> flatten_params p) fields

(* Helper functions for parameter tree flattening - removed as not currently
   used *)

(* Core transformations *)

let identity () =
  {
    init = (fun _ -> State ());
    update = (fun state _params grads -> (grads, state));
  }

let scale factor =
  {
    init = (fun _ -> State ());
    update =
      (fun state _params grads ->
        let updates =
          map_params grads (fun g ->
              Rune.(mul g (scalar (device g) (dtype g) factor)))
        in
        (updates, state));
  }

let scale_by_neg_one () = scale (-1.)

let add_decayed_weights weight_decay =
  {
    init = (fun _ -> State ());
    update =
      (fun state params grads ->
        let updates =
          map_params2 grads params (fun g p ->
              let dev = Rune.device g in
              let dt = Rune.dtype g in
              Rune.(add g (mul p (scalar dev dt weight_decay))))
        in
        (updates, state));
  }

let clip_by_global_norm max_norm =
  {
    init = (fun _ -> State ());
    update =
      (fun state _params grads ->
        let flat_grads = flatten_params grads in
        let norm_sq =
          List.fold_left
            (fun acc g ->
              let g_sq = Rune.sum (Rune.mul g g) in
              Rune.add acc g_sq)
            (let g0 = List.hd flat_grads in
             Rune.scalar (Rune.device g0) (Rune.dtype g0) 0.)
            flat_grads
        in
        let norm = Rune.sqrt norm_sq in
        let scale_factor =
          let dev = Rune.device norm in
          let dt = Rune.dtype norm in
          Rune.minimum norm (Rune.scalar dev dt max_norm)
        in
        let scaled_grads =
          map_params grads (fun g -> Rune.div (Rune.mul g scale_factor) norm)
        in
        (scaled_grads, state));
  }

let clip max_delta =
  {
    init = (fun _ -> State ());
    update =
      (fun state _params grads ->
        let clipped =
          map_params grads (fun g ->
              let dev = Rune.device g in
              let dt = Rune.dtype g in
              Rune.maximum
                (Rune.minimum g (Rune.scalar dev dt max_delta))
                (Rune.scalar dev dt (-.max_delta)))
        in
        (clipped, state));
  }

(* Momentum and adaptive methods *)

let trace ~decay ?(nesterov = false) () =
  {
    init =
      (fun params -> State (map_params params (fun t -> Rune.zeros_like t)));
    update =
      (fun state _params grads ->
        match state with
        | State momentum ->
            (* Type annotation to help OCaml understand momentum is params *)
            let momentum : ('a, 'b) params = Obj.magic momentum in
            let new_momentum =
              map_params2 momentum grads (fun m g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(add (mul m (scalar dev dt decay)) g))
            in
            let updates =
              if nesterov then
                map_params2 grads new_momentum (fun g m ->
                    let dev = Rune.device g in
                    let dt = Rune.dtype g in
                    Rune.(add g (mul m (scalar dev dt decay))))
              else new_momentum
            in
            (updates, State new_momentum));
  }

let scale_by_rms ?(decay = 0.999) ?(eps = 1e-8) () =
  {
    init =
      (fun params -> State (map_params params (fun t -> Rune.zeros_like t)));
    update =
      (fun state _params grads ->
        match state with
        | State nu ->
            let nu : ('a, 'b) params = Obj.magic nu in
            let new_nu =
              map_params2 nu grads (fun v g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul v (scalar dev dt decay))
                      (mul (mul g g) (scalar dev dt (1. -. decay)))))
            in
            let updates =
              map_params2 grads new_nu (fun g v ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(div g (add (sqrt v) (scalar dev dt eps))))
            in
            (updates, State new_nu));
  }

let scale_by_adam ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  {
    init =
      (fun params ->
        let zeros = map_params params (fun t -> Rune.zeros_like t) in
        State (zeros, zeros, 0));
    update =
      (fun state _params grads ->
        match state with
        | State s ->
            let mu, nu, count = Obj.magic s in
            let count = count + 1 in
            let new_mu =
              map_params2 mu grads (fun m g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul m (scalar dev dt b1))
                      (mul g (scalar dev dt (1. -. b1)))))
            in
            let new_nu =
              map_params2 nu grads (fun v g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul v (scalar dev dt b2))
                      (mul (mul g g) (scalar dev dt (1. -. b2)))))
            in
            (* Bias correction *)
            let bc1 = 1. -. (b1 ** float_of_int count) in
            let bc2 = 1. -. (b2 ** float_of_int count) in
            let updates =
              map_params2 new_mu new_nu (fun m v ->
                  let dev = Rune.device m in
                  let dt = Rune.dtype m in
                  let m_hat = Rune.div m (Rune.scalar dev dt bc1) in
                  let v_hat = Rune.div v (Rune.scalar dev dt bc2) in
                  Rune.(div m_hat (add (sqrt v_hat) (scalar dev dt eps))))
            in
            (updates, State (new_mu, new_nu, count)));
  }

let scale_by_belief ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-16) () =
  {
    init =
      (fun params ->
        let zeros = map_params params (fun t -> Rune.zeros_like t) in
        State (zeros, zeros, 0));
    update =
      (fun state _params grads ->
        match state with
        | State st ->
            let mu, s, count = Obj.magic st in
            let count = count + 1 in
            let new_mu =
              map_params2 mu grads (fun m g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul m (scalar dev dt b1))
                      (mul g (scalar dev dt (1. -. b1)))))
            in
            (* AdaBelief: track variance of gradients from predictions *)
            let new_s =
              map_params2 s (map_params2 grads new_mu Rune.sub)
                (fun s_old diff ->
                  let dev = Rune.device s_old in
                  let dt = Rune.dtype s_old in
                  Rune.(
                    add
                      (mul s_old (scalar dev dt b2))
                      (mul (mul diff diff) (scalar dev dt (1. -. b2)))))
            in
            (* Bias correction *)
            let bc1 = 1. -. (b1 ** float_of_int count) in
            let bc2 = 1. -. (b2 ** float_of_int count) in
            let updates =
              map_params2 new_mu new_s (fun m s ->
                  let dev = Rune.device m in
                  let dt = Rune.dtype m in
                  let m_hat = Rune.div m (Rune.scalar dev dt bc1) in
                  let s_hat = Rune.div s (Rune.scalar dev dt bc2) in
                  Rune.(
                    div m_hat
                      (add
                         (sqrt (add s_hat (scalar dev dt eps)))
                         (scalar dev dt eps))))
            in
            (updates, State (new_mu, new_s, count)));
  }

(* Learning rate schedules *)
module Schedule = struct
  type t = int -> float

  let constant value _step = value

  let exponential_decay ~init_value ~decay_rate ~decay_steps step =
    let steps_ratio = float_of_int step /. float_of_int decay_steps in
    init_value *. (decay_rate ** steps_ratio)

  let polynomial_decay ~init_value ~end_value ~power ~decay_steps step =
    if step >= decay_steps then end_value
    else
      let decay_factor =
        (1. -. (float_of_int step /. float_of_int decay_steps)) ** power
      in
      ((init_value -. end_value) *. decay_factor) +. end_value

  let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () step =
    if step >= decay_steps then alpha *. init_value
    else
      let ratio = float_of_int step /. float_of_int decay_steps in
      let cosine_val = 0.5 *. (1. +. cos (Float.pi *. ratio)) in
      (((1. -. alpha) *. cosine_val) +. alpha) *. init_value

  let piecewise_constant ~boundaries step =
    let rec find_value = function
      | [] -> failwith "piecewise_constant: no value for step"
      | [ (_, v) ] -> v
      | (bound, v) :: rest -> if step < bound then v else find_value rest
    in
    find_value boundaries

  let warmup_linear ~init_value ~peak_value ~warmup_steps step =
    if step >= warmup_steps then peak_value
    else
      let ratio = float_of_int step /. float_of_int warmup_steps in
      init_value +. ((peak_value -. init_value) *. ratio)

  let warmup_cosine ~init_value ~peak_value ~warmup_steps step =
    if step >= warmup_steps then peak_value
    else
      let ratio = float_of_int step /. float_of_int warmup_steps in
      let cosine_val = 0.5 *. (1. -. cos (Float.pi *. ratio)) in
      init_value +. ((peak_value -. init_value) *. cosine_val)

  let join schedules ~boundaries step =
    let rec find_schedule idx = function
      | [] -> List.nth schedules idx
      | bound :: rest ->
          if step < bound then List.nth schedules idx
          else find_schedule (idx + 1) rest
    in
    let schedule = find_schedule 0 boundaries in
    schedule step
end

let scale_by_schedule schedule =
  {
    init = (fun _ -> State 0);
    update =
      (fun state _params grads ->
        match state with
        | State step ->
            let step : int = Obj.magic step in
            let step = step + 1 in
            let lr = schedule step in
            let updates =
              map_params grads (fun g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.mul g (Rune.scalar dev dt lr))
            in
            (updates, State step));
  }

(* Composition *)

let chain transforms =
  {
    init =
      (fun params ->
        let states = List.map (fun t -> t.init params) transforms in
        State states);
    update =
      (fun state params grads ->
        match state with
        | State states ->
            let states : ('a, 'b) opt_state list = Obj.magic states in
            let updates, new_states =
              List.fold_left2
                (fun (updates, states_acc) transform state ->
                  let new_updates, new_state =
                    transform.update state params updates
                  in
                  (new_updates, new_state :: states_acc))
                (grads, []) transforms states
            in
            (updates, State (List.rev new_states)));
  }

(* Type for representing parameter tree structure with integer labels *)
type label_tree =
  | LabelTensor of int
  | LabelList of label_tree list
  | LabelRecord of (string * label_tree) list

(* Type for representing parameter tree structure with boolean masks *)
type mask_tree =
  | MaskTensor of bool
  | MaskList of mask_tree list
  | MaskRecord of (string * mask_tree) list

let multi_transform ~transforms ~labels =
  {
    init =
      (fun params ->
        (* Initialize all transforms with the full params *)
        let states = Array.map (fun t -> t.init params) transforms in
        State states);
    update =
      (fun state params grads ->
        match state with
        | State states ->
            let states : ('a, 'b) opt_state array = Obj.magic states in
            (* Get labels for each parameter *)
            let label_tree = labels params in

            (* Apply each transform separately to the gradients filtered by
               label *)
            let all_updates =
              Array.mapi
                (fun idx transform ->
                  (* Apply transform only to parameters with matching label *)
                  let rec filter_by_label : type a b.
                      int ->
                      label_tree ->
                      (a, b) params ->
                      (a, b) params ->
                      (a, b) params =
                   fun target_idx labels params grads ->
                    match (labels, params, grads) with
                    | LabelTensor label_idx, Tensor _p, Tensor g ->
                        if label_idx = target_idx then Tensor g
                        else Tensor (Rune.zeros_like g)
                    | LabelList ls, List ps, List gs ->
                        List
                          (List.map
                             (fun ((l, p), g) ->
                               filter_by_label target_idx l p g)
                             (List.combine (List.combine ls ps) gs))
                    | LabelRecord fields_l, Record fields_p, Record fields_g ->
                        let sorted_l =
                          List.sort
                            (fun (k1, _) (k2, _) -> String.compare k1 k2)
                            fields_l
                        in
                        let sorted_p =
                          List.sort
                            (fun (k1, _) (k2, _) -> String.compare k1 k2)
                            fields_p
                        in
                        let sorted_g =
                          List.sort
                            (fun (k1, _) (k2, _) -> String.compare k1 k2)
                            fields_g
                        in
                        Record
                          (List.map2
                             (fun (k1, l) (k2, p) ->
                               let k3, g =
                                 List.find (fun (k, _) -> k = k1) sorted_g
                               in
                               assert (k1 = k2 && k2 = k3);
                               (k1, filter_by_label target_idx l p g))
                             sorted_l sorted_p)
                    | _ -> failwith "multi_transform: structure mismatch"
                  in

                  let filtered_grads =
                    filter_by_label idx label_tree params grads
                  in
                  let updates, new_state =
                    transform.update states.(idx) params filtered_grads
                  in
                  (updates, new_state))
                transforms
            in

            (* Sum all updates *)
            let combined_updates =
              Array.fold_left
                (fun acc (updates, _) -> map_params2 acc updates Rune.add)
                (map_params grads (fun t -> Rune.zeros_like t))
                all_updates
            in

            (* Extract new states *)
            let new_states = Array.map snd all_updates in

            (combined_updates, State new_states));
  }

let rec apply_mask : type a b.
    mask_tree -> (a, b) params -> (a, b) params -> (a, b) params =
 fun mask_tree params grads ->
  match (mask_tree, params, grads) with
  | MaskTensor true, Tensor _, Tensor g -> Tensor g
  | MaskTensor false, Tensor p, Tensor _ ->
      (* Return zero gradient for masked parameters *)
      Tensor (Rune.zeros_like p)
  | MaskList masks, List ps, List gs ->
      List
        (List.map
           (fun ((m, p), g) -> apply_mask m p g)
           (List.combine (List.combine masks ps) gs))
  | MaskRecord mask_fields, Record param_fields, Record grad_fields ->
      let sorted_m =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) mask_fields
      in
      let sorted_p =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) param_fields
      in
      let sorted_g =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) grad_fields
      in
      Record
        (List.map2
           (fun (k1, m) (k2, p) ->
             let k3, g = List.find (fun (k, _) -> k = k1) sorted_g in
             assert (k1 = k2 && k2 = k3);
             (k1, apply_mask m p g))
           sorted_m sorted_p)
  | _ -> failwith "apply_mask: structure mismatch"

let masked ~mask ~inner =
  {
    init = (fun params -> inner.init params);
    update =
      (fun state params grads ->
        let mask_tree = mask params in
        let masked_grads = apply_mask mask_tree params grads in
        inner.update state params masked_grads);
  }

(* Pre-configured optimizers *)

let sgd ~lr ?(momentum = 0.) ?(nesterov = false) () =
  if momentum > 0. then chain [ trace ~decay:momentum ~nesterov (); scale lr ]
  else scale lr

let adam ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  chain [ scale_by_adam ~b1 ~b2 ~eps (); scale lr ]

let adamw ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.01) ()
    =
  chain
    [
      scale_by_adam ~b1 ~b2 ~eps (); add_decayed_weights weight_decay; scale lr;
    ]

let rmsprop ~lr ?(decay = 0.9) ?(eps = 1e-8) ?(momentum = 0.) () =
  if momentum > 0. then
    chain [ scale_by_rms ~decay ~eps (); trace ~decay:momentum (); scale lr ]
  else chain [ scale_by_rms ~decay ~eps (); scale lr ]

let adagrad ~lr ?(eps = 1e-8) () =
  {
    init =
      (fun params -> State (map_params params (fun t -> Rune.zeros_like t)));
    update =
      (fun state _params grads ->
        match state with
        | State accum ->
            let accum : ('a, 'b) params = Obj.magic accum in
            let new_accum =
              map_params2 accum grads (fun acc g -> Rune.add acc (Rune.mul g g))
            in
            let updates =
              map_params2 grads new_accum (fun g acc ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    mul (scalar dev dt lr)
                      (div g (add (sqrt acc) (scalar dev dt eps)))))
            in
            (updates, State new_accum));
  }

let adabelief ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-16) () =
  chain [ scale_by_belief ~b1 ~b2 ~eps (); scale lr ]

let lamb ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-6) ?(weight_decay = 0.01) () =
  {
    init =
      (fun params ->
        let zeros = map_params params (fun t -> Rune.zeros_like t) in
        State (zeros, zeros, 0));
    update =
      (fun state params grads ->
        match state with
        | State s ->
            let mu, nu, count = Obj.magic s in
            let count = count + 1 in
            (* Adam-style moments *)
            let new_mu =
              map_params2 mu grads (fun m g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul m (scalar dev dt b1))
                      (mul g (scalar dev dt (1. -. b1)))))
            in
            let new_nu =
              map_params2 nu grads (fun v g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul v (scalar dev dt b2))
                      (mul (mul g g) (scalar dev dt (1. -. b2)))))
            in
            (* Bias correction *)
            let bc1 = 1. -. (b1 ** float_of_int count) in
            let bc2 = 1. -. (b2 ** float_of_int count) in

            let updates =
              map_params2
                (map_params2 new_mu new_nu (fun m v ->
                     let dev = Rune.device m in
                     let dt = Rune.dtype m in
                     let m_hat = Rune.div m (Rune.scalar dev dt bc1) in
                     let v_hat = Rune.div v (Rune.scalar dev dt bc2) in
                     Rune.(
                       add
                         (div m_hat (add (sqrt v_hat) (scalar dev dt eps)))
                         (mul (scalar dev dt weight_decay) m))))
                params
                (fun update p ->
                  (* Layer adaptation *)
                  let update_norm =
                    Rune.sqrt (Rune.sum (Rune.mul update update))
                  in
                  let param_norm = Rune.sqrt (Rune.sum (Rune.mul p p)) in
                  let trust_ratio =
                    let dev = Rune.device update_norm in
                    let dt = Rune.dtype update_norm in
                    Rune.(
                      where
                        (greater param_norm (scalar dev dt 0.))
                        (div param_norm update_norm)
                        (scalar dev dt 1.))
                  in
                  let dev = Rune.device update in
                  let dt = Rune.dtype update in
                  Rune.(mul (mul (scalar dev dt lr) trust_ratio) update))
            in
            (updates, State (new_mu, new_nu, count)));
  }

let radam ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  {
    init =
      (fun params ->
        let zeros = map_params params (fun t -> Rune.zeros_like t) in
        State (zeros, zeros, 0));
    update =
      (fun state _params grads ->
        match state with
        | State s ->
            let mu, nu, count = Obj.magic s in
            let count = count + 1 in
            let new_mu =
              map_params2 mu grads (fun m g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul m (scalar dev dt b1))
                      (mul g (scalar dev dt (1. -. b1)))))
            in
            let new_nu =
              map_params2 nu grads (fun v g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul v (scalar dev dt b2))
                      (mul (mul g g) (scalar dev dt (1. -. b2)))))
            in
            (* Rectified Adam - compute length of approximated SMA *)
            let rho_inf = (2. /. (1. -. b2)) -. 1. in
            let rho_t =
              rho_inf
              -. 2. *. float_of_int count
                 *. (b2 ** float_of_int count)
                 /. (1. -. (b2 ** float_of_int count))
            in

            let updates =
              if rho_t > 4. then
                (* Variance is tractable - use adaptive learning rate *)
                let bc1 = 1. -. (b1 ** float_of_int count) in
                let bc2 = 1. -. (b2 ** float_of_int count) in
                let rect_term =
                  sqrt
                    ((rho_t -. 4.) *. (rho_t -. 2.) *. rho_inf
                    /. ((rho_inf -. 4.) *. (rho_inf -. 2.) *. rho_t))
                in
                map_params2 new_mu new_nu (fun m v ->
                    let dev = Rune.device m in
                    let dt = Rune.dtype m in
                    let m_hat = Rune.div m (Rune.scalar dev dt bc1) in
                    let v_hat = Rune.div v (Rune.scalar dev dt bc2) in
                    Rune.(
                      mul
                        (scalar dev dt (lr *. rect_term))
                        (div m_hat (add (sqrt v_hat) (scalar dev dt eps)))))
              else
                (* Variance is not tractable - use simple moving average *)
                let bc1 = 1. -. (b1 ** float_of_int count) in
                map_params new_mu (fun m ->
                    let dev = Rune.device m in
                    let dt = Rune.dtype m in
                    Rune.(mul (scalar dev dt lr) (div m (scalar dev dt bc1))))
            in
            (updates, State (new_mu, new_nu, count)));
  }

let yogi ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-3) () =
  {
    init =
      (fun params ->
        let zeros = map_params params (fun t -> Rune.zeros_like t) in
        State (zeros, zeros, 0));
    update =
      (fun state _params grads ->
        match state with
        | State s ->
            let mu, nu, count = Obj.magic s in
            let count = count + 1 in
            let new_mu =
              map_params2 mu grads (fun m g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.(
                    add
                      (mul m (scalar dev dt b1))
                      (mul g (scalar dev dt (1. -. b1)))))
            in
            (* Yogi - additive increase, multiplicative decrease *)
            let new_nu =
              map_params2 nu grads (fun v g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  let g_sq = Rune.mul g g in
                  let sign_v_gsq = Rune.sign (Rune.sub v g_sq) in
                  Rune.(
                    sub v (mul (scalar dev dt (1. -. b2)) (mul g_sq sign_v_gsq))))
            in
            (* Bias correction *)
            let bc1 = 1. -. (b1 ** float_of_int count) in
            let updates =
              map_params2 new_mu new_nu (fun m v ->
                  let dev = Rune.device m in
                  let dt = Rune.dtype m in
                  let m_hat = Rune.div m (Rune.scalar dev dt bc1) in
                  Rune.(
                    mul (scalar dev dt lr)
                      (div m_hat (add (sqrt (abs v)) (scalar dev dt eps)))))
            in
            (updates, State (new_mu, new_nu, count)));
  }

(* Utilities *)

let apply_updates params updates =
  map_params2 params updates (fun p u -> Rune.sub p u)

let rec apply_updates_inplace : type a b. (a, b) params -> (a, b) params -> unit
    =
 fun params updates ->
  match (params, updates) with
  | Tensor t, Tensor u -> ignore (Rune.isub t u)
  | List ps, List us -> List.iter2 apply_updates_inplace ps us
  | Record ps, Record us ->
      let sorted_ps =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) ps
      in
      let sorted_us =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) us
      in
      List.iter2
        (fun (k1, p) (k2, u) ->
          assert (k1 = k2);
          apply_updates_inplace p u)
        sorted_ps sorted_us
  | _ -> failwith "apply_updates_inplace: parameter structure mismatch"

let global_norm params =
  let flat_params = flatten_params params in
  let norm_sq =
    List.fold_left
      (fun acc p ->
        let p_sq = Rune.sum (Rune.mul p p) in
        Rune.add acc p_sq)
      (let p0 = List.hd flat_params in
       Rune.scalar (Rune.device p0) (Rune.dtype p0) 0.)
      flat_params
  in
  Rune.unsafe_get [] (Rune.sqrt norm_sq)

let set_to_zero params = map_params params (fun t -> Rune.zeros_like t)

(* Multi-step wrapper *)
let multi_steps ~every transform =
  {
    init =
      (fun params ->
        let inner_state = transform.init params in
        let grads_accum = map_params params (fun t -> Rune.zeros_like t) in
        State (inner_state, grads_accum, 0));
    update =
      (fun state params grads ->
        match state with
        | State s ->
            let inner_state, grads_accum, step = Obj.magic s in
            let step = step + 1 in
            let new_grads_accum = map_params2 grads_accum grads Rune.add in
            if step mod every = 0 then
              (* Apply accumulated gradients *)
              let avg_grads =
                map_params new_grads_accum (fun g ->
                    let dev = Rune.device g in
                    let dt = Rune.dtype g in
                    Rune.div g (Rune.scalar dev dt (float_of_int every)))
              in
              let updates, new_inner_state =
                transform.update inner_state params avg_grads
              in
              let zero_accum =
                map_params new_grads_accum (fun t -> Rune.zeros_like t)
              in
              (updates, State (new_inner_state, zero_accum, step))
            else
              (* Just accumulate *)
              let zero_updates =
                map_params grads (fun t -> Rune.zeros_like t)
              in
              (zero_updates, State (inner_state, new_grads_accum, step)));
  }

(* Debugging *)
let with_gradient_stats ?(prefix = "") transform =
  {
    init = transform.init;
    update =
      (fun state params grads ->
        (* Print gradient statistics *)
        let norm = global_norm grads in
        Printf.printf "%sGradient norm: %.6f\n" prefix norm;
        transform.update state params grads);
  }
