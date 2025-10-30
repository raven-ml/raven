(* optimizer.ml *)

module Snapshot = Checkpoint.Snapshot
module Dtype = Nx_core.Dtype

let flatten_params params =
  let flat_list, _ = Ptree.flatten params in
  flat_list

let float_of_scalar (type a b) (dtype : (a, b) Dtype.t) (value : a) =
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
  | _ ->
      invalid_arg
        "Optimizer: expected floating-point dtype when converting to float"

type 'state spec = {
  init : Ptree.t -> 'state;
  update : 'state -> Ptree.t -> Ptree.t -> Ptree.t * 'state;
  encode : 'state -> Snapshot.t;
  decode : Snapshot.t -> ('state, string) result;
}

type algorithm =
  | Transform : { name : string; spec : 'state spec } -> algorithm

type state = State : 'state * 'state spec -> state

let make ~name spec = Transform { name; spec }
let name (Transform { name; _ }) = name

let init (Transform { spec; _ }) params =
  let value = spec.init params in
  State (value, spec)

let step (Transform _) (State (value, spec)) params grads =
  let updates, next = spec.update value params grads in
  (updates, State (next, spec))

let serialize (State (value, spec)) = spec.encode value

let restore (Transform { spec; _ }) tree =
  match spec.decode tree with
  | Ok value -> Ok (State (value, spec))
  | Error _ as err -> err

let rec find_int field =
  let open Snapshot in
  function
  | Scalar _ | Tensor _ -> None
  | List items ->
      let rec loop = function
        | [] -> None
        | hd :: tl -> (
            match find_int field hd with
            | Some _ as res -> res
            | None -> loop tl)
      in
      loop items
  | Record record ->
      let entries = Snapshot.Record.bindings record in
      let rec fold = function
        | [] -> None
        | (key, value) :: rest -> (
            if String.equal key field then
              match value with
              | Scalar (Int i) -> Some i
              | Scalar (Float f) -> Some (int_of_float f)
              | _ -> fold rest
            else
              match find_int field value with
              | Some _ as res -> res
              | None -> fold rest)
      in
      fold entries

let step_count state = find_int "count" (serialize state)
let encode_unit () = Snapshot.string "unit"

let decode_unit label = function
  | Snapshot.Scalar (Snapshot.String "unit") | Snapshot.Scalar (Snapshot.Int 0)
    ->
      Ok ()
  | _ -> Error (Printf.sprintf "%s: expected unit state" label)

let encode_ptree state = Snapshot.ptree state

let decode_ptree label tree =
  match Snapshot.to_ptree tree with
  | Ok ptree -> Ok ptree
  | Error msg -> Error (Printf.sprintf "%s: %s" label msg)

let identity () =
  let spec =
    {
      init = (fun _ -> ());
      update = (fun state _params grads -> (grads, state));
      encode = (fun () -> encode_unit ());
      decode = decode_unit "Optimizer.identity";
    }
  in
  make ~name:"identity" spec

let scale factor =
  let spec =
    {
      init = (fun _ -> ());
      update =
        (fun state _params grads ->
          let updates =
            Ptree.map (fun g -> Rune.(mul g (scalar (dtype g) factor))) grads
          in
          (updates, state));
      encode = (fun () -> encode_unit ());
      decode = decode_unit "Optimizer.scale";
    }
  in
  make ~name:"scale" spec

let scale_by_neg_one () = scale (-1.)

let add_decayed_weights weight_decay =
  let spec =
    {
      init = (fun _ -> ());
      update =
        (fun state params grads ->
          let updates =
            Ptree.map2
              (fun g p ->
                let dt = Rune.dtype g in
                Rune.(add g (mul p (scalar dt weight_decay))))
              grads params
          in
          (updates, state));
      encode = (fun () -> encode_unit ());
      decode = decode_unit "Optimizer.add_decayed_weights";
    }
  in
  make ~name:"add_decayed_weights" spec

let clip_by_global_norm max_norm =
  let spec =
    {
      init = (fun _ -> ());
      update =
        (fun state _params grads ->
          let flat_grads = flatten_params grads in
          let sum_sq =
            List.fold_left
              (fun acc -> function
                | Ptree.P tensor ->
                    let dtype = Rune.dtype tensor in
                    if Dtype.is_float dtype then
                      let squared = Rune.mul tensor tensor in
                      let total = Rune.sum squared |> Rune.item [] in
                      acc +. float_of_scalar dtype total
                    else
                      invalid_arg
                        "Optimizer.clip_by_global_norm: gradient must be float")
              0. flat_grads
          in
          if sum_sq = 0. then (grads, state)
          else
            let norm = sqrt sum_sq in
            let eps = 1e-12 in
            let safe_norm = if norm < eps then eps else norm in
            let scale =
              if max_norm >= safe_norm then 1. else max_norm /. safe_norm
            in
            if Float.equal scale 1. then (grads, state)
            else
              let scaled_grads =
                Ptree.map
                  (fun g ->
                    let dtype = Rune.dtype g in
                    let scalar = Rune.scalar dtype scale in
                    Rune.mul g scalar)
                  grads
              in
              (scaled_grads, state));
      encode = (fun () -> encode_unit ());
      decode = decode_unit "Optimizer.clip_by_global_norm";
    }
  in
  make ~name:"clip_by_global_norm" spec

let clip max_delta =
  let spec =
    {
      init = (fun _ -> ());
      update =
        (fun state _params grads ->
          let clipped =
            Ptree.map
              (fun g ->
                let dt = Rune.dtype g in
                Rune.maximum
                  (Rune.minimum g (Rune.scalar dt max_delta))
                  (Rune.scalar dt (-.max_delta)))
              grads
          in
          (clipped, state));
      encode = (fun () -> encode_unit ());
      decode = decode_unit "Optimizer.clip";
    }
  in
  make ~name:"clip" spec

let trace ~decay ?(nesterov = false) () =
  let spec =
    {
      init = (fun params -> Ptree.map (fun t -> Rune.zeros_like t) params);
      update =
        (fun momentum _params grads ->
          let new_momentum =
            Ptree.map2
              (fun m g ->
                let dt = Rune.dtype g in
                Rune.(add (mul m (scalar dt decay)) g))
              momentum grads
          in
          let updates =
            if nesterov then
              Ptree.map2
                (fun g m ->
                  let dt = Rune.dtype g in
                  Rune.(add g (mul m (scalar dt decay))))
                grads new_momentum
            else new_momentum
          in
          (updates, new_momentum));
      encode = encode_ptree;
      decode = decode_ptree "Optimizer.trace";
    }
  in
  make ~name:"trace" spec

let scale_by_rms ?(decay = 0.999) ?(eps = 1e-8) () =
  let spec =
    {
      init = (fun params -> Ptree.map (fun t -> Rune.zeros_like t) params);
      update =
        (fun nu _params grads ->
          let new_nu =
            Ptree.map2
              (fun v g ->
                let dt = Rune.dtype g in
                Rune.(
                  add
                    (mul v (scalar dt decay))
                    (mul (mul g g) (scalar dt (1. -. decay)))))
              nu grads
          in
          let updates =
            Ptree.map2
              (fun g v ->
                let dt = Rune.dtype g in
                Rune.(div g (add (sqrt v) (scalar dt eps))))
              grads new_nu
          in
          (updates, new_nu));
      encode = encode_ptree;
      decode = decode_ptree "Optimizer.scale_by_rms";
    }
  in
  make ~name:"scale_by_rms" spec

let scale_by_adam ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  let encode (mu, nu, count) =
    Snapshot.record
      [
        ("kind", Snapshot.string "adam_inner");
        ("mu", Snapshot.ptree mu);
        ("nu", Snapshot.ptree nu);
        ("count", Snapshot.int count);
      ]
  in
  let decode tree =
    let open Result in
    let ( let* ) = bind in
    match tree with
    | Snapshot.Record record ->
        let* () =
          match Snapshot.Record.find_opt "kind" record with
          | None -> Ok ()
          | Some (Snapshot.Scalar (Snapshot.String kind)) ->
              if String.equal kind "adam_inner" then Ok ()
              else
                Error
                  (Printf.sprintf "Optimizer.scale_by_adam: unexpected kind %s"
                     kind)
          | Some _ -> Error "Optimizer.scale_by_adam: invalid kind"
        in
        let find key =
          match Snapshot.Record.find_opt key record with
          | Some value -> Ok value
          | None ->
              Error
                (Printf.sprintf "Optimizer.scale_by_adam: missing field %s" key)
        in
        let* mu_node = find "mu" in
        let* nu_node = find "nu" in
        let* count_node = find "count" in
        let* mu = Snapshot.to_ptree mu_node in
        let* nu = Snapshot.to_ptree nu_node in
        let count =
          match count_node with
          | Snapshot.Scalar (Snapshot.Int i) -> Ok i
          | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
          | _ -> Error "Optimizer.scale_by_adam: expected integer count"
        in
        let* count = count in
        Ok (mu, nu, count)
    | _ -> Error "Optimizer.scale_by_adam: expected record"
  in
  let spec =
    {
      init =
        (fun params ->
          let zeros = Ptree.map (fun t -> Rune.zeros_like t) params in
          (zeros, zeros, 0));
      update =
        (fun (mu, nu, count) _params grads ->
          let count = count + 1 in
          let new_mu =
            Ptree.map2
              (fun m g ->
                let dt = Rune.dtype g in
                Rune.(add (mul m (scalar dt b1)) (mul g (scalar dt (1. -. b1)))))
              mu grads
          in
          let new_nu =
            Ptree.map2
              (fun v g ->
                let dt = Rune.dtype g in
                Rune.(
                  add
                    (mul v (scalar dt b2))
                    (mul (mul g g) (scalar dt (1. -. b2)))))
              nu grads
          in
          let bc1 = 1. -. (b1 ** float_of_int count) in
          let bc2 = 1. -. (b2 ** float_of_int count) in
          let updates =
            Ptree.map2
              (fun m v ->
                let dt = Rune.dtype m in
                let m_hat = Rune.div m (Rune.scalar dt bc1) in
                let v_hat = Rune.div v (Rune.scalar dt bc2) in
                Rune.(div m_hat (add (sqrt v_hat) (scalar dt eps))))
              new_mu new_nu
          in
          (updates, (new_mu, new_nu, count)));
      encode;
      decode;
    }
  in
  make ~name:"scale_by_adam" spec

let scale_by_belief ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-16) () =
  let encode (mu, s, count) =
    Snapshot.record
      [
        ("kind", Snapshot.string "adabelief_inner");
        ("mu", Snapshot.ptree mu);
        ("s", Snapshot.ptree s);
        ("count", Snapshot.int count);
      ]
  in
  let decode tree =
    let open Result in
    let ( let* ) = bind in
    match tree with
    | Snapshot.Record record ->
        let* () =
          match Snapshot.Record.find_opt "kind" record with
          | None -> Ok ()
          | Some (Snapshot.Scalar (Snapshot.String kind)) ->
              if String.equal kind "adabelief_inner" then Ok ()
              else
                Error
                  (Printf.sprintf
                     "Optimizer.scale_by_belief: unexpected kind %s" kind)
          | Some _ -> Error "Optimizer.scale_by_belief: invalid kind"
        in
        let find key =
          match Snapshot.Record.find_opt key record with
          | Some value -> Ok value
          | None ->
              Error
                (Printf.sprintf "Optimizer.scale_by_belief: missing field %s"
                   key)
        in
        let* mu_node = find "mu" in
        let* s_node = find "s" in
        let* count_node = find "count" in
        let* mu = Snapshot.to_ptree mu_node in
        let* s = Snapshot.to_ptree s_node in
        let count =
          match count_node with
          | Snapshot.Scalar (Snapshot.Int i) -> Ok i
          | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
          | _ -> Error "Optimizer.scale_by_belief: expected integer count"
        in
        let* count = count in
        Ok (mu, s, count)
    | _ -> Error "Optimizer.scale_by_belief: expected record"
  in
  let spec =
    {
      init =
        (fun params ->
          let zeros = Ptree.map (fun t -> Rune.zeros_like t) params in
          (zeros, zeros, 0));
      update =
        (fun (mu, s, count) _params grads ->
          let count = count + 1 in
          let new_mu =
            Ptree.map2
              (fun m g ->
                let dt = Rune.dtype g in
                Rune.(add (mul m (scalar dt b1)) (mul g (scalar dt (1. -. b1)))))
              mu grads
          in
          let diff = Ptree.map2 Rune.sub grads new_mu in
          let new_s =
            Ptree.map2
              (fun s_old d ->
                let dt = Rune.dtype s_old in
                Rune.(
                  add
                    (mul s_old (scalar dt b2))
                    (mul (mul d d) (scalar dt (1. -. b2)))))
              s diff
          in
          let bc1 = 1. -. (b1 ** float_of_int count) in
          let bc2 = 1. -. (b2 ** float_of_int count) in
          let updates =
            Ptree.map2
              (fun m s_val ->
                let dt = Rune.dtype m in
                let m_hat = Rune.div m (Rune.scalar dt bc1) in
                let s_hat = Rune.div s_val (Rune.scalar dt bc2) in
                Rune.(div m_hat (add (sqrt s_hat) (scalar dt eps))))
              new_mu new_s
          in
          (updates, (new_mu, new_s, count)));
      encode;
      decode;
    }
  in
  make ~name:"scale_by_belief" spec

module Schedule = struct
  type t = int -> float

  let constant value _ = value

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
      let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
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
      let cosine_val = 0.5 *. (1. -. Stdlib.cos (Float.pi *. ratio)) in
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
  let spec =
    {
      init = (fun _ -> 0);
      update =
        (fun step _params grads ->
          let step = step + 1 in
          let lr = schedule step in
          let updates =
            Ptree.map
              (fun g ->
                let dt = Rune.dtype g in
                Rune.mul g (Rune.scalar dt lr))
              grads
          in
          (updates, step));
      encode = (fun step -> Snapshot.int step);
      decode =
        (function
        | Snapshot.Scalar (Snapshot.Int step) -> Ok step
        | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
        | _ -> Error "Optimizer.scale_by_schedule: expected integer step");
    }
  in
  make ~name:"scale_by_schedule" spec

let chain transforms =
  let spec =
    {
      init = (fun params -> List.map (fun t -> init t params) transforms);
      update =
        (fun states params grads ->
          let updates, new_states_rev =
            List.fold_left2
              (fun (current_grads, acc_states) transform state ->
                let updated_grads, new_state =
                  step transform state params current_grads
                in
                (updated_grads, new_state :: acc_states))
              (grads, []) transforms states
          in
          (updates, List.rev new_states_rev));
      encode =
        (fun states ->
          let encodes =
            if List.length states <> List.length transforms then
              invalid_arg "Optimizer.chain: state length mismatch"
            else List.map serialize states
          in
          Snapshot.list encodes);
      decode =
        (fun tree ->
          match tree with
          | Snapshot.List nodes ->
              if List.length nodes <> List.length transforms then
                Error "Optimizer.chain: checkpoint length mismatch"
              else
                let rec decode_all ts ns acc =
                  match (ts, ns) with
                  | [], [] -> Ok (List.rev acc)
                  | transform :: ts_rest, node :: ns_rest -> (
                      match restore transform node with
                      | Ok state -> decode_all ts_rest ns_rest (state :: acc)
                      | Error _ as err -> err)
                  | _ ->
                      Error "Optimizer.chain: inconsistent checkpoint structure"
                in
                decode_all transforms nodes []
          | _ -> Error "Optimizer.chain: expected list");
    }
  in
  make ~name:"chain" spec

type label_tree =
  | Label_tensor of int
  | Label_list of label_tree list
  | Label_record of (string * label_tree) list

type mask_tree =
  | Mask_tensor of bool
  | Mask_list of mask_tree list
  | Mask_record of (string * mask_tree) list

let rec apply_mask mask params grads =
  match (mask, params, grads) with
  | Mask_tensor true, Ptree.Tensor _, Ptree.Tensor pack -> Ptree.Tensor pack
  | Mask_tensor false, Ptree.Tensor (Ptree.P p), Ptree.Tensor _ ->
      Ptree.tensor (Rune.zeros_like p)
  | Mask_list masks, Ptree.List ps, Ptree.List gs ->
      Ptree.list
        (List.map
           (fun ((m, p), g) -> apply_mask m p g)
           (List.combine (List.combine masks ps) gs))
  | Mask_record mask_fields, Ptree.Dict param_fields, Ptree.Dict grad_fields ->
      let result_rev =
        List.fold_left
          (fun acc (key, mask_child) ->
            match
              (List.assoc_opt key param_fields, List.assoc_opt key grad_fields)
            with
            | Some p, Some g -> (key, apply_mask mask_child p g) :: acc
            | _ -> failwith ("apply_mask: missing field " ^ key))
          [] mask_fields
      in
      Ptree.Dict (List.rev result_rev)
  | _ -> failwith "apply_mask: structure mismatch"

let multi_transform ~transforms ~labels =
  let transforms_array = Array.of_list transforms in
  let spec =
    {
      init = (fun params -> Array.map (fun t -> init t params) transforms_array);
      update =
        (fun states params grads ->
          let label_tree = labels params in
          let all_updates =
            Array.mapi
              (fun idx transform ->
                let rec filter_by_label target label params grads =
                  match (label, params, grads) with
                  | ( Label_tensor label_idx,
                      Ptree.Tensor (Ptree.P _),
                      Ptree.Tensor (Ptree.P g) ) ->
                      if label_idx = target then Ptree.tensor g
                      else Ptree.tensor (Rune.zeros_like g)
                  | Label_list ls, Ptree.List ps, Ptree.List gs ->
                      Ptree.list
                        (List.map
                           (fun ((l, p), g) -> filter_by_label target l p g)
                           (List.combine (List.combine ls ps) gs))
                  | ( Label_record fields_l,
                      Ptree.Dict fields_p,
                      Ptree.Dict fields_g ) ->
                      let result_rev =
                        List.fold_left
                          (fun acc (key, lbl) ->
                            match
                              ( List.assoc_opt key fields_p,
                                List.assoc_opt key fields_g )
                            with
                            | Some p, Some g ->
                                (key, filter_by_label target lbl p g) :: acc
                            | _ ->
                                failwith
                                  ("multi_transform: missing field " ^ key))
                          [] fields_l
                      in
                      Ptree.Dict (List.rev result_rev)
                  | _ -> failwith "multi_transform: structure mismatch"
                in
                let filtered_grads =
                  filter_by_label idx label_tree params grads
                in
                let updates, new_state =
                  step transform states.(idx) params filtered_grads
                in
                (updates, new_state))
              transforms_array
          in
          let combined_updates =
            Array.fold_left
              (fun acc (updates, _) -> Ptree.map2 Rune.add acc updates)
              (Ptree.map (fun t -> Rune.zeros_like t) grads)
              all_updates
          in
          let new_states = Array.map snd all_updates in
          (combined_updates, new_states));
      encode =
        (fun states ->
          let nodes = Array.to_list (Array.map serialize states) in
          Snapshot.list nodes);
      decode =
        (fun tree ->
          match tree with
          | Snapshot.List nodes ->
              if List.length nodes <> Array.length transforms_array then
                Error "Optimizer.multi_transform: checkpoint length mismatch"
              else
                let transforms = Array.to_list transforms_array in
                let rec decode_all transforms nodes acc =
                  match (transforms, nodes) with
                  | [], [] -> Ok (Array.of_list (List.rev acc))
                  | transform :: ts, node :: ns -> (
                      match restore transform node with
                      | Ok state -> decode_all ts ns (state :: acc)
                      | Error _ as err -> err)
                  | _ ->
                      Error "Optimizer.multi_transform: inconsistent checkpoint"
                in
                decode_all transforms nodes []
          | _ -> Error "Optimizer.multi_transform: expected list");
    }
  in
  make ~name:"multi_transform" spec

let masked ~mask ~inner =
  let spec =
    {
      init = (fun params -> init inner params);
      update =
        (fun state params grads ->
          let mask_tree = mask params in
          let masked_grads = apply_mask mask_tree params grads in
          let updates, inner_state = step inner state params masked_grads in
          (updates, inner_state));
      encode = (fun state -> serialize state);
      decode = (fun tree -> restore inner tree);
    }
  in
  make ~name:"masked" spec

let apply_updates params updates =
  Ptree.map2 (fun p u -> Rune.add p u) params updates

let rec apply_updates_inplace params updates =
  match (params, updates) with
  | Ptree.Tensor (Ptree.P t), Ptree.Tensor (Ptree.P u) -> (
      match Dtype.equal_witness (Rune.dtype t) (Rune.dtype u) with
      | Some Type.Equal -> ignore (Rune.iadd t u)
      | None -> invalid_arg "apply_updates_inplace: dtype mismatch")
  | Ptree.List ps, Ptree.List us -> List.iter2 apply_updates_inplace ps us
  | Ptree.Dict ps, Ptree.Dict us ->
      let sorted_ps =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) ps
      in
      let sorted_us =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) us
      in
      List.iter2
        (fun (k1, p) (k2, u) ->
          assert (String.equal k1 k2);
          apply_updates_inplace p u)
        sorted_ps sorted_us
  | _ -> failwith "apply_updates_inplace: parameter structure mismatch"

let global_norm params =
  let flat_params = flatten_params params in
  let sum_sq =
    List.fold_left
      (fun acc -> function
        | Ptree.P tensor ->
            let dtype = Rune.dtype tensor in
            if Dtype.is_float dtype then
              let squared = Rune.mul tensor tensor in
              let total = Rune.sum squared |> Rune.item [] in
              acc +. float_of_scalar dtype total
            else invalid_arg "Optimizer.global_norm: expected float tensor")
      0. flat_params
  in
  sqrt sum_sq

let set_to_zero params = Ptree.map (fun t -> Rune.zeros_like t) params

let multi_steps ~every transform =
  let spec =
    {
      init =
        (fun params ->
          let inner_state = init transform params in
          let grads_accum = Ptree.map (fun t -> Rune.zeros_like t) params in
          (inner_state, grads_accum, 0));
      update =
        (fun (inner_state, grads_accum, step_count) params grads ->
          let step_count = step_count + 1 in
          let new_grads_accum = Ptree.map2 Rune.add grads_accum grads in
          if step_count mod every = 0 then
            let avg_grads =
              Ptree.map
                (fun g ->
                  let dt = Rune.dtype g in
                  Rune.div g (Rune.scalar dt (float_of_int every)))
                new_grads_accum
            in
            let updates, inner_state =
              step transform inner_state params avg_grads
            in
            let zero_accum =
              Ptree.map (fun t -> Rune.zeros_like t) new_grads_accum
            in
            (updates, (inner_state, zero_accum, step_count))
          else
            let zero_updates = Ptree.map (fun t -> Rune.zeros_like t) grads in
            (zero_updates, (inner_state, new_grads_accum, step_count)));
      encode =
        (fun (inner_state, grads_accum, step) ->
          Snapshot.record
            [
              ("inner", serialize inner_state);
              ("accum", Snapshot.ptree grads_accum);
              ("step", Snapshot.int step);
            ]);
      decode =
        (fun tree ->
          let open Result in
          let ( let* ) = bind in
          match tree with
          | Snapshot.Record record ->
              let find key =
                match Snapshot.Record.find_opt key record with
                | Some value -> Ok value
                | None -> Error ("Optimizer.multi_steps: missing " ^ key)
              in
              let* inner_node = find "inner" in
              let* accum_node = find "accum" in
              let* step_node = find "step" in
              let* inner_state = restore transform inner_node in
              let* accum_ptree = Snapshot.to_ptree accum_node in
              let step =
                match step_node with
                | Snapshot.Scalar (Snapshot.Int i) -> Ok i
                | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
                | _ -> Error "Optimizer.multi_steps: expected integer step"
              in
              let* step = step in
              Ok (inner_state, accum_ptree, step)
          | _ -> Error "Optimizer.multi_steps: expected record");
    }
  in
  make ~name:"multi_steps" spec

let with_gradient_stats ?(prefix = "") transform =
  let spec =
    {
      init = (fun params -> init transform params);
      update =
        (fun state params grads ->
          let norm = global_norm grads in
          Printf.printf "%sGradient norm: %.6f\n" prefix norm;
          step transform state params grads);
      encode = (fun state -> serialize state);
      decode = (fun tree -> restore transform tree);
    }
  in
  make ~name:"with_gradient_stats" spec

let sgd ~lr ?(momentum = 0.) ?(nesterov = false) () =
  let base =
    if momentum > 0. then
      chain
        [
          trace ~decay:momentum ~nesterov ();
          scale_by_neg_one ();
          scale_by_schedule lr;
        ]
    else chain [ scale_by_neg_one (); scale_by_schedule lr ]
  in
  base

let adam ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  chain
    [ scale_by_adam ~b1 ~b2 ~eps (); scale_by_neg_one (); scale_by_schedule lr ]

let adamw ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.01) ()
    =
  chain
    [
      scale_by_adam ~b1 ~b2 ~eps ();
      add_decayed_weights weight_decay;
      scale_by_neg_one ();
      scale_by_schedule lr;
    ]

let rmsprop ~lr ?(decay = 0.9) ?(eps = 1e-8) ?(momentum = 0.) () =
  let base =
    if momentum > 0. then
      chain
        [
          scale_by_rms ~decay ~eps ();
          trace ~decay:momentum ();
          scale_by_neg_one ();
          scale_by_schedule lr;
        ]
    else
      chain
        [
          scale_by_rms ~decay ~eps (); scale_by_neg_one (); scale_by_schedule lr;
        ]
  in
  base

let adagrad ~lr ?(eps = 1e-8) () =
  let base_spec =
    {
      init = (fun params -> Ptree.map (fun t -> Rune.zeros_like t) params);
      update =
        (fun accum _params grads ->
          let new_accum =
            Ptree.map2 (fun acc g -> Rune.add acc (Rune.mul g g)) accum grads
          in
          let updates =
            Ptree.map2
              (fun g acc ->
                let dt = Rune.dtype g in
                Rune.(neg (div g (add (sqrt acc) (scalar dt eps)))))
              grads new_accum
          in
          (updates, new_accum));
      encode = encode_ptree;
      decode = decode_ptree "Optimizer.adagrad";
    }
  in
  let base = make ~name:"adagrad" base_spec in
  chain [ base; scale_by_schedule lr ]

(* Similar for others. *)
let adabelief ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-16) () =
  chain
    [
      scale_by_belief ~b1 ~b2 ~eps (); scale_by_neg_one (); scale_by_schedule lr;
    ]

let lamb ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-6) ?(weight_decay = 0.01) () =
  let encode (mu, nu, count) =
    Snapshot.record
      [
        ("kind", Snapshot.string "lamb_inner");
        ("mu", Snapshot.ptree mu);
        ("nu", Snapshot.ptree nu);
        ("count", Snapshot.int count);
      ]
  in
  let decode tree =
    let open Result in
    let ( let* ) = bind in
    match tree with
    | Snapshot.Record record ->
        let* () =
          match Snapshot.Record.find_opt "kind" record with
          | None -> Ok ()
          | Some (Snapshot.Scalar (Snapshot.String kind)) ->
              if String.equal kind "lamb_inner" then Ok ()
              else
                Error (Printf.sprintf "Optimizer.lamb: unexpected kind %s" kind)
          | Some _ -> Error "Optimizer.lamb: invalid kind"
        in
        let find key =
          match Snapshot.Record.find_opt key record with
          | Some value -> Ok value
          | None ->
              Error (Printf.sprintf "Optimizer.lamb: missing field %s" key)
        in
        let* mu_node = find "mu" in
        let* nu_node = find "nu" in
        let* count_node = find "count" in
        let* mu = Snapshot.to_ptree mu_node in
        let* nu = Snapshot.to_ptree nu_node in
        let count =
          match count_node with
          | Snapshot.Scalar (Snapshot.Int i) -> Ok i
          | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
          | _ -> Error "Optimizer.lamb: expected integer count"
        in
        let* count = count in
        Ok (mu, nu, count)
    | _ -> Error "Optimizer.lamb: expected record"
  in
  let spec =
    {
      init =
        (fun params ->
          let zeros = Ptree.map (fun t -> Rune.zeros_like t) params in
          (zeros, zeros, 0));
      update =
        (fun (mu, nu, count) params grads ->
          let count = count + 1 in
          let lr_t = lr count in
          let new_mu =
            Ptree.map2
              (fun m g ->
                let dt = Rune.dtype g in
                Rune.(add (mul m (scalar dt b1)) (mul g (scalar dt (1. -. b1)))))
              mu grads
          in
          let new_nu =
            Ptree.map2
              (fun v g ->
                let dt = Rune.dtype g in
                Rune.(
                  add
                    (mul v (scalar dt b2))
                    (mul (mul g g) (scalar dt (1. -. b2)))))
              nu grads
          in
          let bc1 = 1. -. (b1 ** float_of_int count) in
          let bc2 = 1. -. (b2 ** float_of_int count) in
          let raw_updates =
            Ptree.map2
              (fun m v ->
                let dt = Rune.dtype m in
                let m_hat = Rune.div m (Rune.scalar dt bc1) in
                let v_hat = Rune.div v (Rune.scalar dt bc2) in
                Rune.(
                  add
                    (div m_hat (add (sqrt v_hat) (scalar dt eps)))
                    (mul (scalar dt weight_decay) m)))
              new_mu new_nu
          in
          let trust_scaled =
            Ptree.map2
              (fun update param ->
                let dt = Rune.dtype update in
                let update_norm =
                  Rune.sqrt (Rune.sum (Rune.mul update update))
                in
                let param_norm = Rune.sqrt (Rune.sum (Rune.mul param param)) in
                let trust_ratio =
                  Rune.(
                    where
                      (greater param_norm (scalar dt 0.))
                      (div param_norm update_norm)
                      (scalar dt 1.))
                in
                Rune.mul trust_ratio update)
              raw_updates params
          in
          let updates =
            Ptree.map
              (fun u ->
                let dt = Rune.dtype u in
                Rune.mul u (Rune.scalar dt (-.lr_t)))
              trust_scaled
          in
          (updates, (new_mu, new_nu, count)));
      encode;
      decode;
    }
  in
  make ~name:"lamb" spec

(* Fix typos and integrate lr in update for lamb, radam, yogi to avoid chain if
   preferred, but for consistency, use chain. *)
let radam ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) () =
  let encode (mu, nu, count) =
    Snapshot.record
      [
        ("kind", Snapshot.string "radam_inner");
        ("mu", Snapshot.ptree mu);
        ("nu", Snapshot.ptree nu);
        ("count", Snapshot.int count);
      ]
  in
  let decode tree =
    let open Result in
    let ( let* ) = bind in
    match tree with
    | Snapshot.Record record ->
        let* () =
          match Snapshot.Record.find_opt "kind" record with
          | None -> Ok ()
          | Some (Snapshot.Scalar (Snapshot.String kind)) ->
              if String.equal kind "radam_inner" then Ok ()
              else
                Error
                  (Printf.sprintf "Optimizer.radam: unexpected kind %s" kind)
          | Some _ -> Error "Optimizer.radam: invalid kind"
        in
        let find key =
          match Snapshot.Record.find_opt key record with
          | Some value -> Ok value
          | None ->
              Error (Printf.sprintf "Optimizer.radam: missing field %s" key)
        in
        let* mu_node = find "mu" in
        let* nu_node = find "nu" in
        let* count_node = find "count" in
        let* mu = Snapshot.to_ptree mu_node in
        let* nu = Snapshot.to_ptree nu_node in
        let count =
          match count_node with
          | Snapshot.Scalar (Snapshot.Int i) -> Ok i
          | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
          | _ -> Error "Optimizer.radam: expected integer count"
        in
        let* count = count in
        Ok (mu, nu, count)
    | _ -> Error "Optimizer.radam: expected record"
  in
  let spec =
    {
      init =
        (fun params ->
          let zeros = Ptree.map (fun t -> Rune.zeros_like t) params in
          (zeros, zeros, 0));
      update =
        (fun (mu, nu, count) _params grads ->
          let count = count + 1 in
          let lr_t = lr count in
          let new_mu =
            Ptree.map2
              (fun m g ->
                let dt = Rune.dtype g in
                Rune.(add (mul m (scalar dt b1)) (mul g (scalar dt (1. -. b1)))))
              mu grads
          in
          let new_nu =
            Ptree.map2
              (fun v g ->
                let dt = Rune.dtype g in
                Rune.(
                  add
                    (mul v (scalar dt b2))
                    (mul (mul g g) (scalar dt (1. -. b2)))))
              nu grads
          in
          let rho_inf = (2. /. (1. -. b2)) -. 1. in
          let rho_t =
            rho_inf
            -. 2. *. float_of_int count
               *. (b2 ** float_of_int count)
               /. (1. -. (b2 ** float_of_int count))
          in
          let updates =
            if rho_t > 4. then
              let bc1 = 1. -. (b1 ** float_of_int count) in
              let bc2 = 1. -. (b2 ** float_of_int count) in
              let rect_term =
                Stdlib.sqrt
                  ((rho_t -. 4.) *. (rho_t -. 2.) *. rho_inf
                  /. ((rho_inf -. 4.) *. (rho_inf -. 2.) *. rho_t))
              in
              Ptree.map2
                (fun m v ->
                  let dt = Rune.dtype m in
                  let m_hat = Rune.div m (Rune.scalar dt bc1) in
                  let v_hat = Rune.div v (Rune.scalar dt bc2) in
                  Rune.(
                    mul
                      (scalar dt (-.(lr_t *. rect_term)))
                      (div m_hat (add (sqrt v_hat) (scalar dt eps)))))
                new_mu new_nu
            else
              let bc1 = 1. -. (b1 ** float_of_int count) in
              Ptree.map
                (fun m ->
                  let dt = Rune.dtype m in
                  Rune.(mul (scalar dt (-.lr_t)) (div m (scalar dt bc1))))
                new_mu
          in
          (updates, (new_mu, new_nu, count)));
      encode;
      decode;
    }
  in
  make ~name:"radam" spec

let yogi ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-3) () =
  let encode (mu, nu, count) =
    Snapshot.record
      [
        ("kind", Snapshot.string "yogi_inner");
        ("mu", Snapshot.ptree mu);
        ("nu", Snapshot.ptree nu);
        ("count", Snapshot.int count);
      ]
  in
  let decode tree =
    let open Result in
    let ( let* ) = bind in
    match tree with
    | Snapshot.Record record ->
        let* () =
          match Snapshot.Record.find_opt "kind" record with
          | None -> Ok ()
          | Some (Snapshot.Scalar (Snapshot.String kind)) ->
              if String.equal kind "yogi_inner" then Ok ()
              else
                Error (Printf.sprintf "Optimizer.yogi: unexpected kind %s" kind)
          | Some _ -> Error "Optimizer.yogi: invalid kind"
        in
        let find key =
          match Snapshot.Record.find_opt key record with
          | Some value -> Ok value
          | None ->
              Error (Printf.sprintf "Optimizer.yogi: missing field %s" key)
        in
        let* mu_node = find "mu" in
        let* nu_node = find "nu" in
        let* count_node = find "count" in
        let* mu = Snapshot.to_ptree mu_node in
        let* nu = Snapshot.to_ptree nu_node in
        let count =
          match count_node with
          | Snapshot.Scalar (Snapshot.Int i) -> Ok i
          | Snapshot.Scalar (Snapshot.Float f) -> Ok (int_of_float f)
          | _ -> Error "Optimizer.yogi: expected integer count"
        in
        let* count = count in
        Ok (mu, nu, count)
    | _ -> Error "Optimizer.yogi: expected record"
  in
  let spec =
    {
      init =
        (fun params ->
          let zeros = Ptree.map (fun t -> Rune.zeros_like t) params in
          (zeros, zeros, 0));
      update =
        (fun (mu, nu, count) _params grads ->
          let count = count + 1 in
          let new_mu =
            Ptree.map2
              (fun m g ->
                let dt = Rune.dtype g in
                Rune.(add (mul m (scalar dt b1)) (mul g (scalar dt (1. -. b1)))))
              mu grads
          in
          let new_nu =
            Ptree.map2
              (fun v g ->
                let dt = Rune.dtype g in
                let g_sq = Rune.mul g g in
                let sign_v_gsq = Rune.sign (Rune.sub v g_sq) in
                Rune.(sub v (mul (scalar dt (1. -. b2)) (mul g_sq sign_v_gsq))))
              nu grads
          in
          let bc1 = 1. -. (b1 ** float_of_int count) in
          let updates =
            Ptree.map2
              (fun m v ->
                let dt = Rune.dtype m in
                let m_hat = Rune.div m (Rune.scalar dt bc1) in
                Rune.(
                  mul
                    (scalar dt (-.lr count))
                    (div m_hat (add (sqrt (abs v)) (scalar dt eps)))))
              new_mu new_nu
          in
          (updates, (new_mu, new_nu, count)));
      encode;
      decode;
    }
  in
  make ~name:"yogi" spec
