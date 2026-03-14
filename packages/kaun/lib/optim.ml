(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype

(* Helpers *)

let err_expected_float_dtype = "Optim: expected floating-point dtype"

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

(* Per-leaf packed Vega state with captured dtype for type unification *)

type packed_vega_state =
  | PVS : {
      dtype : ('a, 'b) Dtype.t;
      st : ('a, 'b) Vega.state;
    }
      -> packed_vega_state

(* State *)

type state = { tx : Vega.t; leaf_states : packed_vega_state array }

(* Init *)

let init tx params =
  let leaves, _ = Ptree.flatten params in
  let leaf_states =
    Array.of_list
      (List.map
         (fun pt ->
           Ptree.with_tensor pt
             {
               run = (fun t -> PVS { dtype = Nx.dtype t; st = Vega.init tx t });
             })
         leaves)
  in
  { tx; leaf_states }

(* Update: returns updates tree (not new params) *)

let update st params grads =
  let param_leaves, rebuild = Ptree.flatten params in
  let grad_leaves, _ = Ptree.flatten grads in
  let n = Array.length st.leaf_states in
  let update_packed = Array.make n (List.hd param_leaves) in
  let new_leaf_states = Array.make n st.leaf_states.(0) in
  List.iteri
    (fun i param_pt ->
      let grad_pt = List.nth grad_leaves i in
      let (PVS { dtype = dt; st = vega_st }) = st.leaf_states.(i) in
      let param_t = Ptree.Tensor.to_typed_exn dt param_pt in
      let grad_t = Ptree.Tensor.to_typed_exn dt grad_pt in
      let upd, new_vega_st = Vega.update vega_st ~grad:grad_t ~param:param_t in
      update_packed.(i) <- Ptree.P upd;
      new_leaf_states.(i) <- PVS { dtype = dt; st = new_vega_st })
    param_leaves;
  let updates = rebuild (Array.to_list update_packed) in
  (updates, { tx = st.tx; leaf_states = new_leaf_states })

(* Apply updates: add updates to params *)

let apply_updates params updates =
  Ptree.map2 { run = (fun param upd -> Nx.add param upd) } params updates

(* Step: convenience for update + apply_updates *)

let step st params grads =
  let updates, new_st = update st params grads in
  let new_params = apply_updates params updates in
  (new_params, new_st)

(* Serialization *)

let state_to_trees st =
  let n = Array.length st.leaf_states in
  if n = 0 then (0, [])
  else
    (* Get count from first leaf (all leaves share the same count) *)
    let (PVS { st = first_st; _ }) = st.leaf_states.(0) in
    let count, _ = Vega.state_to_tensors first_st in
    (* Extract per-leaf tensor arrays *)
    let per_leaf_tensors = Array.make n [||] in
    for i = 0 to n - 1 do
      let (PVS { st = vega_st; _ }) = st.leaf_states.(i) in
      let _, tensors = Vega.state_to_tensors vega_st in
      per_leaf_tensors.(i) <- Array.map (fun t -> Ptree.P t) tensors
    done;
    (* Determine number of state tensors per leaf *)
    let n_tensors = Array.length per_leaf_tensors.(0) in
    if n_tensors = 0 then (count, [])
    else
      (* Transpose: per-leaf x per-tensor -> per-tensor x per-leaf *)
      let tensor_trees =
        List.init n_tensors (fun m ->
            let leaves =
              List.init n (fun i -> Ptree.Tensor per_leaf_tensors.(i).(m))
            in
            Ptree.List leaves)
      in
      (count, tensor_trees)

let state_of_trees tx ~count trees =
  let n_trees = List.length trees in
  let expected_tensors = Vega.n_tensors tx in
  if n_trees <> expected_tensors then
    invalid_arg
      (Printf.sprintf "Optim.state_of_trees: expected %d moment trees, got %d"
         expected_tensors n_trees);
  if n_trees = 0 then { tx; leaf_states = [||] }
  else
    let first_tree = List.hd trees in
    let first_items = Ptree.List.items_exn first_tree in
    let n_leaves = List.length first_items in
    (* Collect per-tensor leaf lists *)
    let tensor_leaves =
      List.map
        (fun tree -> List.map Ptree.as_tensor_exn (Ptree.List.items_exn tree))
        trees
    in
    (* Transpose: per-tensor x per-leaf -> per-leaf x per-tensor *)
    let leaf_states =
      Array.init n_leaves (fun i ->
          let leaf_tensors =
            List.map (fun moment -> List.nth moment i) tensor_leaves
          in
          (* Use the first tensor's dtype as reference *)
          let ref_pt = List.hd leaf_tensors in
          Ptree.with_tensor ref_pt
            {
              run =
                (fun ref_t ->
                  let dt = Nx.dtype ref_t in
                  let typed_tensors =
                    Array.of_list
                      (List.map (Ptree.Tensor.to_typed_exn dt) leaf_tensors)
                  in
                  let vega_st = Vega.state_of_tensors tx ~count typed_tensors in
                  PVS { dtype = dt; st = vega_st });
            })
    in
    { tx; leaf_states }

(* Gradient utilities *)

let global_norm t =
  let sum_sq = Ptree.fold (fun acc p -> acc +. tensor_sum_sq p) 0. t in
  sqrt sum_sq

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
