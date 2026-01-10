(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype

let value_and_grad f params =
  let leaves, rebuild = Ptree.flatten params in
  let leaf_array = Array.of_list leaves in
  let grad_array : Ptree.tensor option array =
    Array.make (Array.length leaf_array) None
  in

  let float_groups_tbl : (string, Ptree.tensor * int list) Hashtbl.t =
    Hashtbl.create 8
  in

  Array.iteri
    (fun idx -> function
      | Ptree.P tensor ->
          let dtype = Rune.dtype tensor in
          if Dtype.is_float dtype then
            let key = Dtype.to_string dtype in
            match Hashtbl.find_opt float_groups_tbl key with
            | None -> Hashtbl.add float_groups_tbl key (Ptree.P tensor, [ idx ])
            | Some (repr, indices) ->
                Hashtbl.replace float_groups_tbl key (repr, idx :: indices)
          else
            invalid_arg
              "Transformations.value_and_grad: cannot differentiate w.r.t \
               non-floating tensor")
    leaf_array;

  let groups =
    Hashtbl.fold
      (fun _ (repr, indices) acc -> (repr, List.rev indices) :: acc)
      float_groups_tbl []
  in

  let value_ref = ref None in

  List.iter
    (fun (repr, indices) ->
      Ptree.with_tensor repr
        {
          run =
            (fun (type a) (type layout) (repr_tensor : (a, layout) Rune.t) ->
              let dtype = Rune.dtype repr_tensor in
              let rec collect remaining acc =
                match remaining with
                | [] -> List.rev acc
                | idx :: rest ->
                    let tensor =
                      Ptree.with_tensor leaf_array.(idx)
                        {
                          run =
                            (fun (type a_idx)
                              (type layout_idx)
                              (tensor : (a_idx, layout_idx) Rune.t)
                            ->
                              match
                                Dtype.equal_witness dtype (Rune.dtype tensor)
                              with
                              | Some Type.Equal -> (tensor : (a, layout) Rune.t)
                              | None ->
                                  invalid_arg
                                    "Transformations.value_and_grad: tensor \
                                     dtype mismatch");
                        }
                    in
                    collect rest (tensor :: acc)
              in
              let typed_inputs = collect indices [] in
              let f_group inputs =
                let updated = Array.copy leaf_array in
                List.iter2
                  (fun idx tensor -> updated.(idx) <- Ptree.P tensor)
                  indices inputs;
                f (rebuild (Array.to_list updated))
              in
              let value, grad_inputs =
                Rune.value_and_grads f_group typed_inputs
              in
              if !value_ref = None then value_ref := Some value;
              List.iter2
                (fun idx grad_tensor ->
                  grad_array.(idx) <- Some (Ptree.P grad_tensor))
                indices grad_inputs);
        })
    groups;

  let value = match !value_ref with Some v -> v | None -> f params in

  let grad_leaves =
    Array.map
      (function
        | Some pack -> pack
        | None ->
            invalid_arg "Transformations.value_and_grad: internal grad missing")
      grad_array
    |> Array.to_list
  in
  (value, rebuild grad_leaves)

let grad f params =
  let _, grads = value_and_grad f params in
  grads
