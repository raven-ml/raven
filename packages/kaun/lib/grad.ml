(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let path_label path = if path = "" then "<root>" else path

let err_non_float fn_name path dtype =
  invalid_argf "%s: %s expected float dtype, got %s" fn_name (path_label path)
    (Dtype.to_string dtype)

let err_mismatch fn_name path expected actual =
  invalid_argf "%s: %s has dtype/layout %s but expected %s" fn_name
    (path_label path) (Dtype.to_string actual) (Dtype.to_string expected)

let value_and_grad_aux f params =
  let fn_name = "Grad.value_and_grad" in
  let leaves, rebuild = Ptree.flatten params in
  let path_leaves = Ptree.flatten_with_paths params in
  let leaf_count = List.length leaves in
  let path_count = List.length path_leaves in
  if leaf_count <> path_count then
    invalid_argf
      "%s: internal error: flatten/flatten_with_paths length mismatch (%d vs \
       %d)"
      fn_name leaf_count path_count;
  if leaf_count = 0 then
    let value, aux = f params in
    (value, rebuild [], aux)
  else
    let leaves_array = Array.of_list leaves in
    let paths_array = Array.of_list (List.map fst path_leaves) in
    let first_leaf = leaves_array.(0) in
    let first_path = paths_array.(0) in
    Ptree.with_tensor first_leaf
      {
        run =
          (fun (type a layout) (first_tensor : (a, layout) Nx.t) ->
            let first_dtype : (a, layout) Dtype.t = Nx.dtype first_tensor in
            if not (Dtype.is_float first_dtype) then
              err_non_float fn_name first_path first_dtype;
            let typed_inputs =
              List.mapi
                (fun index leaf ->
                  let path = paths_array.(index) in
                  Ptree.with_tensor leaf
                    {
                      run =
                        (fun (type a2 layout2) (tensor : (a2, layout2) Nx.t) ->
                          let dtype = Nx.dtype tensor in
                          if not (Dtype.is_float dtype) then
                            err_non_float fn_name path dtype;
                          match Dtype.equal_witness first_dtype dtype with
                          | Some Type.Equal -> (tensor : (a, layout) Nx.t)
                          | None -> err_mismatch fn_name path first_dtype dtype);
                    })
                leaves
            in
            let aux = ref None in
            let objective typed_params =
              let packed =
                List.map (fun tensor -> Ptree.P tensor) typed_params
              in
              let value, aux_value = f (rebuild packed) in
              if Option.is_none !aux then aux := Some aux_value;
              value
            in
            let value, grads = Rune.value_and_grads objective typed_inputs in
            let aux_value =
              match !aux with
              | Some value -> value
              | None ->
                  invalid_argf
                    "%s: internal error: objective did not produce auxiliary \
                     output"
                    fn_name
            in
            let grad_leaves = List.map (fun grad -> Ptree.P grad) grads in
            (value, rebuild grad_leaves, aux_value));
      }

let value_and_grad f params =
  let value, grads, () = value_and_grad_aux (fun tree -> (f tree, ())) params in
  (value, grads)

let value_and_grad_mixed f params =
  let fn_name = "Grad.value_and_grad_mixed" in
  let leaves, rebuild = Ptree.flatten params in
  if List.length leaves = 0 then (f params, rebuild [])
  else
    let path_leaves = Ptree.flatten_with_paths params in
    let leaf_count = List.length leaves in
    let path_count = List.length path_leaves in
    if leaf_count <> path_count then
      invalid_argf
        "%s: internal error: flatten/flatten_with_paths length mismatch (%d vs \
         %d)"
        fn_name leaf_count path_count;
    let leaves_array = Array.of_list leaves in
    let paths_array = Array.of_list (List.map fst path_leaves) in
    let grads_array = Array.make leaf_count None in
    let groups = Hashtbl.create 8 in
    Array.iteri
      (fun index (Ptree.P tensor) ->
        let dtype = Nx.dtype tensor in
        if not (Dtype.is_float dtype) then
          err_non_float fn_name paths_array.(index) dtype;
        let group_key = Dtype.to_string dtype in
        match Hashtbl.find_opt groups group_key with
        | None -> Hashtbl.add groups group_key (Ptree.P tensor, [ index ])
        | Some (repr, indices) ->
            Hashtbl.replace groups group_key (repr, index :: indices))
      leaves_array;
    let grouped_indices =
      Hashtbl.fold
        (fun _ (repr, indices) acc -> (repr, List.rev indices) :: acc)
        groups []
    in
    let value = ref None in
    List.iter
      (fun (repr, indices) ->
        Ptree.with_tensor repr
          {
            run =
              (fun (type a layout) (repr_tensor : (a, layout) Nx.t) ->
                let repr_dtype : (a, layout) Dtype.t = Nx.dtype repr_tensor in
                let typed_inputs =
                  List.map
                    (fun index ->
                      Ptree.with_tensor leaves_array.(index)
                        {
                          run =
                            (fun (type a2 layout2)
                              (tensor : (a2, layout2) Nx.t)
                            ->
                              let dtype = Nx.dtype tensor in
                              match Dtype.equal_witness repr_dtype dtype with
                              | Some Type.Equal -> (tensor : (a, layout) Nx.t)
                              | None ->
                                  err_mismatch fn_name paths_array.(index)
                                    repr_dtype dtype);
                        })
                    indices
                in
                let objective group_params =
                  let packed = Array.copy leaves_array in
                  List.iter2
                    (fun index tensor -> packed.(index) <- Ptree.P tensor)
                    indices group_params;
                  f (rebuild (Array.to_list packed))
                in
                let current_value, current_grads =
                  Rune.value_and_grads objective typed_inputs
                in
                if Option.is_none !value then value := Some current_value;
                List.iter2
                  (fun index grad -> grads_array.(index) <- Some (Ptree.P grad))
                  indices current_grads);
          })
      grouped_indices;
    let value =
      match !value with
      | Some v -> v
      | None ->
          invalid_argf "%s: internal error: no autodiff group produced a value"
            fn_name
    in
    let grad_leaves =
      Array.to_list
        (Array.mapi
           (fun index grad ->
             match grad with
             | Some g -> g
             | None ->
                 invalid_argf "%s: internal error: missing gradient for leaf %s"
                   fn_name
                   (path_label paths_array.(index)))
           grads_array)
    in
    (value, rebuild grad_leaves)

let grad f params = snd (value_and_grad f params)
