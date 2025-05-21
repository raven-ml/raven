open Ir

type kernel_spec_t = {
  name : string;
  nodes : any_node list;
  inputs : Var.t list; (* HL Vars *)
  outputs : Var.t list; (* HL Vars *)
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
}

let get_node_input_vars (Any_Node node) : Var.t list =
  match node with
  | Placeholder _ | Const_Scalar _ | Buffer _ -> []
  | Add { in_a_var; in_b_var; _ } | Mul { in_a_var; in_b_var; _ } ->
      [ in_a_var; in_b_var ]
  | Reduce_Axis { in_var; _ }
  | Expand { in_var; _ }
  | Reshape { in_var; _ }
  | Permute { in_var; _ } ->
      [ in_var ]

let get_node_output_var (Any_Node node) : Var.t =
  match node with
  | Placeholder { out_var; _ }
  | Const_Scalar { out_var; _ }
  | Buffer { out_var; _ }
  | Add { out_var; _ }
  | Mul { out_var; _ }
  | Reduce_Axis { out_var; _ }
  | Expand { out_var; _ }
  | Reshape { out_var; _ }
  | Permute { out_var; _ } ->
      out_var

let is_boundary_node (Any_Node node) : bool =
  match node with
  | Reduce_Axis _ -> true
  | Buffer _ ->
      true (* Assuming global buffers, not shared memory, mark boundaries *)
  | _ -> false

let is_fusible_elementwise (Any_Node node) : bool =
  match node with
  | Add _ | Mul _ | Const_Scalar _ | Expand _ | Reshape _ | Permute _ -> true
  | _ -> false

let schedule (graph : Ir.graph_t) : (kernel_spec_t list, string) result =
  let scheduled_kernels = ref [] in
  let current_kernel_nodes = ref [] in
  let kernel_idx = ref 0 in

  let var_consumers : (Var.t, any_node list) Hashtbl.t =
    Hashtbl.create (List.length graph.nodes)
  in
  List.iter
    (fun consumer_node ->
      List.iter
        (fun in_var ->
          let current_consumers =
            Hashtbl.find_opt var_consumers in_var |> Option.value ~default:[]
          in
          Hashtbl.replace var_consumers in_var
            (consumer_node :: current_consumers))
        (get_node_input_vars consumer_node))
    graph.nodes;

  let finalize_current_kernel () =
    if List.length !current_kernel_nodes > 0 then (
      let nodes_in_kernel = List.rev !current_kernel_nodes in
      let kernel_produced_vars =
        List.map get_node_output_var nodes_in_kernel |> Var.Set.of_list
      in

      let kernel_vars_metadata = Hashtbl.create 16 in
      let all_vars_in_kernel_nodes = ref Var.Set.empty in

      List.iter
        (fun node ->
          Var.Set.iter
            (fun v ->
              all_vars_in_kernel_nodes :=
                Var.Set.add v !all_vars_in_kernel_nodes)
            (Var.Set.of_list
               (get_node_output_var node :: get_node_input_vars node)))
        nodes_in_kernel;

      Var.Set.iter
        (fun v ->
          match Hashtbl.find_opt graph.vars_metadata v with
          | Some meta -> Hashtbl.replace kernel_vars_metadata v meta
          | None -> () (* Should be an error for missing metadata *))
        !all_vars_in_kernel_nodes;

      let inputs_to_kernel_nodes =
        List.concat_map get_node_input_vars nodes_in_kernel |> Var.Set.of_list
      in
      let final_kernel_inputs_set =
        Var.Set.diff inputs_to_kernel_nodes kernel_produced_vars
      in

      let potential_outputs = ref Var.Set.empty in
      Var.Set.iter
        (fun v_prod ->
          let is_graph_output = List.mem v_prod graph.output_vars in
          let consumers =
            Hashtbl.find_opt var_consumers v_prod |> Option.value ~default:[]
          in
          let consumed_outside =
            List.exists
              (fun cons_node -> not (List.memq cons_node nodes_in_kernel))
              consumers
          in
          if is_graph_output || consumed_outside then
            potential_outputs := Var.Set.add v_prod !potential_outputs)
        kernel_produced_vars;

      let spec =
        {
          name = "kernel_" ^ string_of_int !kernel_idx;
          nodes = nodes_in_kernel;
          inputs =
            Var.Set.elements final_kernel_inputs_set
            |> List.sort_uniq Var.compare;
          outputs =
            Var.Set.elements !potential_outputs |> List.sort_uniq Var.compare;
          vars_metadata = kernel_vars_metadata;
        }
      in
      scheduled_kernels := spec :: !scheduled_kernels;
      incr kernel_idx;
      current_kernel_nodes := [])
  in

  List.iter
    (fun node ->
      let can_fuse_with_current_kernel =
        match !current_kernel_nodes with
        | [] -> true
        | _prev_nodes ->
            if is_boundary_node node then false
            else if is_fusible_elementwise node then
              let current_kernel_produced_vars =
                List.map get_node_output_var !current_kernel_nodes
                |> Var.Set.of_list
              in
              List.for_all
                (fun invar ->
                  List.mem invar
                    graph.input_vars (* Global graph inputs are available *)
                  || Var.Set.mem invar current_kernel_produced_vars)
                (get_node_input_vars node)
            else false
      in
      if can_fuse_with_current_kernel then
        current_kernel_nodes := node :: !current_kernel_nodes
      else (
        finalize_current_kernel ();
        current_kernel_nodes := [ node ]))
    graph.nodes;

  finalize_current_kernel ();
  Ok (List.rev !scheduled_kernels)
