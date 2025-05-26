(* scheduler.ml *)

open Ir

(* ───── kernel-spec record ───── *)

type kernel_spec_t = {
  name : string;
  nodes : any_node list;
  inputs : Var.t list; (* HL vars *)
  outputs : Var.t list; (* HL vars *)
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
}

(* ───── helpers on nodes ───── *)

let get_node_input_vars (Any_Node node) : Var.t list =
  match node with
  | Placeholder _ | Const_Scalar _ | Buffer _ | Vconst _ -> []
  | Binop { a_var; b_var; _ } -> [ a_var; b_var ]
  | Unary { in_var; _ }
  | Reduce_Axis { in_var; _ }
  | Expand { in_var; _ }
  | Reshape { in_var; _ }
  | Permute { in_var; _ }
  | Pad { in_var; _ }
  | Shrink { in_var; _ }
  | Flip { in_var; _ }
  | Cast { in_var; _ }
  | Bitcast { in_var; _ }
  | Contiguous { in_var; _ }
  | Copy { in_var; _ }
  | View { in_var; _ }
  | Valid { in_var; _ }
  | Detach { in_var; _ }
  | Contiguous_Backward { in_var; _ }
  | Fuse { in_var; _ } ->
      [ in_var ]
  | Ternary { a_var; b_var; c_var; _ } -> [ a_var; b_var; c_var ]
  | Cat { in_vars; _ } -> Array.to_list in_vars
  | Vectorize { in_vars; _ } -> Array.to_list in_vars
  | Contract { in_vars; _ } -> Array.to_list in_vars
  | Assign { target_var; updates; _ } ->
      target_var :: List.map (fun (src, _, _) -> src) (Array.to_list updates)
  | Threefry { ctr_var; key_var; _ } -> [ ctr_var; key_var ]
  | Gather { src_var; indices_var; _ } -> [ src_var; indices_var ]
  | Scatter { indices_var; updates_var; _ } -> [ indices_var; updates_var ]
  | Index { in_var; idx_var; valid_var; _ } ->
      in_var :: idx_var :: (match valid_var with None -> [] | Some v -> [ v ])
  | Gep { in_var; _ } -> [ in_var ]
  | Wmma { a_var; b_var; c_var; _ } -> [ a_var; b_var; c_var ]
  | Define_Var _ | Unique _ | Device _ -> []
  | Bind { sym_var; _ } -> [ sym_var ]
  | Buffer_View { buffer_var; _ } -> [ buffer_var ]
  | Multi { device_vars; _ } -> Array.to_list device_vars
  | Unroll { loop_var; _ } -> [ loop_var ]
  | Sink { deps; _ } -> Array.to_list deps
  | Kernel { input_vars; _ } -> Array.to_list input_vars
  | Custom { in_vars; _ } -> Array.to_list in_vars
  | Noop { in_var; _ } -> ( match in_var with None -> [] | Some v -> [ v ])

let get_node_output_var (Any_Node node) : Var.t =
  match node with
  | Placeholder { out_var; _ }
  | Const_Scalar { out_var; _ }
  | Vconst { out_var; _ }
  | Buffer { out_var; _ }
  | Buffer_View { out_var; _ }
  | Binop { out_var; _ }
  | Unary { out_var; _ }
  | Ternary { out_var; _ }
  | Reduce_Axis { out_var; _ }
  | Expand { out_var; _ }
  | Reshape { out_var; _ }
  | Permute { out_var; _ }
  | Pad { out_var; _ }
  | Shrink { out_var; _ }
  | Flip { out_var; _ }
  | Cat { out_var; _ }
  | Cast { out_var; _ }
  | Bitcast { out_var; _ }
  | Contiguous { out_var; _ }
  | Copy { out_var; _ }
  | Assign { out_var; _ }
  | Threefry { out_var; _ }
  | Gather { out_var; _ }
  | Scatter { out_var; _ }
  | View { out_var; _ }
  | Valid { out_var; _ }
  | Index { out_var; _ }
  | Gep { out_var; _ }
  | Vectorize { out_var; _ }
  | Wmma { out_var; _ }
  | Define_Var { out_var; _ }
  | Bind { out_var; _ }
  | Detach { out_var; _ }
  | Contiguous_Backward { out_var; _ }
  | Multi { out_var; _ }
  | Fuse { out_var; _ }
  | Unroll { out_var; _ }
  | Contract { out_var; _ }
  | Kernel { out_var; _ }
  | Unique { out_var; _ }
  | Device { out_var; _ }
  | Custom { out_var; _ }
  | Noop { out_var; _ } ->
      out_var
  | Sink _ -> Var.fresh () (* Sink has no output var, create dummy *)

let is_boundary_node (Any_Node node) =
  match node with
  | Reduce_Axis _ | Buffer _ | Cat _ | Scatter _ | Assign _ | Wmma _ | Multi _
  | Kernel _ | Sink _ ->
      true
  | _ -> false

let is_fusible_elementwise (Any_Node node) =
  match node with
  | Binop _ | Unary _ | Ternary _ | Const_Scalar _ | Vconst _ | Expand _
  | Reshape _ | Permute _ | Placeholder _ | Pad _ | Shrink _ | Flip _ | Cast _
  | Bitcast _ | Contiguous _ | Copy _ | View _ | Valid _ | Detach _
  | Contiguous_Backward _ | Fuse _ | Noop _ ->
      true
  | _ -> false

(* ───── main scheduling pass ───── *)

let schedule (graph : Ir.graph_t) : kernel_spec_t list =
  let scheduled = ref [] in
  let current = ref [] in
  let kidx = ref 0 in

  (* Map var -> list of nodes that read it (for output detection) *)
  let var_consumers : (Var.t, any_node list) Hashtbl.t =
    Hashtbl.create (List.length graph.nodes)
  in
  List.iter
    (fun consumer ->
      List.iter
        (fun v ->
          let lst =
            Option.value ~default:[] (Hashtbl.find_opt var_consumers v)
          in
          Hashtbl.replace var_consumers v (consumer :: lst))
        (get_node_input_vars consumer))
    graph.nodes;

  let flush_current () =
    if !current <> [] then (
      let nodes = List.rev !current in

      let produced =
        List.filter
          (function
            | Ir.Any_Node (Placeholder _) | Ir.Any_Node (Sink _) -> false
            | _ -> true)
          nodes
        |> List.map get_node_output_var
        |> Var.Set.of_list
      in
      let node_inputs =
        List.concat_map get_node_input_vars nodes |> Var.Set.of_list
      in
      let inputs = Var.Set.diff node_inputs produced in
      let placeholder_inputs =
        List.filter_map
          (function
            | Ir.Any_Node (Placeholder { out_var; _ }) -> Some out_var
            | _ -> None)
          nodes
        |> Var.Set.of_list
      in
      let inputs = Var.Set.union inputs placeholder_inputs in

      (* vars metadata used inside kernel *)
      let vars_md = Hashtbl.create 16 in
      List.iter
        (fun (Any_Node n) ->
          Var.Set.iter
            (fun v ->
              match Hashtbl.find_opt graph.vars_metadata v with
              | Some m -> Hashtbl.replace vars_md v m
              | None -> ())
            (Var.Set.of_list
               (get_node_output_var (Any_Node n)
               :: get_node_input_vars (Any_Node n))))
        nodes;

      (* outputs: graph outputs OR consumed outside kernel *)
      let outputs =
        Var.Set.filter
          (fun v ->
            List.mem v graph.output_vars
            ||
            match Hashtbl.find_opt var_consumers v with
            | None -> false
            | Some readers ->
                List.exists (fun n -> not (List.memq n nodes)) readers)
          produced
        |> Var.Set.elements
      in

      scheduled :=
        {
          name = Printf.sprintf "kernel_%d" !kidx;
          nodes;
          inputs = Var.Set.elements inputs;
          outputs;
          vars_metadata = vars_md;
        }
        :: !scheduled;
      incr kidx;
      current := [])
  in

  List.iter
    (fun node ->
      let can_fuse =
        match !current with
        | [] -> true
        | _ ->
            (not (is_boundary_node node))
            && is_fusible_elementwise node
            &&
            (* check all inputs are either kernel inputs or already produced *)
            let produced =
              List.map get_node_output_var !current |> Var.Set.of_list
            in
            List.for_all
              (fun v -> List.mem v graph.input_vars || Var.Set.mem v produced)
              (get_node_input_vars node)
      in
      if can_fuse then current := node :: !current
      else (
        flush_current ();
        current := [ node ]))
    graph.nodes;

  flush_current ();
  List.rev !scheduled
