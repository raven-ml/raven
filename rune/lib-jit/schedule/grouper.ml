(* schedule/grouper.ml Tinygrad-style graph grouper: fuse elementwise ops into
   kernel clusters. *)

open Ir

(* Public cluster record (kept compatible with your old scheduler.ml) *)

type cluster_t = {
  name : string;
  nodes : any_node list;
  inputs : Var.t list; (* High-level vars feeding this cluster *)
  outputs : Var.t list; (* High-level vars produced and needed outside *)
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
}

(* Helpers on nodes *)

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
  | Sink _ ->
      (* Sink has no real SSA output; we return a fresh to keep types total.
         Sinks are excluded from 'produced' below. *)
      Var.fresh ()

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

(* Public API *)

let group (graph : Ir.graph_t) : cluster_t list =
  let scheduled = ref [] in
  let current = ref [] in
  let kidx = ref 0 in

  (* Build var -> (list of consuming nodes) to detect cross-cluster uses *)
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

  (* Precompute all placeholder outputs across the HL graph (treated as allowed
     inputs). *)
  let all_placeholder_outputs : Var.Set.t =
    List.filter_map
      (function
        | Any_Node (Placeholder { out_var; _ }) -> Some out_var | _ -> None)
      graph.nodes
    |> Var.Set.of_list
  in

  let flush_current () =
    if !current <> [] then (
      let nodes = List.rev !current in

      (* Vars produced inside this cluster (exclude Placeholder & Sink) *)
      let produced : Var.Set.t =
        List.filter
          (function
            | Any_Node (Placeholder _) | Any_Node (Sink _) -> false | _ -> true)
          nodes
        |> List.map get_node_output_var
        |> Var.Set.of_list
      in

      (* All vars read by nodes of this cluster *)
      let node_inputs : Var.Set.t =
        List.concat_map get_node_input_vars nodes |> Var.Set.of_list
      in

      (* Inputs are node_inputs minus those produced inside, plus any
         Placeholder outputs used *)
      let inputs = Var.Set.diff node_inputs produced in
      let placeholder_inputs_in_cluster : Var.Set.t =
        List.filter_map
          (function
            | Any_Node (Placeholder { out_var; _ }) -> Some out_var | _ -> None)
          nodes
        |> Var.Set.of_list
      in
      let inputs : Var.Set.t =
        Var.Set.union inputs placeholder_inputs_in_cluster
      in

      (* Vars metadata used inside kernel (copy from graph.vars_metadata) *)
      let vars_md = Hashtbl.create 16 in
      let copy_md v =
        match Hashtbl.find_opt graph.vars_metadata v with
        | Some m -> Hashtbl.replace vars_md v m
        | None -> ()
      in
      List.iter
        (fun n ->
          let outs_plus_ins = get_node_output_var n :: get_node_input_vars n in
          List.iter copy_md outs_plus_ins)
        nodes;

      (* Outputs are produced vars that are graph outputs or consumed outside
         this cluster *)
      let outputs : Var.t list =
        Var.Set.filter
          (fun v ->
            List.mem v graph.output_vars
            ||
            match Hashtbl.find_opt var_consumers v with
            | None -> false
            | Some readers ->
                (* If any reader node is not in 'nodes', it's consumed
                   outside *)
                List.exists (fun n -> not (List.memq n nodes)) readers)
          produced
        |> Var.Set.elements
      in

      let spec =
        {
          name = Printf.sprintf "kernel_%d" !kidx;
          nodes;
          inputs = Var.Set.elements inputs;
          outputs;
          vars_metadata = vars_md;
        }
      in
      scheduled := spec :: !scheduled;
      incr kidx;
      current := [])
  in

  (* Greedy linear scan: fuse maximal runs of elementwise ops, split at
     boundaries. *)
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
            let produced_inside =
              List.map get_node_output_var !current |> Var.Set.of_list
            in
            let inputs = get_node_input_vars node in
            List.for_all
              (fun v ->
                List.mem v graph.input_vars
                || Var.Set.mem v produced_inside
                || Var.Set.mem v all_placeholder_outputs)
              inputs
      in
      if can_fuse then current := node :: !current
      else (
        flush_current ();
        current := [ node ]))
    graph.nodes;

  flush_current ();
  List.rev !scheduled
