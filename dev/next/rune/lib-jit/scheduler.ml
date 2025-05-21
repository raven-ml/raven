(* scheduler.ml *)

open Ir

(* ────────── kernel-spec record ────────── *)

type kernel_spec_t = {
  name : string;
  nodes : any_node list;
  inputs : Var.t list; (* HL vars *)
  outputs : Var.t list; (* HL vars *)
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
}

(* ────────── helpers on nodes ────────── *)

let get_node_input_vars (Any_Node node) : Var.t list =
  match node with
  | Placeholder _ | Const_Scalar _ | Buffer _ -> []
  | Binop { a_var; b_var; _ } -> [ a_var; b_var ]
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
  | Binop { out_var; _ }
  | Reduce_Axis { out_var; _ }
  | Expand { out_var; _ }
  | Reshape { out_var; _ }
  | Permute { out_var; _ } ->
      out_var

let is_boundary_node (Any_Node node) =
  match node with Reduce_Axis _ | Buffer _ -> true | _ -> false

let is_fusible_elementwise (Any_Node node) =
  match node with
  | Binop _ | Const_Scalar _ | Expand _ | Reshape _ | Permute _ -> true
  | _ -> false

(* ────────── main scheduling pass ────────── *)

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

      let produced = List.map get_node_output_var nodes |> Var.Set.of_list in
      let inputs =
        List.concat_map get_node_input_vars nodes
        |> Var.Set.of_list |> Var.Set.diff produced
      in

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
