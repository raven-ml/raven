(* lib/jit.ml *)
open Nx_core
open Nx_rune
module Var = Ir.Var (* Use Var from the Ir module for consistency *)

(* The state managed by the JIT tracer. *)
type jit_tracer_state = {
  concrete_tensor_to_var_map : (Obj.t, Var.t) Hashtbl.t;
      (** Maps concrete tensor objects (if encountered) to their Var IDs. *)
  vars_metadata : (Var.t, Ir.var_metadata) Hashtbl.t;
      (** Stores metadata (dtype, shape) for each Var ID. *)
  recorded_nodes : Ir.any_node list ref;
      (** Accumulates the IR nodes in reverse execution order. *)
  mutable input_vars_acc : Var.t list;
      (** Accumulates Var IDs of tensors identified as graph inputs. *)
}

(** [record_var_and_make_symbolic_rune_tensor tracer_state out_var
     dtype_specific shape] Records metadata for [out_var] and creates a symbolic
    [Nx_rune.t] representing it. *)
let record_var_and_make_symbolic_rune_tensor (tracer_state : jit_tracer_state)
    (out_var : Var.t) (dtype_specific : ('a, 'b) Dtype.t) (shape : int array) :
    ('a, 'b) Nx_rune.t =
  Hashtbl.replace tracer_state.vars_metadata out_var
    { Ir.dtype = Ir.Any_Dtype dtype_specific; Ir.shape };
  let view = View.create shape in
  (* The symbolic tensor uses the out_var as its ID. The shape_repr in
     Symbolic_buffer should match the initial shape associated with this var. *)
  let symbolic_tensor : ('a, 'b) Nx_rune.t =
    {
      dtype = dtype_specific;
      buffer =
        Symbolic_buffer
          { id = out_var; dtype_repr = dtype_specific; shape_repr = shape };
      view;
    }
  in
  symbolic_tensor

(** [get_jit_var_for_rune_tensor tracer_state tensor] Retrieves or creates a Var
    ID and its metadata for a given [Nx_rune.t]. If the tensor is concrete and
    not yet seen, it's added as a Placeholder node. *)
let get_jit_var_for_rune_tensor (tracer_state : jit_tracer_state)
    (tensor : ('c, 'd) Nx_rune.t) : Var.t * Ir.var_metadata =
  match Nx_rune.get_symbolic_info tensor with
  | Some (sid, _symbolic_dtype, _symbolic_buffer_shape) -> (
      try
        (* The metadata for 'sid' reflects the current logical shape/dtype of
           the tensor/view that 'sid' represents. *)
        let meta = Hashtbl.find tracer_state.vars_metadata sid in
        (sid, meta)
      with Not_found ->
        failwith
          (Printf.sprintf
             "JIT Error: Symbolic tensor with ID %s encountered, but not found \
              in JIT var_metadata."
             (Var.pp Format.str_formatter sid;
              Format.flush_str_formatter ())))
  | None -> (
      (* Tensor is concrete, not yet part of the symbolic graph. *)
      let tensor_obj = Obj.repr tensor in
      match
        Hashtbl.find_opt tracer_state.concrete_tensor_to_var_map tensor_obj
      with
      | Some existing_sid ->
          (* Already encountered this concrete tensor, reuse its Var ID. *)
          let meta = Hashtbl.find tracer_state.vars_metadata existing_sid in
          (existing_sid, meta)
      | None ->
          (* New concrete tensor: create a Placeholder for it. *)
          let new_sid = Var.fresh () in
          let concrete_view = Nx_rune.view tensor in
          let concrete_shape = View.shape concrete_view in
          let concrete_dtype_specific = Nx_rune.dtype tensor in
          let meta : Ir.var_metadata =
            {
              Ir.dtype = Ir.Any_Dtype concrete_dtype_specific;
              Ir.shape = concrete_shape;
            }
          in
          tracer_state.recorded_nodes :=
            Ir.Any_Node
              (Ir.Placeholder
                 {
                   out_var = new_sid;
                   dtype = concrete_dtype_specific;
                   shape = concrete_shape;
                 })
            :: !(tracer_state.recorded_nodes);
          Hashtbl.replace tracer_state.vars_metadata new_sid meta;
          Hashtbl.replace tracer_state.concrete_tensor_to_var_map tensor_obj
            new_sid;
          (* Add to graph inputs if not already there (shouldn't be). *)
          if not (List.exists (Var.equal new_sid) tracer_state.input_vars_acc)
          then
            tracer_state.input_vars_acc <-
              new_sid :: tracer_state.input_vars_acc;
          (new_sid, meta))

(** [calculate_reduction_output_shape in_shape axes_to_reduce_list keepdims]
    Computes the shape of a tensor after a reduction operation. *)
let calculate_reduction_output_shape (in_shape : int array)
    (axes_to_reduce_list : int list) (keepdims : bool) : int array =
  let rank = Array.length in_shape in
  let axes_normalized =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf
                  "JIT calculate_reduction_output_shape: Reduction axis %d out \
                   of bounds for rank %d"
                  ax rank);
           ax')
         axes_to_reduce_list)
  in
  let out_dims = ref [] in
  if rank = 0 then
    if axes_normalized <> [] && not keepdims then
      invalid_arg
        "JIT calculate_reduction_output_shape: Cannot reduce scalar with \
         non-empty axes and keepdims=false"
    else if keepdims then [| 1 |] (* Scalar reduced with keepdims becomes [1] *)
    else [||] (* Scalar reduced without keepdims remains scalar [||] *)
  else
    let fully_reduced = ref true in
    for d = 0 to rank - 1 do
      if List.mem d axes_normalized then (
        if keepdims then out_dims := 1 :: !out_dims)
      else (
        out_dims := in_shape.(d) :: !out_dims;
        fully_reduced := false)
    done;
    let result_list = List.rev !out_dims in
    if !fully_reduced && List.length axes_normalized = rank && not keepdims then
      [||] (* Fully reduced to scalar *)
    else if result_list = [] && keepdims then
      (* E.g. input [1,1], axes=[0,1], keepdims=true -> output [1,1] *)
      Array.make rank 1
    else if result_list = [] && not keepdims then
      (* This case implies input was like [1,1] and reduced fully, or rank 0
         input *)
      [||]
    else Array.of_list result_list

(** [make_jit_handler tracer_state] Creates the effect handler for JIT tracing.
    This handler intercepts [Nx_rune] effects and translates them into
    [Ir.node_t] graph nodes. *)
let make_jit_handler (tracer_state : jit_tracer_state) =
  let open Effect.Deep in
  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | E_buffer { context = _; dtype = dt_specific; size_in_elements } ->
        Some
          (fun k ->
            let out_var = Var.fresh () in
            let node =
              Ir.Buffer { dtype = dt_specific; size_in_elements; out_var }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node node :: !(tracer_state.recorded_nodes);
            (* Buffers are initially 1D. Reshape will give them structure. *)
            let initial_shape =
              if size_in_elements = 0 then [| 0 |] else [| size_in_elements |]
            in
            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_var
                dt_specific initial_shape
            in
            continue k symbolic_tensor)
    | E_const_scalar { context = _; value; dtype = dt_specific } ->
        Some
          (fun k ->
            let out_var = Var.fresh () in
            let node =
              Ir.Const_Scalar { value; dtype = dt_specific; out_var }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node node :: !(tracer_state.recorded_nodes);
            let scalar_shape = [||] in
            (* Const_Scalar represents a 0-D tensor. *)
            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_var
                dt_specific scalar_shape
            in
            continue k symbolic_tensor)
    | E_add { context = _; a; b } ->
        Some
          (fun k ->
            let var_a, meta_a = get_jit_var_for_rune_tensor tracer_state a in
            let var_b, meta_b = get_jit_var_for_rune_tensor tracer_state b in
            (* Assume type promotion is handled, or dtypes are compatible.
               Result dtype is taken from 'a'. *)
            let res_dtype_specific = Nx_rune.dtype a in
            let res_shape = View.broadcast_shapes meta_a.shape meta_b.shape in
            let res_numel = View.prod res_shape in

            (* 1. Create a Buffer node for the output tensor's memory. *)
            let out_buffer_var = Var.fresh () in
            tracer_state.recorded_nodes :=
              Ir.Any_Node
                (Ir.Buffer
                   {
                     dtype = res_dtype_specific;
                     size_in_elements = res_numel;
                     out_var = out_buffer_var;
                   })
              :: !(tracer_state.recorded_nodes);
            (* The metadata for out_buffer_var uses the result shape. *)
            Hashtbl.replace tracer_state.vars_metadata out_buffer_var
              {
                Ir.dtype = Ir.Any_Dtype res_dtype_specific;
                Ir.shape = res_shape;
              };

            (* 2. Create the Add node, linking inputs to the output buffer
               var. *)
            let add_node =
              Ir.Add
                {
                  in_a_var = var_a;
                  in_b_var = var_b;
                  out_var = out_buffer_var;
                  dtype = res_dtype_specific;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node add_node :: !(tracer_state.recorded_nodes);

            (* 3. Create the symbolic tensor representing the result. *)
            let symbolic_result_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state
                out_buffer_var res_dtype_specific res_shape
            in
            continue k symbolic_result_tensor)
    | E_mul { context = _; a; b } ->
        Some
          (fun k ->
            let var_a, meta_a = get_jit_var_for_rune_tensor tracer_state a in
            let var_b, meta_b = get_jit_var_for_rune_tensor tracer_state b in
            let res_dtype_specific = Nx_rune.dtype a in
            let res_shape = View.broadcast_shapes meta_a.shape meta_b.shape in
            let res_numel = View.prod res_shape in

            let out_buffer_var = Var.fresh () in
            tracer_state.recorded_nodes :=
              Ir.Any_Node
                (Ir.Buffer
                   {
                     dtype = res_dtype_specific;
                     size_in_elements = res_numel;
                     out_var = out_buffer_var;
                   })
              :: !(tracer_state.recorded_nodes);
            Hashtbl.replace tracer_state.vars_metadata out_buffer_var
              {
                Ir.dtype = Ir.Any_Dtype res_dtype_specific;
                Ir.shape = res_shape;
              };

            let mul_node =
              Ir.Mul
                {
                  in_a_var = var_a;
                  in_b_var = var_b;
                  out_var = out_buffer_var;
                  dtype = res_dtype_specific;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node mul_node :: !(tracer_state.recorded_nodes);

            let symbolic_result_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state
                out_buffer_var res_dtype_specific res_shape
            in
            continue k symbolic_result_tensor)
    | E_sum { context = _; t_in; axes; keepdims } ->
        Some
          (fun k ->
            let var_in, meta_in =
              get_jit_var_for_rune_tensor tracer_state t_in
            in
            let dtype_in_specific = Nx_rune.dtype t_in in
            let res_shape =
              calculate_reduction_output_shape meta_in.shape
                (Array.to_list axes) keepdims
            in
            let res_numel = View.prod res_shape in

            let out_buffer_var = Var.fresh () in
            tracer_state.recorded_nodes :=
              Ir.Any_Node
                (Ir.Buffer
                   {
                     dtype = dtype_in_specific;
                     size_in_elements = res_numel;
                     out_var = out_buffer_var;
                   })
              :: !(tracer_state.recorded_nodes);
            Hashtbl.replace tracer_state.vars_metadata out_buffer_var
              {
                Ir.dtype = Ir.Any_Dtype dtype_in_specific;
                Ir.shape = res_shape;
              };

            let reduce_node =
              Ir.Reduce_Axis
                {
                  in_var = var_in;
                  reduce_op_kind = Ir.Reduce_Sum;
                  (* For E_sum *)
                  axes;
                  out_var = out_buffer_var;
                  dtype = dtype_in_specific;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node reduce_node :: !(tracer_state.recorded_nodes);

            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state
                out_buffer_var dtype_in_specific res_shape
            in
            continue k symbolic_tensor)
    | E_expand { context = _; t_in; new_target_shape } ->
        Some
          (fun k ->
            let var_in, _meta_in =
              get_jit_var_for_rune_tensor tracer_state t_in
            in
            let dtype_in_specific = Nx_rune.dtype t_in in
            (* Expand creates a new view, represented by a new Var. This new Var
               does not get a new Buffer node; it reuses in_var's buffer. The
               metadata for out_view_var will store the new_target_shape. *)
            let out_view_var = Var.fresh () in
            let expand_node =
              Ir.Expand
                {
                  in_var = var_in;
                  new_target_shape;
                  out_var = out_view_var;
                  dtype = dtype_in_specific;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node expand_node :: !(tracer_state.recorded_nodes);

            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_view_var
                dtype_in_specific new_target_shape
            in
            continue k symbolic_tensor)
    | E_reshape { context = _; t_in; new_shape } ->
        Some
          (fun k ->
            let var_in, _meta_in =
              get_jit_var_for_rune_tensor tracer_state t_in
            in
            let dtype_in_specific = Nx_rune.dtype t_in in
            let out_view_var = Var.fresh () in
            let reshape_node =
              Ir.Reshape
                {
                  in_var = var_in;
                  new_shape;
                  out_var = out_view_var;
                  dtype = dtype_in_specific;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node reshape_node :: !(tracer_state.recorded_nodes);
            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_view_var
                dtype_in_specific new_shape
            in
            continue k symbolic_tensor)
    | _ -> None (* For any other effects not explicitly handled. *)
  in
  { retc = (fun result_val -> result_val); exnc = raise; effc }

(** [trace _ f input_tensor] Traces the execution of function [f] with
    [input_tensor], producing an [Ir.graph_t] and the symbolic result tensor. *)
let trace (_trace_infra_context : Nx_rune.context)
    (f : ('a, 'b) Nx_rune.t -> ('c, 'd) Nx_rune.t)
    (input_argument_concrete_tensor : ('a, 'b) Nx_rune.t) :
    Ir.graph_t * ('c, 'd) Nx_rune.t =
  let tracer_state =
    {
      concrete_tensor_to_var_map = Hashtbl.create 16;
      vars_metadata = Hashtbl.create (32 * 2);
      (* Increased default size slightly *)
      recorded_nodes = ref [];
      input_vars_acc = [];
    }
  in
  (* Process the initial concrete input tensor. *)
  let initial_sid_for_input, initial_meta_for_input =
    get_jit_var_for_rune_tensor tracer_state input_argument_concrete_tensor
  in
  (* Create the initial symbolic tensor that will be passed to 'f'. This uses
     the SID of the Placeholder node created for the input. *)
  let input_arg_for_f : ('a, 'b) Nx_rune.t =
    record_var_and_make_symbolic_rune_tensor tracer_state initial_sid_for_input
      (Nx_rune.dtype input_argument_concrete_tensor)
      initial_meta_for_input.Ir.shape
  in

  let handler = make_jit_handler tracer_state in
  (* Execute the function 'f' under the JIT handler's control. *)
  let final_symbolic_result_tensor =
    Effect.Deep.match_with f input_arg_for_f handler
  in

  (* The Var ID of the final result tensor/view. *)
  let output_var_sid, _final_result_meta =
    get_jit_var_for_rune_tensor tracer_state final_symbolic_result_tensor
  in

  let final_nodes = List.rev !(tracer_state.recorded_nodes) in
  let final_input_vars = List.rev tracer_state.input_vars_acc in

  let graph : Ir.graph_t =
    {
      Ir.nodes = final_nodes;
      Ir.vars_metadata = tracer_state.vars_metadata;
      Ir.input_vars = final_input_vars;
      Ir.output_vars = [ output_var_sid ];
      (* SID of the final result var. *)
    }
  in
  (graph, final_symbolic_result_tensor)
