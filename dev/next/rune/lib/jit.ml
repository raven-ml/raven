(* lib/jit.ml (or lib/rune_next/jit.ml) - This file uses Rune_jit.Ir *)
open Nx_core
open Nx_rune
open Rune_jit (* For Rune_jit.Ir *)

(* Alias Var from Rune_jit.Ir for convenience *)
module Var = Ir.Var

(*** Dtype Conversion Helper ***)

(** * Converts an Nx_core.Dtype.t to the corresponding Rune_jit.Ir.Dtype.t * and
    wraps it in Rune_jit.Ir.Dtype.any. * * This function is crucial for bridging
    the Nx world with the independent Rune JIT IR. * The GADT matching ensures
    type safety during the conversion. *)
let nx_dtype_to_ir_any_dtype (type a b) (nx_dt : (a, b) Nx_core.Dtype.t) :
    Ir.Dtype.any =
  match nx_dt with
  | Dtype.Float32 -> Ir.Dtype.Any_Dtype Ir.Dtype.Float32
  | Dtype.Int32 -> Ir.Dtype.Any_Dtype Ir.Dtype.Int32
  | Dtype.UInt8 -> Ir.Dtype.Any_Dtype Ir.Dtype.Uint8
  (* Add other conversions as needed, e.g.: *)
  (* | Dtype.Float16 -> Ir.Dtype.Any_Dtype Ir.Dtype.Float16 *)
  (* | Dtype.Int64 -> Ir.Dtype.Any_Dtype Ir.Dtype.Int64 *)
  | _ ->
      failwith
        ("JIT Error: Unsupported Nx_core.Dtype encountered during conversion \
          to IR Dtype: " ^ Dtype.to_string nx_dt)

(** * A version of the converter that preserves the GADT type parameters. * This
    is useful when you need the specific ('a_elt, 'b_layout_phantom) Dtype.t *
    for constructing typed IR nodes. *)
let nx_dtype_to_ir_dtype (type a b) (nx_dt : (a, b) Nx_core.Dtype.t) :
    ('c, 'd) Ir.Dtype.t =
  match nx_dt with
  | Dtype.Float32 ->
      Obj.magic Ir.Dtype.Float32
      (* ('a, 'b) -> ('c, 'd) requires Obj.magic here *)
  | Dtype.Int32 -> Obj.magic Ir.Dtype.Int32
  | Dtype.UInt8 -> Obj.magic Ir.Dtype.Uint8
  | _ ->
      failwith
        ("JIT Error: Unsupported Nx_core.Dtype encountered during specific \
          conversion to IR Dtype: " ^ Dtype.to_string nx_dt)
(* Note on Obj.magic in nx_dtype_to_ir_dtype: The input is ('a, 'b) Dtype.t from
   Nx_core. The output is ('c, 'd) Ir.Dtype.t from Rune_jit. While conceptually
   Float32 maps to Float32, their phantom type parameters ('b and 'd) are
   different and unrelated. Obj.magic is used here to bridge this. The type
   safety relies on the correct mapping within the match cases. The 'a and 'c
   (element types) should align (e.g., float for Float32). *)

(*** JIT Tracer State and Helpers ***)

type jit_tracer_state = {
  concrete_tensor_to_var_map : (Obj.t, Var.t) Hashtbl.t;
  vars_metadata : (Var.t, Ir.var_metadata) Hashtbl.t;
      (* Uses Rune_jit.Ir.var_metadata *)
  recorded_nodes : Ir.any_node list ref; (* Uses Rune_jit.Ir.any_node *)
  mutable input_vars_acc : Var.t list;
}

let record_var_and_make_symbolic_rune_tensor (tracer_state : jit_tracer_state)
    (out_var : Var.t) (nx_dtype_specific : ('a, 'b) Dtype.t) (shape : int array)
    : ('a, 'b) Nx_rune.t =
  (* Convert Nx Dtype to IR Dtype for metadata storage *)
  let ir_any_dt = nx_dtype_to_ir_any_dtype nx_dtype_specific in
  Hashtbl.replace tracer_state.vars_metadata out_var
    { Ir.dtype = ir_any_dt; Ir.shape };

  (* Store IR Dtype in metadata *)
  let view = View.create shape in
  let symbolic_tensor : ('a, 'b) Nx_rune.t =
    {
      dtype = nx_dtype_specific;
      (* Nx_rune.t still uses Nx_core.Dtype *)
      buffer =
        Symbolic_buffer
          {
            id = out_var;
            dtype_repr = nx_dtype_specific;
            (* Store original Nx Dtype for Nx_rune *)
            shape_repr = shape;
          };
      view;
    }
  in
  symbolic_tensor

let get_jit_var_for_rune_tensor (tracer_state : jit_tracer_state)
    (tensor : ('c, 'd) Nx_rune.t) : Var.t * Ir.var_metadata =
  match Nx_rune.get_symbolic_info tensor with
  | Some (sid, _symbolic_nx_dtype, _symbolic_buffer_shape) -> (
      try
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
      let tensor_obj = Obj.repr tensor in
      match
        Hashtbl.find_opt tracer_state.concrete_tensor_to_var_map tensor_obj
      with
      | Some existing_sid ->
          let meta = Hashtbl.find tracer_state.vars_metadata existing_sid in
          (existing_sid, meta)
      | None ->
          let new_sid = Var.fresh () in
          let concrete_view = Nx_rune.view tensor in
          let concrete_shape = View.shape concrete_view in
          let concrete_nx_dtype = Nx_rune.dtype tensor in
          let concrete_ir_any_dtype =
            nx_dtype_to_ir_any_dtype concrete_nx_dtype
          in

          let meta : Ir.var_metadata =
            (* This is Rune_jit.Ir.var_metadata *)
            { Ir.dtype = concrete_ir_any_dtype; Ir.shape = concrete_shape }
          in
          (* When creating IR.Placeholder, we need the specific IR Dtype *)
          let concrete_ir_dtype : ('val_t, 'layout_t) Ir.Dtype.t =
            nx_dtype_to_ir_dtype concrete_nx_dtype
          in
          tracer_state.recorded_nodes :=
            Ir.Any_Node
              (* This is Rune_jit.Ir.Any_Node *)
              (Ir.Placeholder
                 {
                   out_var = new_sid;
                   dtype = concrete_ir_dtype;
                   (* Use specific IR Dtype *)
                   shape = concrete_shape;
                 })
            :: !(tracer_state.recorded_nodes);
          Hashtbl.replace tracer_state.vars_metadata new_sid meta;
          Hashtbl.replace tracer_state.concrete_tensor_to_var_map tensor_obj
            new_sid;
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

(*** JIT Effect Handler ***)
let make_jit_handler (tracer_state : jit_tracer_state) =
  let open Effect.Deep in
  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | E_buffer { context = _; dtype = nx_dt_specific; size_in_elements } ->
        Some
          (fun k ->
            let out_var = Var.fresh () in
            let ir_dt_specific : ('val_t, 'layout_t) Ir.Dtype.t =
              nx_dtype_to_ir_dtype
                nx_dt_specific (* Convert to specific IR Dtype *)
            in
            let node =
              Ir.Buffer { dtype = ir_dt_specific; size_in_elements; out_var }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node node :: !(tracer_state.recorded_nodes);
            let initial_shape =
              if size_in_elements = 0 then [| 0 |] else [| size_in_elements |]
            in
            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_var
                nx_dt_specific initial_shape
            in
            continue k symbolic_tensor)
    | E_const_scalar { context = _; value; dtype = nx_dt_specific } ->
        Some
          (fun k ->
            let out_var = Var.fresh () in
            let ir_dt_specific = nx_dtype_to_ir_dtype nx_dt_specific in
            let node =
              Ir.Const_Scalar
                {
                  value (* 'a matches 'val_t due to Obj.magic in converter *);
                  dtype = ir_dt_specific;
                  out_var;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node node :: !(tracer_state.recorded_nodes);
            let scalar_shape = [||] in
            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_var
                nx_dt_specific scalar_shape
            in
            continue k symbolic_tensor)
    | E_add { context = _; a; b } ->
        Some
          (fun k ->
            let var_a, meta_a = get_jit_var_for_rune_tensor tracer_state a in
            let var_b, meta_b = get_jit_var_for_rune_tensor tracer_state b in
            let res_nx_dtype = Nx_rune.dtype a in
            let res_ir_dtype : ('val_t, 'layout_t) Ir.Dtype.t =
              nx_dtype_to_ir_dtype res_nx_dtype
            in
            let res_shape =
              View.broadcast_shapes meta_a.Ir.shape meta_b.Ir.shape
            in
            let res_numel = View.prod res_shape in

            let out_buffer_var = Var.fresh () in
            tracer_state.recorded_nodes :=
              Ir.Any_Node
                (Ir.Buffer
                   {
                     dtype = res_ir_dtype;
                     size_in_elements = res_numel;
                     out_var = out_buffer_var;
                   })
              :: !(tracer_state.recorded_nodes);
            let res_ir_any_dtype = nx_dtype_to_ir_any_dtype res_nx_dtype in
            Hashtbl.replace tracer_state.vars_metadata out_buffer_var
              { Ir.dtype = res_ir_any_dtype; Ir.shape = res_shape };

            let add_node =
              Ir.Add
                {
                  in_a_var = var_a;
                  in_b_var = var_b;
                  out_var = out_buffer_var;
                  dtype = res_ir_dtype;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node add_node :: !(tracer_state.recorded_nodes);

            let symbolic_result_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state
                out_buffer_var res_nx_dtype res_shape
            in
            continue k symbolic_result_tensor)
    | E_mul { context = _; a; b } ->
        Some
          (fun k ->
            let var_a, meta_a = get_jit_var_for_rune_tensor tracer_state a in
            let var_b, meta_b = get_jit_var_for_rune_tensor tracer_state b in
            let res_nx_dtype = Nx_rune.dtype a in
            let res_ir_dtype : ('val_t, 'layout_t) Ir.Dtype.t =
              nx_dtype_to_ir_dtype res_nx_dtype
            in
            let res_shape =
              View.broadcast_shapes meta_a.Ir.shape meta_b.Ir.shape
            in
            let res_numel = View.prod res_shape in

            let out_buffer_var = Var.fresh () in
            tracer_state.recorded_nodes :=
              Ir.Any_Node
                (Ir.Buffer
                   {
                     dtype = res_ir_dtype;
                     size_in_elements = res_numel;
                     out_var = out_buffer_var;
                   })
              :: !(tracer_state.recorded_nodes);
            let res_ir_any_dtype = nx_dtype_to_ir_any_dtype res_nx_dtype in
            Hashtbl.replace tracer_state.vars_metadata out_buffer_var
              { Ir.dtype = res_ir_any_dtype; Ir.shape = res_shape };

            let mul_node =
              Ir.Mul
                {
                  in_a_var = var_a;
                  in_b_var = var_b;
                  out_var = out_buffer_var;
                  dtype = res_ir_dtype;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node mul_node :: !(tracer_state.recorded_nodes);

            let symbolic_result_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state
                out_buffer_var res_nx_dtype res_shape
            in
            continue k symbolic_result_tensor)
    | E_sum { context = _; t_in; axes; keepdims } ->
        Some
          (fun k ->
            let var_in, meta_in =
              get_jit_var_for_rune_tensor tracer_state t_in
            in
            let dtype_in_nx = Nx_rune.dtype t_in in
            let dtype_in_ir : ('val_t, 'layout_t) Ir.Dtype.t =
              nx_dtype_to_ir_dtype dtype_in_nx
            in
            let res_shape =
              calculate_reduction_output_shape meta_in.Ir.shape
                (Array.to_list axes) keepdims
            in
            let res_numel = View.prod res_shape in

            let out_buffer_var = Var.fresh () in
            tracer_state.recorded_nodes :=
              Ir.Any_Node
                (Ir.Buffer
                   {
                     dtype = dtype_in_ir;
                     size_in_elements = res_numel;
                     out_var = out_buffer_var;
                   })
              :: !(tracer_state.recorded_nodes);
            let dtype_in_ir_any = nx_dtype_to_ir_any_dtype dtype_in_nx in
            Hashtbl.replace tracer_state.vars_metadata out_buffer_var
              { Ir.dtype = dtype_in_ir_any; Ir.shape = res_shape };

            let reduce_node =
              Ir.Reduce_Axis
                {
                  in_var = var_in;
                  reduce_op_kind = Ir.Reduce_Sum;
                  axes;
                  out_var = out_buffer_var;
                  dtype = dtype_in_ir;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node reduce_node :: !(tracer_state.recorded_nodes);

            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state
                out_buffer_var dtype_in_nx res_shape
            in
            continue k symbolic_tensor)
    | E_expand { context = _; t_in; new_target_shape } ->
        Some
          (fun k ->
            let var_in, _meta_in =
              get_jit_var_for_rune_tensor tracer_state t_in
            in
            let dtype_in_nx = Nx_rune.dtype t_in in
            let dtype_in_ir : ('val_t, 'layout_t) Ir.Dtype.t =
              nx_dtype_to_ir_dtype dtype_in_nx
            in
            let out_view_var = Var.fresh () in
            let expand_node =
              Ir.Expand
                {
                  in_var = var_in;
                  new_target_shape;
                  out_var = out_view_var;
                  dtype = dtype_in_ir;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node expand_node :: !(tracer_state.recorded_nodes);

            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_view_var
                dtype_in_nx new_target_shape
            in
            continue k symbolic_tensor)
    | E_reshape { context = _; t_in; new_shape } ->
        Some
          (fun k ->
            let var_in, _meta_in =
              get_jit_var_for_rune_tensor tracer_state t_in
            in
            let dtype_in_nx = Nx_rune.dtype t_in in
            let dtype_in_ir : ('val_t, 'layout_t) Ir.Dtype.t =
              nx_dtype_to_ir_dtype dtype_in_nx
            in
            let out_view_var = Var.fresh () in
            let reshape_node =
              Ir.Reshape
                {
                  in_var = var_in;
                  new_shape;
                  out_var = out_view_var;
                  dtype = dtype_in_ir;
                }
            in
            tracer_state.recorded_nodes :=
              Ir.Any_Node reshape_node :: !(tracer_state.recorded_nodes);
            let symbolic_tensor =
              record_var_and_make_symbolic_rune_tensor tracer_state out_view_var
                dtype_in_nx new_shape
            in
            continue k symbolic_tensor)
    | _ -> None
  in
  { retc = (fun result_val -> result_val); exnc = raise; effc }

(*** Trace Function ***)
let trace (_trace_infra_context : Nx_rune.context)
    (f : ('a, 'b) Nx_rune.t -> ('c, 'd) Nx_rune.t)
    (input_argument_concrete_tensor : ('a, 'b) Nx_rune.t) :
    Ir.graph_t * ('c, 'd) Nx_rune.t =
  (* Ir.graph_t is Rune_jit.Ir.graph_t *)
  let tracer_state =
    {
      concrete_tensor_to_var_map = Hashtbl.create 16;
      vars_metadata = Hashtbl.create (32 * 2);
      recorded_nodes = ref [];
      input_vars_acc = [];
    }
  in
  let initial_sid_for_input, initial_meta_for_input =
    get_jit_var_for_rune_tensor tracer_state input_argument_concrete_tensor
  in
  let input_arg_for_f : ('a, 'b) Nx_rune.t =
    record_var_and_make_symbolic_rune_tensor tracer_state initial_sid_for_input
      (Nx_rune.dtype input_argument_concrete_tensor)
      initial_meta_for_input.Ir.shape
  in

  let handler = make_jit_handler tracer_state in
  let final_symbolic_result_tensor =
    Effect.Deep.match_with f input_arg_for_f handler
  in

  let output_var_sid, _final_result_meta =
    get_jit_var_for_rune_tensor tracer_state final_symbolic_result_tensor
  in

  let final_nodes = List.rev !(tracer_state.recorded_nodes) in
  let final_input_vars = List.rev tracer_state.input_vars_acc in

  let graph : Ir.graph_t =
    (* This is Rune_jit.Ir.graph_t *)
    {
      Ir.nodes = final_nodes;
      Ir.vars_metadata = tracer_state.vars_metadata;
      Ir.input_vars = final_input_vars;
      Ir.output_vars = [ output_var_sid ];
    }
  in
  (graph, final_symbolic_result_tensor)
