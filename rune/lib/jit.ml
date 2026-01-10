(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core
open Nx_rune
module Shape_expr = Rune_jit.Shape_expr
open Nx_buffer
module Ir = Rune_jit.Ir
module Var = Ir.Var

(* Mathematical constants *)
let ln_2 = 0.6931471805599453
let log2_e = 1.4426950408889634
let pi_over_2 = 1.5707963267948966
let shape_prod = Array.fold_left ( * ) 1

let string_of_shape arr =
  let buf = Buffer.create 16 in
  Buffer.add_char buf '[';
  Array.iteri
    (fun i v ->
      if i > 0 then Buffer.add_char buf ',';
      Buffer.add_string buf (string_of_int v))
    arr;
  Buffer.add_char buf ']';
  Buffer.contents buf

let signature_of_inputs (graph : Rune_jit.Ir.graph_t) =
  let buf = Buffer.create 64 in
  List.iteri
    (fun idx var ->
      if idx > 0 then Buffer.add_char buf ';';
      let meta = Hashtbl.find graph.vars_metadata var in
      match meta.Ir.shape_expr with
      | Some expr -> Buffer.add_string buf ("E:" ^ Shape_expr.to_string expr)
      | None -> Buffer.add_string buf ("S:" ^ string_of_shape meta.Ir.shape))
    graph.input_vars;
  Buffer.contents buf

let guard_key_of_bindings guard =
  let sorted =
    List.sort (fun (a, _, _, _) (b, _, _, _) -> Int.compare a b) guard
  in
  let buf = Buffer.create 64 in
  List.iteri
    (fun idx (id, _, _, value) ->
      if idx > 0 then Buffer.add_char buf ';';
      Buffer.add_string buf (Printf.sprintf "%d=%d" id value))
    sorted;
  Buffer.contents buf

let get_cache_table cache signature =
  match Hashtbl.find_opt cache signature with
  | Some tbl -> tbl
  | None ->
      let tbl = Hashtbl.create 4 in
      Hashtbl.add cache signature tbl;
      tbl

let bind_graph (graph : Rune_jit.Ir.graph_t) (input_shapes : int array list) :
    (int * int * int * int) list =
  let bindings : (int, Shape_expr.Var.t * int) Hashtbl.t = Hashtbl.create 16 in
  let add_binding var value =
    let id = Shape_expr.Var.id var in
    let min_v = Shape_expr.Var.min var in
    let max_v = Shape_expr.Var.max var in
    if value < min_v || value > max_v then
      invalid_arg
        (Printf.sprintf
           "Rune.jit: binding %d for var %d is outside bounds [%d, %d]" value id
           min_v max_v);
    match Hashtbl.find_opt bindings id with
    | None -> Hashtbl.add bindings id (var, value)
    | Some (_, existing) ->
        if existing <> value then
          invalid_arg
            (Printf.sprintf
               "Rune.jit: inconsistent binding for var %d (%d vs %d)" id
               existing value)
  in
  let module SE = Shape_expr in
  let rec eval_expr = function
    | SE.Const n -> Some n
    | SE.Var v -> Option.map snd (Hashtbl.find_opt bindings (SE.Var.id v))
    | SE.Add (a, b) -> (
        match (eval_expr a, eval_expr b) with
        | Some va, Some vb -> Some (va + vb)
        | _ -> None)
    | SE.Mul (a, b) -> (
        match (eval_expr a, eval_expr b) with
        | Some va, Some vb -> Some (va * vb)
        | _ -> None)
    | SE.Neg e -> Option.map (fun v -> -v) (eval_expr e)
  in
  let rec assign expr value =
    match expr with
    | SE.Const n ->
        if n <> value then
          invalid_arg
            (Printf.sprintf "Rune.jit: expected const %d, got %d" n value)
    | SE.Var v -> add_binding v value
    | SE.Add (a, b) -> (
        match (eval_expr a, eval_expr b) with
        | Some va, Some vb ->
            if va + vb <> value then
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: add expression mismatch (%d + %d <> %d)" va vb
                   value)
        | Some va, None -> assign b (value - va)
        | None, Some vb -> assign a (value - vb)
        | None, None ->
            invalid_arg
              "Rune.jit: cannot bind composite add expression with two unknowns"
        )
    | SE.Mul (a, b) -> (
        match (eval_expr a, eval_expr b) with
        | Some va, Some vb ->
            if va * vb <> value then
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: mul expression mismatch (%d * %d <> %d)" va vb
                   value)
        | Some va, None ->
            if va = 0 then
              invalid_arg "Rune.jit: ambiguous mul binding (known term is zero)";
            if value mod va <> 0 then
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: mul expression mismatch (%d does not divide %d)"
                   va value);
            assign b (value / va)
        | None, Some vb ->
            if vb = 0 then
              invalid_arg "Rune.jit: ambiguous mul binding (known term is zero)";
            if value mod vb <> 0 then
              invalid_arg
                (Printf.sprintf
                   "Rune.jit: mul expression mismatch (%d does not divide %d)"
                   vb value);
            assign a (value / vb)
        | None, None ->
            invalid_arg
              "Rune.jit: cannot bind composite mul expression with two unknowns"
        )
    | SE.Neg e -> assign e (-value)
  in
  List.iter2
    (fun var shape ->
      let meta = Hashtbl.find graph.vars_metadata var in
      (match meta.Ir.shape_expr with
      | Some exprs -> Array.iter2 assign exprs shape
      | None -> ());
      Hashtbl.replace graph.vars_metadata var { meta with Ir.shape })
    graph.input_vars input_shapes;
  let binding_list =
    Hashtbl.fold (fun id (_, value) acc -> (id, value) :: acc) bindings []
  in
  Hashtbl.iter
    (fun var meta ->
      match meta.Ir.shape_expr with
      | Some expr ->
          let evaluated = Shape_expr.eval binding_list expr in
          if Array.for_all Option.is_some evaluated then
            let ints = Array.map Option.get evaluated in
            Hashtbl.replace graph.vars_metadata var
              { meta with Ir.shape = ints }
      | None -> ())
    graph.vars_metadata;
  Hashtbl.fold
    (fun _ (var, value) acc ->
      ( Shape_expr.Var.id var,
        Shape_expr.Var.min var,
        Shape_expr.Var.max var,
        value )
      :: acc)
    bindings []
  |> List.sort (fun (a, _, _, _) (b, _, _, _) -> Int.compare a b)

let shape_info view =
  (Nx_rune.view_shape_expr view, Nx_rune.view_shape_eval view)

let concrete_shape_of_view view =
  let shape_expr, concrete_opt = shape_info view in
  match concrete_opt with
  | Some arr -> arr
  | None -> Nx_rune.shape_upper_bound shape_expr

let concrete_shape (meta : Ir.var_metadata) = meta.Ir.shape

let rec expr_is_symbolic = function
  | Shape_expr.Const _ -> false
  | Shape_expr.Var _ -> true
  | Shape_expr.Add _ -> true
  | Shape_expr.Mul _ -> true
  | Shape_expr.Neg e -> expr_is_symbolic e

let shape_expr_option exprs =
  if Array.exists expr_is_symbolic exprs then Some exprs else None

let shape_expr_or_ints (meta : Ir.var_metadata) =
  match meta.Ir.shape_expr with
  | Some expr -> expr
  | None -> Shape_expr.of_int_array meta.Ir.shape

let broadcast_shape_expr (metas : Ir.var_metadata list) out_shape =
  let res_rank = Array.length out_shape in
  if
    List.for_all
      (fun (meta : Ir.var_metadata) -> Option.is_none meta.Ir.shape_expr)
      metas
  then None
  else
    let exprs =
      Array.mapi
        (fun idx out_dim ->
          let axis_infos =
            List.filter_map
              (fun (meta : Ir.var_metadata) ->
                let shape = meta.Ir.shape in
                let rank = Array.length shape in
                let offset = res_rank - rank in
                if idx < offset then None
                else
                  let axis = idx - offset in
                  if axis < 0 || axis >= rank then None
                  else
                    let dim = shape.(axis) in
                    let expr_opt =
                      match meta.Ir.shape_expr with
                      | Some arr when axis < Array.length arr -> Some arr.(axis)
                      | _ -> None
                    in
                    Some (dim, expr_opt))
              metas
          in
          let preferred =
            List.find_map
              (fun (dim, expr_opt) ->
                if dim = out_dim && dim <> 1 then expr_opt else None)
              axis_infos
          in
          match preferred with
          | Some e -> e
          | None -> (
              match
                List.find_map (fun (_dim, expr_opt) -> expr_opt) axis_infos
              with
              | Some e -> e
              | None -> Shape_expr.const out_dim))
        out_shape
    in
    shape_expr_option exprs

let reduce_shape_expr (meta : Ir.var_metadata) axes keepdims =
  match meta.Ir.shape_expr with
  | None -> None
  | Some expr ->
      let axes_list = Array.to_list axes in
      if keepdims then
        let exprs =
          Array.mapi
            (fun i e -> if List.mem i axes_list then Shape_expr.const 1 else e)
            expr
        in
        shape_expr_option exprs
      else
        let exprs =
          expr |> Array.to_list
          |> List.mapi (fun i e -> (i, e))
          |> List.filter (fun (i, _) -> not (List.mem i axes_list))
          |> List.map snd |> Array.of_list
        in
        shape_expr_option exprs

let permute_shape_expr (meta : Ir.var_metadata) axes =
  match meta.Ir.shape_expr with
  | None -> None
  | Some expr ->
      let perm =
        Array.map
          (fun axis ->
            if axis < Array.length expr then expr.(axis) else Shape_expr.const 1)
          axes
      in
      shape_expr_option perm

let pad_shape_expr (meta : Ir.var_metadata) (padding_config : (int * int) array)
    =
  let base = shape_expr_or_ints meta in
  let exprs =
    Array.mapi
      (fun i base_expr ->
        let low, high = padding_config.(i) in
        let delta = low + high in
        if delta = 0 then base_expr
        else
          match base_expr with
          | Shape_expr.Const n -> Shape_expr.const (n + delta)
          | _ -> Shape_expr.add base_expr (Shape_expr.const delta))
      base
  in
  shape_expr_option exprs

let cat_shape_expr axis metas out_shape =
  match metas with
  | [] -> None
  | first_meta :: _ ->
      let rank = Array.length out_shape in
      let base_expr = shape_expr_or_ints first_meta in
      let exprs =
        Array.init rank (fun dim ->
            if dim = axis then
              let sum_expr =
                List.fold_left
                  (fun acc (meta : Ir.var_metadata) ->
                    let shape = meta.Ir.shape in
                    let term =
                      if axis < Array.length shape then
                        let dim_size = shape.(axis) in
                        match meta.Ir.shape_expr with
                        | Some arr when axis < Array.length arr -> arr.(axis)
                        | _ -> Shape_expr.const dim_size
                      else Shape_expr.const 1
                    in
                    match acc with
                    | None -> Some term
                    | Some expr -> Some (Shape_expr.add expr term))
                  None metas
              in
              match sum_expr with
              | Some expr -> expr
              | None -> Shape_expr.const out_shape.(axis)
            else if dim < Array.length base_expr then base_expr.(dim)
            else Shape_expr.const out_shape.(dim))
      in
      shape_expr_option exprs

let gather_shape_expr meta_data meta_indices axis _out_shape =
  match meta_data.Ir.shape_expr with
  | None -> (
      match meta_indices.Ir.shape_expr with
      | None -> None
      | Some _ ->
          let exprs = shape_expr_or_ints meta_data |> Array.copy in
          let replacement =
            match meta_indices.Ir.shape_expr with
            | Some arr when Array.length arr > 0 -> arr.(0)
            | _ ->
                let idx_shape = meta_indices.Ir.shape in
                Shape_expr.const idx_shape.(0)
          in
          if axis < Array.length exprs then exprs.(axis) <- replacement;
          shape_expr_option exprs)
  | Some data_expr ->
      let exprs = Array.copy data_expr in
      let replacement =
        match meta_indices.Ir.shape_expr with
        | Some arr when Array.length arr > 0 -> arr.(0)
        | _ ->
            let idx_shape = meta_indices.Ir.shape in
            Shape_expr.const idx_shape.(0)
      in
      if axis < Array.length exprs then exprs.(axis) <- replacement;
      shape_expr_option exprs

(* ───── Dtype Conversion Helpers ───── *)

let nx_dtype_to_ir_dtype (type a b) (nx_dt : (a, b) Dtype.t) : a Ir.Dtype.t =
  match nx_dt with
  | Nx_core.Dtype.Float32 -> Float32
  | Dtype.Int32 -> Int32
  | Dtype.UInt8 -> Uint8
  | _ ->
      failwith
        (Printf.sprintf "JIT: Unsupported dtype %s for conversion to IR"
           (Dtype.to_string nx_dt))

let nx_dtype_to_ir_any_dtype (type a b) (nx_dt : (a, b) Dtype.t) : Ir.Dtype.any
    =
  Ir.Dtype.Any_Dtype (nx_dtype_to_ir_dtype nx_dt)

(* ───── Tracing State ───── *)

type jit_tracer_state = {
  mutable recorded_nodes : Ir.any_node list;
  vars_metadata : (Var.t, Ir.var_metadata) Hashtbl.t;
  mutable input_vars_acc : Var.t list;
  symbolic_to_var : (Symbolic_id.t, Var.t) Hashtbl.t;
}

let create_state () =
  {
    recorded_nodes = [];
    vars_metadata = Hashtbl.create 32;
    input_vars_acc = [];
    symbolic_to_var = Hashtbl.create 32;
  }

let add_node state node = state.recorded_nodes <- node :: state.recorded_nodes

let record_metadata state var dtype ~shape ~shape_expr =
  Hashtbl.replace state.vars_metadata var
    {
      Ir.dtype = nx_dtype_to_ir_any_dtype dtype;
      shape;
      shape_expr;
      device = Some "CPU";
    }

let record_metadata_like state var dtype meta =
  let { Ir.shape; shape_expr; _ } = meta in
  record_metadata state var dtype ~shape ~shape_expr

let create_symbolic_tensor state out_var dtype shape =
  let id = Symbolic_id.fresh () in
  Hashtbl.add state.symbolic_to_var id out_var;
  Symbolic_tensor { id; dtype; shape }

let allocate_buffer ?shape_expr ?concrete_shape state dtype shape =
  let var = Var.fresh () in
  let ir_dtype = nx_dtype_to_ir_dtype dtype in
  add_node state
    (Ir.Any_Node
       (Ir.buffer ~dtype:ir_dtype ~size:(shape_prod shape) ~device:"CPU"
          ~out_var:var));
  let final_shape = match concrete_shape with Some c -> c | None -> shape in
  let final_shape_expr_opt =
    match shape_expr with
    | Some expr -> Some expr
    | None -> Some (Shape_expr.of_int_array final_shape)
  in
  record_metadata state var dtype ~shape:final_shape
    ~shape_expr:final_shape_expr_opt;
  (var, ir_dtype)

let get_node_output_var (Ir.Any_Node node) =
  match node with
  | Ir.Buffer { out_var; _ }
  | Ir.Const_Scalar { out_var; _ }
  | Ir.Vconst { out_var; _ }
  | Ir.Unary { out_var; _ }
  | Ir.Binop { out_var; _ }
  | Ir.Ternary { out_var; _ }
  | Ir.Reshape { out_var; _ }
  | Ir.Permute { out_var; _ }
  | Ir.Expand { out_var; _ }
  | Ir.Pad { out_var; _ }
  | Ir.Shrink { out_var; _ }
  | Ir.Reduce_Axis { out_var; _ }
  | Ir.Cast { out_var; _ }
  | Ir.Bitcast { out_var; _ }
  | Ir.View { out_var; _ }
  | Ir.Contiguous { out_var; _ }
  | Ir.Assign { out_var; _ }
  | Ir.Kernel { out_var; _ }
  | Ir.Unique { out_var; _ }
  | Ir.Device { out_var; _ }
  | Ir.Multi { out_var; _ }
  | Ir.Fuse { out_var; _ }
  | Ir.Unroll { out_var; _ }
  | Ir.Contract { out_var; _ }
  | Ir.Cat { out_var; _ }
  | Ir.Threefry { out_var; _ }
  | Ir.Gather { out_var; _ }
  | Ir.Scatter { out_var; _ }
  | Ir.Custom { out_var; _ }
  | Ir.Noop { out_var; _ }
  | Ir.Placeholder { out_var; _ }
  | Ir.Buffer_View { out_var; _ }
  | Ir.Contiguous_Backward { out_var; _ }
  | Ir.Copy { out_var; _ }
  | Ir.Detach { out_var; _ }
  | Ir.Flip { out_var; _ }
  | Ir.Gep { out_var; _ }
  | Ir.Index { out_var; _ }
  | Ir.Valid { out_var; _ }
  | Ir.Vectorize { out_var; _ }
  | Ir.Wmma { out_var; _ }
  | Ir.Bind { out_var; _ }
  | Ir.Define_Var { out_var; _ } ->
      out_var
  | Ir.Sink _ -> failwith "Sink node has no out_var"

let get_var_and_meta state tensor =
  match tensor with
  | Symbolic_tensor { id; _ } -> (
      match Hashtbl.find_opt state.symbolic_to_var id with
      | Some var ->
          let meta = Hashtbl.find state.vars_metadata var in
          (var, meta)
      | None -> failwith "JIT: Symbolic tensor not found in recorded nodes")
  | _ ->
      let var = Var.fresh () in
      let dt = dtype tensor in
      let view = view tensor in
      let shape_expr, concrete_opt = shape_info view in
      let shape =
        match concrete_opt with
        | Some arr -> arr
        | None -> Nx_rune.shape_upper_bound shape_expr
      in
      add_node state
        (Ir.Any_Node
           (Ir.Placeholder
              {
                out_var = var;
                dtype = nx_dtype_to_ir_dtype dt;
                shape = shape_expr;
              }));
      if not (List.mem var state.input_vars_acc) then
        state.input_vars_acc <- var :: state.input_vars_acc;
      record_metadata state var dt ~shape ~shape_expr:(Some shape_expr);
      let meta = Hashtbl.find state.vars_metadata var in
      (var, meta)

(* ───── Operation Handlers ───── *)

let handle_binop state op out a b =
  let var_a, _ = get_var_and_meta state a in
  let var_b, _ = get_var_and_meta state b in
  let out_var, meta_out = get_var_and_meta state out in
  let ir_dtype = nx_dtype_to_ir_dtype (dtype out) in
  add_node state
    (Ir.Any_Node
       (Ir.binary ~op ~a_var:var_a ~b_var:var_b ~out_var ~dtype:ir_dtype));
  ignore meta_out

let handle_unary state op out t_in =
  let var_in, _ = get_var_and_meta state t_in in
  let out_var, _ = get_var_and_meta state out in
  let ir_dtype = nx_dtype_to_ir_dtype (dtype out) in
  add_node state
    (Ir.Any_Node (Ir.unary ~op ~in_var:var_in ~out_var ~dtype:ir_dtype))

let reduce_shape in_shape axes keepdims =
  if keepdims then
    Array.mapi (fun i dim -> if Array.mem i axes then 1 else dim) in_shape
  else
    in_shape |> Array.to_list
    |> List.filteri (fun i _ -> not (Array.mem i axes))
    |> Array.of_list

let handle_reduction state op out t_in axes =
  let var_in, _ = get_var_and_meta state t_in in
  let out_var, _ = get_var_and_meta state out in
  let ir_dtype = nx_dtype_to_ir_dtype (dtype out) in
  add_node state
    (Ir.Any_Node
       (Ir.reduce_axis ~reduce_op_kind:op ~in_var:var_in ~axes ~out_var
          ~dtype:ir_dtype))

let bigarray_to_vconst_node (type a b) (nx_dt : (a, b) Dtype.t)
    (array : (a, b, c_layout) Array1.t) out_var =
  let numel = Array1.dim array in
  let ir_dtype = nx_dtype_to_ir_dtype nx_dt in
  let values = Array.init numel (fun i -> Array1.unsafe_get array i) in
  Ir.Any_Node (Ir.Vconst { values; out_var; dtype = ir_dtype })

(* ───── Main Effect Handler ───── *)

let make_jit_handler (state : jit_tracer_state) =
  let open Effect.Deep in
  let open Ir in
  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | E_buffer { dtype; size_in_elements; _ } ->
        Some
          (fun k ->
            let var = Var.fresh () in
            add_node state
              (Any_Node
                 (buffer
                    ~dtype:(nx_dtype_to_ir_dtype dtype)
                    ~size:size_in_elements ~device:"CPU" ~out_var:var));
            let shape = [| size_in_elements |] in
            let shape_expr = Shape_expr.of_int_array shape in
            record_metadata state var dtype ~shape ~shape_expr:(Some shape_expr);
            let symbolic_shape = Symbolic_shape.of_ints shape in
            continue k (create_symbolic_tensor state var dtype symbolic_shape))
    | E_const_scalar { value; dtype; _ } ->
        Some
          (fun k ->
            let var = Var.fresh () in
            add_node state
              (Any_Node
                 (Const_Scalar
                    { value; out_var = var; dtype = nx_dtype_to_ir_dtype dtype }));
            let shape = [||] in
            let shape_expr = Shape_expr.of_int_array shape in
            record_metadata state var dtype ~shape ~shape_expr:(Some shape_expr);
            let symbolic_shape = Symbolic_shape.of_ints [||] in
            continue k (create_symbolic_tensor state var dtype symbolic_shape))
    | E_from_host { array; _ } ->
        Some
          (fun k ->
            let kind = Array1.kind array in
            let numel = Array1.dim array in
            let var = Var.fresh () in
            (* Convert bigarray to regular OCaml array for Vconst *)
            match kind with
            | Float32 ->
                let nx_dt = Nx_core.Dtype.Float32 in
                add_node state (bigarray_to_vconst_node nx_dt array var);
                let shape = [| numel |] in
                let shape_expr = Shape_expr.of_int_array shape in
                record_metadata state var nx_dt ~shape
                  ~shape_expr:(Some shape_expr);
                let symbolic_shape = Symbolic_shape.of_ints [| numel |] in
                continue k
                  (create_symbolic_tensor state var nx_dt symbolic_shape)
            | Int32 ->
                let nx_dt = Nx_core.Dtype.Int32 in
                add_node state (bigarray_to_vconst_node nx_dt array var);
                let shape = [| numel |] in
                let shape_expr = Shape_expr.of_int_array shape in
                record_metadata state var nx_dt ~shape
                  ~shape_expr:(Some shape_expr);
                let symbolic_shape = Symbolic_shape.of_ints [| numel |] in
                continue k
                  (create_symbolic_tensor state var nx_dt symbolic_shape)
            | Int8_unsigned ->
                let nx_dt = Nx_core.Dtype.UInt8 in
                add_node state (bigarray_to_vconst_node nx_dt array var);
                let shape = [| numel |] in
                let shape_expr = Shape_expr.of_int_array shape in
                record_metadata state var nx_dt ~shape
                  ~shape_expr:(Some shape_expr);
                let symbolic_shape = Symbolic_shape.of_ints [| numel |] in
                continue k
                  (create_symbolic_tensor state var nx_dt symbolic_shape)
            | _ ->
                failwith
                  (Printf.sprintf
                     "JIT: Unsupported bigarray kind for const_array"))
    | E_add { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Add out a b;
            continue k ())
    | E_sub { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Sub out a b;
            continue k ())
    | E_mul { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Mul out a b;
            continue k ())
    | E_idiv { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Idiv out a b;
            continue k ())
    | E_fdiv { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Fdiv out a b;
            continue k ())
    | E_mod { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Mod out a b;
            continue k ())
    | E_pow { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Pow out a b;
            continue k ())
    | E_max { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Max out a b;
            continue k ())
    | E_min { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Min out a b;
            continue k ())
    | E_and { out; a; b } ->
        Some
          (fun k ->
            handle_binop state And out a b;
            continue k ())
    | E_or { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Or out a b;
            continue k ())
    | E_xor { out; a; b } ->
        Some
          (fun k ->
            handle_binop state Xor out a b;
            continue k ())
    | E_cmplt { out; a; b } ->
        Some
          (fun k ->
            let var_a, _ = get_var_and_meta state a in
            let var_b, _ = get_var_and_meta state b in
            let out_var, _ = get_var_and_meta state out in
            let ir_dtype = nx_dtype_to_ir_dtype (dtype out) in
            add_node state
              (Any_Node
                 (binary ~op:Cmplt ~a_var:var_a ~b_var:var_b ~out_var
                    ~dtype:ir_dtype));
            continue k ())
    | E_cmpne { out; a; b } ->
        Some
          (fun k ->
            let var_a, _ = get_var_and_meta state a in
            let var_b, _ = get_var_and_meta state b in
            let out_var, _ = get_var_and_meta state out in
            let ir_dtype = nx_dtype_to_ir_dtype (dtype out) in
            add_node state
              (Any_Node
                 (binary ~op:Cmpne ~a_var:var_a ~b_var:var_b ~out_var
                    ~dtype:ir_dtype));
            continue k ())
    | E_neg { out; t_in } ->
        Some
          (fun k ->
            handle_unary state Neg out t_in;
            continue k ())
    | E_sin { out; t_in } ->
        Some
          (fun k ->
            handle_unary state Sin out t_in;
            continue k ())
    | E_sqrt { out; t_in } ->
        Some
          (fun k ->
            handle_unary state Sqrt out t_in;
            continue k ())
    | E_recip { out; t_in } ->
        Some
          (fun k ->
            handle_unary state Recip out t_in;
            continue k ())
    (* Composite operations: decomposed into primitive IR ops *)
    (* These use Float32 for intermediate constants since the mathematical
       decompositions are specifically for float operations *)
    | E_log { out; t_in } ->
        (* log(x) = log2(x) * ln(2) *)
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let out_var, _ = get_var_and_meta state out in
            let shape = concrete_shape meta_in in
            let ir_dtype = Ir.Dtype.Float32 in
            (* log2(x) -> temp *)
            let log2_var, _ =
              allocate_buffer state Nx_core.Dtype.Float32 shape
            in
            add_node state
              (Any_Node
                 (unary ~op:Log2 ~in_var:var_in ~out_var:log2_var
                    ~dtype:ir_dtype));
            (* const ln(2) *)
            let ln2_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Const_Scalar
                    { value = ln_2; out_var = ln2_var; dtype = ir_dtype }));
            record_metadata state ln2_var Nx_core.Dtype.Float32 ~shape:[||]
              ~shape_expr:None;
            (* log2(x) * ln(2) -> out *)
            add_node state
              (Any_Node
                 (binary ~op:Mul ~a_var:log2_var ~b_var:ln2_var ~out_var
                    ~dtype:ir_dtype));
            continue k ())
    | E_exp { out; t_in } ->
        (* exp(x) = exp2(x * log2(e)) *)
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let out_var, _ = get_var_and_meta state out in
            let shape = concrete_shape meta_in in
            let ir_dtype = Ir.Dtype.Float32 in
            (* const log2(e) *)
            let log2e_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Const_Scalar
                    { value = log2_e; out_var = log2e_var; dtype = ir_dtype }));
            record_metadata state log2e_var Nx_core.Dtype.Float32 ~shape:[||]
              ~shape_expr:None;
            (* x * log2(e) -> temp *)
            let scaled_var, _ =
              allocate_buffer state Nx_core.Dtype.Float32 shape
            in
            add_node state
              (Any_Node
                 (binary ~op:Mul ~a_var:var_in ~b_var:log2e_var
                    ~out_var:scaled_var ~dtype:ir_dtype));
            (* exp2(temp) -> out *)
            add_node state
              (Any_Node
                 (unary ~op:Exp2 ~in_var:scaled_var ~out_var ~dtype:ir_dtype));
            continue k ())
    | E_cos { out; t_in } ->
        (* cos(x) = sin(x + π/2) *)
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let out_var, _ = get_var_and_meta state out in
            let shape = concrete_shape meta_in in
            let ir_dtype = Ir.Dtype.Float32 in
            (* const π/2 *)
            let pi2_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Const_Scalar
                    { value = pi_over_2; out_var = pi2_var; dtype = ir_dtype }));
            record_metadata state pi2_var Nx_core.Dtype.Float32 ~shape:[||]
              ~shape_expr:None;
            (* x + π/2 -> temp *)
            let shifted_var, _ =
              allocate_buffer state Nx_core.Dtype.Float32 shape
            in
            add_node state
              (Any_Node
                 (binary ~op:Add ~a_var:var_in ~b_var:pi2_var
                    ~out_var:shifted_var ~dtype:ir_dtype));
            (* sin(temp) -> out *)
            add_node state
              (Any_Node
                 (unary ~op:Sin ~in_var:shifted_var ~out_var ~dtype:ir_dtype));
            continue k ())
    | E_abs { out; t_in } ->
        (* abs(x) = where(x < 0, -x, x) *)
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let out_var, _ = get_var_and_meta state out in
            let shape = concrete_shape meta_in in
            let ir_dtype = Ir.Dtype.Float32 in
            (* const 0 *)
            let zero_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Const_Scalar
                    { value = 0.0; out_var = zero_var; dtype = ir_dtype }));
            record_metadata state zero_var Nx_core.Dtype.Float32 ~shape:[||]
              ~shape_expr:None;
            (* x < 0 -> cond *)
            let cond_var, _ = allocate_buffer state Nx_core.Dtype.UInt8 shape in
            let bool_dtype = Ir.Dtype.Bool in
            add_node state
              (Any_Node
                 (binary ~op:Cmplt ~a_var:var_in ~b_var:zero_var
                    ~out_var:cond_var ~dtype:bool_dtype));
            (* -x -> neg *)
            let neg_var, _ =
              allocate_buffer state Nx_core.Dtype.Float32 shape
            in
            add_node state
              (Any_Node
                 (unary ~op:Neg ~in_var:var_in ~out_var:neg_var ~dtype:ir_dtype));
            (* where(cond, neg, x) -> out *)
            add_node state
              (Any_Node
                 (ternary ~op:Where ~a_var:cond_var ~b_var:neg_var ~c_var:var_in
                    ~out_var ~dtype:ir_dtype));
            continue k ())
    | E_reduce_sum { out; t_in; axes; keepdims = _ } ->
        Some
          (fun k ->
            handle_reduction state Reduce_Sum out t_in axes;
            continue k ())
    | E_reduce_max { out; t_in; axes; keepdims = _ } ->
        Some
          (fun k ->
            handle_reduction state Reduce_Max out t_in axes;
            continue k ())
    | E_reduce_prod { out; t_in; axes; keepdims = _ } ->
        Some
          (fun k ->
            handle_reduction state Reduce_Prod out t_in axes;
            continue k ())
    | E_reduce_min { out; t_in; axes; keepdims } ->
        (* reduce_min(x) = -reduce_max(-x) *)
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let out_var, _ = get_var_and_meta state out in
            let in_shape = concrete_shape meta_in in
            let out_shape = reduce_shape in_shape axes keepdims in
            let ir_dtype = Ir.Dtype.Float32 in
            (* -x -> neg_in *)
            let neg_in_var, _ =
              allocate_buffer state Nx_core.Dtype.Float32 in_shape
            in
            add_node state
              (Any_Node
                 (unary ~op:Neg ~in_var:var_in ~out_var:neg_in_var
                    ~dtype:ir_dtype));
            (* reduce_max(-x) -> max_result *)
            let max_var, _ =
              allocate_buffer state Nx_core.Dtype.Float32 out_shape
            in
            add_node state
              (Any_Node
                 (reduce_axis ~reduce_op_kind:Reduce_Max ~in_var:neg_in_var
                    ~axes ~out_var:max_var ~dtype:ir_dtype));
            (* -reduce_max(-x) -> out *)
            add_node state
              (Any_Node (unary ~op:Neg ~in_var:max_var ~out_var ~dtype:ir_dtype));
            continue k ())
    | E_reshape { t_in; new_shape } ->
        Some
          (fun k ->
            let var_in, _ = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_var = Var.fresh () in
            let shape_expr = Nx_rune.shape_expr_of_symbolic new_shape in
            let shape_array =
              match Symbolic_shape.eval new_shape with
              | Some arr -> arr
              | None -> Nx_rune.shape_upper_bound shape_expr
            in
            add_node state
              (Any_Node
                 (Reshape
                    {
                      in_var = var_in;
                      new_shape = shape_expr;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            record_metadata state out_var dt ~shape:shape_array
              ~shape_expr:(Some shape_expr);
            continue k (create_symbolic_tensor state out_var dt new_shape))
    | E_expand { t_in; new_target_shape } ->
        Some
          (fun k ->
            let var_in, _ = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_var = Var.fresh () in
            let shape_expr = Nx_rune.shape_expr_of_symbolic new_target_shape in
            let shape_array =
              match Symbolic_shape.eval new_target_shape with
              | Some arr -> arr
              | None -> Nx_rune.shape_upper_bound shape_expr
            in
            add_node state
              (Any_Node
                 (Expand
                    {
                      in_var = var_in;
                      new_target_shape = shape_expr;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            record_metadata state out_var dt ~shape:shape_array
              ~shape_expr:(Some shape_expr);
            continue k
              (create_symbolic_tensor state out_var dt new_target_shape))
    | E_permute { t_in; axes } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_shape =
              let concrete = concrete_shape meta_in in
              Array.init (Array.length axes) (fun i -> concrete.(axes.(i)))
            in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Permute
                    {
                      in_var = var_in;
                      axes_permutation = axes;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            let shape_expr = permute_shape_expr meta_in axes in
            record_metadata state out_var dt ~shape:out_shape ~shape_expr;
            let symbolic_shape = Symbolic_shape.of_ints out_shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_where { out; condition; if_true; if_false } ->
        Some
          (fun k ->
            let cond_var, _ = get_var_and_meta state condition in
            let x_var, _ = get_var_and_meta state if_true in
            let y_var, _ = get_var_and_meta state if_false in
            let out_var, _ = get_var_and_meta state out in
            let ir_dtype = nx_dtype_to_ir_dtype (dtype out) in
            add_node state
              (Any_Node
                 (Ternary
                    {
                      op = Where;
                      a_var = cond_var;
                      b_var = x_var;
                      c_var = y_var;
                      out_var;
                      dtype = ir_dtype;
                    }));
            continue k ())
    | E_cast { t_in; target_dtype } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let concrete = concrete_shape meta_in in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr:meta_in.Ir.shape_expr state
                target_dtype concrete
            in
            add_node state
              (Any_Node
                 (Cast
                    {
                      in_var = var_in;
                      target_dtype = nx_dtype_to_ir_any_dtype target_dtype;
                      out_var;
                      dtype = ir_dtype;
                    }));
            let symbolic_shape = Symbolic_shape.of_ints concrete in
            continue k
              (create_symbolic_tensor state out_var target_dtype symbolic_shape))
    | E_contiguous { t_in } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let concrete = concrete_shape meta_in in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr:meta_in.Ir.shape_expr state dt
                concrete
            in
            add_node state
              (Any_Node
                 (Contiguous { in_var = var_in; out_var; dtype = ir_dtype }));
            let symbolic_shape = Symbolic_shape.of_ints concrete in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_copy { t_in } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let concrete = concrete_shape meta_in in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr:meta_in.Ir.shape_expr state dt
                concrete
            in
            add_node state
              (Any_Node
                 (Copy
                    {
                      in_var = var_in;
                      target_device = "CPU";
                      clone = true;
                      out_var;
                      dtype = ir_dtype;
                    }));
            let symbolic_shape = Symbolic_shape.of_ints concrete in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_pad { t_in; padding_config; _ } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let input_shape = concrete_shape meta_in in
            let out_shape =
              Array.mapi
                (fun i dim ->
                  let low, high = padding_config.(i) in
                  dim + low + high)
                input_shape
            in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Pad
                    {
                      in_var = var_in;
                      pad_width = padding_config;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            let shape_expr = pad_shape_expr meta_in padding_config in
            record_metadata state out_var dt ~shape:out_shape ~shape_expr;
            let symbolic_shape = Symbolic_shape.of_ints out_shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_shrink { t_in; limits } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let input_shape = concrete_shape meta_in in
            let out_shape =
              Array.mapi
                (fun i _ ->
                  let low, high = limits.(i) in
                  high - low)
                input_shape
            in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Shrink
                    {
                      in_var = var_in;
                      limits;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            let expr = Shape_expr.of_int_array out_shape in
            record_metadata state out_var dt ~shape:out_shape
              ~shape_expr:(Some expr);
            let symbolic_shape = Symbolic_shape.of_ints out_shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_flip { t_in; dims_to_flip } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let axes_to_flip =
              dims_to_flip |> Array.to_list
              |> List.mapi (fun i flip -> if flip then Some i else None)
              |> List.filter_map Fun.id |> Array.of_list
            in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Flip
                    {
                      in_var = var_in;
                      axes = axes_to_flip;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            record_metadata state out_var dt ~shape:meta_in.Ir.shape
              ~shape_expr:meta_in.Ir.shape_expr;
            let symbolic_shape = Symbolic_shape.of_ints meta_in.Ir.shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            let vars_and_metas = List.map (get_var_and_meta state) t_list in
            let in_vars = List.map fst vars_and_metas |> Array.of_list in
            let first_meta = List.hd (List.map snd vars_and_metas) in
            let dt = dtype (List.hd t_list) in
            let out_shape = Array.copy first_meta.Ir.shape in
            out_shape.(axis) <-
              List.fold_left
                (fun acc ((_, meta) : Var.t * Ir.var_metadata) ->
                  acc + meta.Ir.shape.(axis))
                0 vars_and_metas;
            let metas = List.map snd vars_and_metas in
            let shape_expr = cat_shape_expr axis metas out_shape in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr state dt out_shape
            in
            add_node state
              (Any_Node (cat ~in_vars ~axis ~out_var ~dtype:ir_dtype));
            let symbolic_shape = Symbolic_shape.of_ints out_shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_assign { dst; src } ->
        Some
          (fun k ->
            let dst_var, _ = get_var_and_meta state dst in
            let src_var, _ = get_var_and_meta state src in
            let dt = dtype dst in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Assign
                    {
                      target_var = dst_var;
                      updates = [| (src_var, dst_var, None) |];
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            continue k ())
    | E_threefry { key; ctr } ->
        Some
          (fun k ->
            let key_var, _ = get_var_and_meta state key in
            let ctr_var, meta_ctr = get_var_and_meta state ctr in
            let dt = Nx_core.Dtype.int32 in
            let shape = meta_ctr.Ir.shape in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr:meta_ctr.Ir.shape_expr state dt shape
            in
            add_node state
              (Any_Node
                 (Threefry { ctr_var; key_var; out_var; dtype = ir_dtype }));
            let symbolic_shape = Symbolic_shape.of_ints shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            let data_var, meta_data = get_var_and_meta state data in
            let indices_var, meta_indices = get_var_and_meta state indices in
            let dt = dtype data in
            let out_shape = Array.copy meta_data.Ir.shape in
            out_shape.(axis) <- meta_indices.Ir.shape.(0);
            let shape_expr =
              gather_shape_expr meta_data meta_indices axis out_shape
            in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr state dt out_shape
            in
            add_node state
              (Any_Node
                 (Gather
                    {
                      src_var = data_var;
                      indices_var;
                      axis;
                      out_var;
                      dtype = ir_dtype;
                    }));
            let symbolic_shape = Symbolic_shape.of_ints out_shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_scatter { data_template; indices; updates; axis } ->
        Some
          (fun k ->
            let _template_var, meta_template =
              get_var_and_meta state data_template
            in
            let indices_var, _ = get_var_and_meta state indices in
            let updates_var, _ = get_var_and_meta state updates in
            let dt = dtype data_template in
            let shape = meta_template.Ir.shape in
            let out_var, ir_dtype =
              allocate_buffer ?shape_expr:meta_template.Ir.shape_expr state dt
                shape
            in
            add_node state
              (Any_Node
                 (Scatter
                    {
                      indices_var;
                      updates_var;
                      axis;
                      shape;
                      out_var;
                      dtype = ir_dtype;
                    }));
            let symbolic_shape = Symbolic_shape.of_ints shape in
            continue k (create_symbolic_tensor state out_var dt symbolic_shape))
    | E_fft { t = _; axes = _ } ->
        Some
          (fun _k ->
            (* FFT operations are not supported in JIT yet *)
            failwith "JIT: FFT operations not yet supported")
    | E_ifft { t = _; axes = _ } ->
        Some
          (fun _k ->
            (* IFFT operations are not supported in JIT yet *)
            failwith "JIT: IFFT operations not yet supported")
    | E_rfft { t = _; axes = _ } ->
        Some
          (fun _k ->
            (* RFFT operations are not supported in JIT yet *)
            failwith "JIT: RFFT operations not yet supported")
    | E_irfft { t = _; axes = _; s = _ } ->
        Some
          (fun _k ->
            (* IRFFT operations are not supported in JIT yet *)
            failwith "JIT: IRFFT operations not yet supported")
    | _ -> None
  in
  { effc; retc = Fun.id; exnc = raise }

(* ───── Trace Function ───── *)

let trace _ctx f input =
  let state = create_state () in
  let handler = make_jit_handler state in
  let result = Effect.Deep.match_with f input handler in
  let output_var, _ = get_var_and_meta state result in
  let graph : Ir.graph_t =
    {
      nodes = List.rev state.recorded_nodes;
      vars_metadata = state.vars_metadata;
      input_vars = List.rev state.input_vars_acc;
      output_vars = [ output_var ];
      symbolic_vars = [];
    }
  in
  (graph, result)

(* ───── Compilation and Execution ───── *)

(* Helper to get Metal backend module *)
(* Backend selection. Default is Metal if available; LLVM is a CPU JIT fallback. *)
type jit_device = [ `metal | `llvm ]

let compile_graph (type kernel_native)
    ~(backend :
       (module Rune_jit.Backend_intf.S
          with type callable_kernel_native = kernel_native))
    (graph : Ir.graph_t) =
  match Rune_jit.compile ~backend graph with
  | Ok executable -> executable
  | Error e -> failwith (Printf.sprintf "JIT compilation failed: %s" e)

let ir_dtype_to_bigarray_kind_any (Ir.Dtype.Any_Dtype dt) =
  match dt with
  | Ir.Dtype.Float32 -> Obj.magic Float32
  | Ir.Dtype.Int32 -> Obj.magic Int32
  | Ir.Dtype.Bool -> Obj.magic Int8_unsigned
  | Ir.Dtype.Uint8 -> Obj.magic Int8_unsigned
  | Ir.Dtype.Unit -> failwith "Unit dtype has no bigarray kind"

(* ───── Compiled Function State ───── *)

type 'kernel_native compiled_state = {
  executable :
    'kernel_native Rune_jit.Backend_intf.callable_kernel Rune_jit.executable;
  input_vars : Var.t list;
  output_vars : Var.t list;
  output_shape : int array;
  output_dtype : Ir.Dtype.any;
  shape_signature : string;
  guard_key : string;
  guard : (int * int * int * int) list;
}

(* ───── Execute Compiled Function ───── *)

let execute_compiled_fn (type kernel_native)
    ~(backend :
       (module Rune_jit.Backend_intf.S
          with type callable_kernel_native = kernel_native)) state input =
  let module B =
    (val backend
        : Rune_jit.Backend_intf.S
        with type callable_kernel_native = kernel_native)
  in
  let input_ba =
    match input with
    | Native_tensor cpu_t -> Nx_c.to_host cpu_t
    | Symbolic_tensor _ -> failwith "JIT: Cannot execute with symbolic tensor"
  in

  let input_buf =
    match
      Rune_jit.allocate_buffer
        ~backend:(module B)
        ~size_in_bytes:(Array1.size_in_bytes input_ba)
        ~dtype:(nx_dtype_to_ir_dtype (dtype input))
    with
    | Ok buf -> buf
    | Error e -> failwith (Printf.sprintf "Buffer allocation failed: %s" e)
  in

  (match
     Rune_jit.copy_to_device
       ~backend:(module B)
       ~dest_buffer:input_buf ~host:input_ba
   with
  | Ok () -> ()
  | Error e -> failwith (Printf.sprintf "Copy to device failed: %s" e));

  let inputs = Hashtbl.create (List.length state.input_vars) in
  (* For operations like "add x x", multiple input vars might refer to the same
     tensor *)
  List.iter
    (fun var ->
      Hashtbl.add inputs var (Rune_jit.Backend_intf.Any_Device_Buffer input_buf))
    state.input_vars;

  let outputs =
    match
      Rune_jit.execute
        ~backend:(module B)
        state.executable ~inputs ~outputs:state.output_vars
    with
    | Ok outputs -> outputs
    | Error e -> failwith (Printf.sprintf "Execution failed: %s" e)
  in

  let (Rune_jit.Backend_intf.Any_Device_Buffer dev_buf) =
    Hashtbl.find outputs (List.hd state.output_vars)
  in

  let out_ba =
    let len = shape_prod state.output_shape in
    let kind = ir_dtype_to_bigarray_kind_any state.output_dtype in
    Array1.create kind c_layout len
  in

  (match
     B.Runtime.copy_from_device ~src_buffer:dev_buf
       ~host_dest_ptr:
         Ctypes.(raw_address_of_ptr (to_voidp (bigarray_start array1 out_ba)))
       ~device_data_offset_bytes:0
       ~copy_size_bytes:(Array1.size_in_bytes out_ba)
   with
  | Ok () -> ()
  | Error e -> failwith (Printf.sprintf "Copy from device failed: %s" e));

  (* Return a host (CPU) tensor. Outside JIT we are host-only. *)
  match input with
  | Native_tensor _ ->
      let cpu_ctx = Nx_rune.create_context () in
      Nx_rune.from_host cpu_ctx out_ba
  | Symbolic_tensor _ -> assert false

(* ───── Main JIT Function ───── *)

let jit ?(device : jit_device = `metal)
    (f : ('a, 'b) Nx_rune.t -> ('c, 'd) Nx_rune.t) =
  (* Separate caches per backend device, with concrete native types. *)
  let module M = Rune_jit_metal_or_missing in
  let module L = Rune_jit_llvm in
  let metal_cache :
      ( string,
        (string, M.callable_kernel_native compiled_state) Hashtbl.t )
      Hashtbl.t =
    Hashtbl.create 8
  in
  let llvm_cache :
      ( string,
        (string, L.callable_kernel_native compiled_state) Hashtbl.t )
      Hashtbl.t =
    Hashtbl.create 8
  in
  let get_or_compile (type kernel_native)
      ~(backend :
         (module Rune_jit.Backend_intf.S
            with type callable_kernel_native = kernel_native))
      ~(cache :
         (string, (string, kernel_native compiled_state) Hashtbl.t) Hashtbl.t)
      ~(graph : Ir.graph_t) ~(shape_signature : string) ~(guard_key : string)
      ~(guard : (int * int * int * int) list) ~(input_shapes : int array list) =
    let existing =
      match Hashtbl.find_opt cache shape_signature with
      | Some table -> Hashtbl.find_opt table guard_key
      | None -> None
    in
    match existing with
    | Some state when state.guard = guard -> state
    | _ ->
        let executable = compile_graph ~backend graph in
        let output_var =
          match graph.output_vars with
          | v :: _ -> v
          | [] -> failwith "JIT: graph has no outputs"
        in
        let out_meta = Hashtbl.find graph.vars_metadata output_var in
        let inputs_str =
          match input_shapes with
          | [] -> "(none)"
          | shapes -> String.concat ";" (List.map string_of_shape shapes)
        in
        Printf.eprintf
          "JIT: Compiling graph (signature=%s, guard=%s, inputs=%s) with %d \
           nodes\n"
          shape_signature guard_key inputs_str (List.length graph.nodes);
        let state =
          {
            executable;
            input_vars = graph.input_vars;
            output_vars = graph.output_vars;
            output_shape = out_meta.Ir.shape;
            output_dtype = out_meta.Ir.dtype;
            shape_signature;
            guard_key;
            guard;
          }
        in
        let table = get_cache_table cache shape_signature in
        Hashtbl.replace table guard_key state;
        state
  in
  let run_with_backend (type kernel_native)
      ~(backend :
         (module Rune_jit.Backend_intf.S
            with type callable_kernel_native = kernel_native))
      ~(cache :
         (string, (string, kernel_native compiled_state) Hashtbl.t) Hashtbl.t)
      ~(backend_name : string) (input : ('a, 'b) Nx_rune.t) =
    let module B =
      (val backend
          : Rune_jit.Backend_intf.S
          with type callable_kernel_native = kernel_native)
    in
    try
      let _ = B.Device_info.get_default () in
      let input_shape = concrete_shape_of_view (view input) in
      let ctx = Nx_rune.create_context () in
      let graph, _symbolic_result = trace ctx f input in
      let input_shapes = [ input_shape ] in
      let guard = bind_graph graph input_shapes in
      let shape_signature = signature_of_inputs graph in
      let guard_key = guard_key_of_bindings guard in
      let state =
        get_or_compile ~backend ~cache ~graph ~shape_signature ~guard_key ~guard
          ~input_shapes
      in
      execute_compiled_fn ~backend state input
    with e ->
      Printf.eprintf
        "JIT: Backend %s unavailable or compilation failed (%s); falling back \
         to eager\n"
        backend_name (Printexc.to_string e);
      f input
  in
  fun input ->
    match device with
    | `metal ->
        run_with_backend
          ~backend:
            (module M : Rune_jit.Backend_intf.S
              with type callable_kernel_native = M.callable_kernel_native)
          ~cache:metal_cache ~backend_name:M.name input
    | `llvm ->
        run_with_backend
          ~backend:
            (module L : Rune_jit.Backend_intf.S
              with type callable_kernel_native = L.callable_kernel_native)
          ~cache:llvm_cache ~backend_name:L.name input
