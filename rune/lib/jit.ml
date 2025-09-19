open Nx_core
open Nx_rune
open Bigarray_ext
module Ir = Rune_jit.Ir
module Var = Ir.Var
(* Backends are selected at the JIT boundary. Outside JIT we are host-only. *)

let shape_prod = Array.fold_left ( * ) 1

(* Helper to get shape from view *)
let get_shape view =
  match Symbolic_shape.eval (Lazy_view.shape view) with
  | Some arr -> arr
  | None -> failwith "Cannot evaluate symbolic shape in JIT"

(* ───── Dtype Conversion Helpers ───── *)

let nx_dtype_to_ir_dtype (type a b) (nx_dt : (a, b) Dtype.t) : a Ir.Dtype.t =
  match nx_dt with
  | Dtype.Float32 -> Float32
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

let record_metadata state var dtype shape =
  Hashtbl.add state.vars_metadata var
    { Ir.dtype = nx_dtype_to_ir_any_dtype dtype; shape; device = Some "CPU" }

let create_symbolic_tensor state out_var dtype shape =
  let id = Symbolic_id.fresh () in
  Hashtbl.add state.symbolic_to_var id out_var;
  Symbolic_tensor { id; dtype; shape }

let allocate_buffer state dtype shape =
  let var = Var.fresh () in
  let ir_dtype = nx_dtype_to_ir_dtype dtype in
  add_node state
    (Ir.Any_Node
       (Ir.buffer ~dtype:ir_dtype ~size:(shape_prod shape) ~device:"CPU"
          ~out_var:var));
  record_metadata state var dtype shape;
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
      let shape = get_shape (view tensor) in
      add_node state
        (Ir.Any_Node
           (Ir.Placeholder
              { out_var = var; dtype = nx_dtype_to_ir_dtype dt; shape }));
      if not (List.mem var state.input_vars_acc) then
        state.input_vars_acc <- var :: state.input_vars_acc;
      record_metadata state var dt shape;
      let meta = Hashtbl.find state.vars_metadata var in
      (var, meta)

(* ───── Operation Handlers ───── *)

let handle_binop state op a b =
  let var_a, meta_a = get_var_and_meta state a in
  let var_b, meta_b = get_var_and_meta state b in
  let res_shape = Shape.broadcast meta_a.shape meta_b.shape in
  let res_dtype = dtype a in
  let out_var, ir_dtype = allocate_buffer state res_dtype res_shape in
  add_node state
    (Ir.Any_Node
       (Ir.binary ~op ~a_var:var_a ~b_var:var_b ~out_var ~dtype:ir_dtype));
  create_symbolic_tensor state out_var res_dtype res_shape

let handle_unary state op t_in =
  let var_in, meta_in = get_var_and_meta state t_in in
  let shape = meta_in.shape in
  let dt = dtype t_in in
  let out_var, ir_dtype = allocate_buffer state dt shape in
  add_node state
    (Ir.Any_Node (Ir.unary ~op ~in_var:var_in ~out_var ~dtype:ir_dtype));
  create_symbolic_tensor state out_var dt shape

let reduce_shape in_shape axes keepdims =
  if keepdims then
    Array.mapi (fun i dim -> if Array.mem i axes then 1 else dim) in_shape
  else
    in_shape |> Array.to_list
    |> List.filteri (fun i _ -> not (Array.mem i axes))
    |> Array.of_list

let handle_reduction state op t_in axes keepdims =
  let var_in, meta_in = get_var_and_meta state t_in in
  let out_shape = reduce_shape meta_in.shape axes keepdims in
  let dt = dtype t_in in
  let out_var, ir_dtype = allocate_buffer state dt out_shape in
  add_node state
    (Ir.Any_Node
       (Ir.reduce_axis ~reduce_op_kind:op ~in_var:var_in ~axes ~out_var
          ~dtype:ir_dtype));
  create_symbolic_tensor state out_var dt out_shape

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
            record_metadata state var dtype shape;
            continue k (create_symbolic_tensor state var dtype shape))
    | E_const_scalar { value; dtype; _ } ->
        Some
          (fun k ->
            let var = Var.fresh () in
            add_node state
              (Any_Node
                 (Const_Scalar
                    { value; out_var = var; dtype = nx_dtype_to_ir_dtype dtype }));
            record_metadata state var dtype [||];
            continue k (create_symbolic_tensor state var dtype [||]))
    | E_const_array { array; _ } ->
        Some
          (fun k ->
            let numel = Array1.dim array in
            let nx_dtype =
              Nx_core.Dtype.of_bigarray_ext_kind (Array1.kind array)
            in
            let var = Var.fresh () in
            add_node state
              (Any_Node
                 (Placeholder
                    {
                      out_var = var;
                      dtype = nx_dtype_to_ir_dtype nx_dtype;
                      shape = [| numel |];
                    }));
            if not (List.mem var state.input_vars_acc) then
              state.input_vars_acc <- var :: state.input_vars_acc;
            record_metadata state var nx_dtype [| numel |];
            continue k (create_symbolic_tensor state var nx_dtype [| numel |]))
    | E_add { a; b } -> Some (fun k -> continue k (handle_binop state Add a b))
    | E_mul { a; b } -> Some (fun k -> continue k (handle_binop state Mul a b))
    | E_idiv { a; b } ->
        Some (fun k -> continue k (handle_binop state Idiv a b))
    | E_fdiv { a; b } ->
        Some (fun k -> continue k (handle_binop state Fdiv a b))
    | E_mod { a; b } -> Some (fun k -> continue k (handle_binop state Mod a b))
    | E_pow { a; b } -> Some (fun k -> continue k (handle_binop state Pow a b))
    | E_max { a; b } -> Some (fun k -> continue k (handle_binop state Max a b))
    | E_and { a; b } -> Some (fun k -> continue k (handle_binop state And a b))
    | E_or { a; b } -> Some (fun k -> continue k (handle_binop state Or a b))
    | E_xor { a; b } -> Some (fun k -> continue k (handle_binop state Xor a b))
    | E_cmplt { a; b } ->
        Some
          (fun k ->
            let var_a, meta_a = get_var_and_meta state a in
            let var_b, meta_b = get_var_and_meta state b in
            let res_shape = Shape.broadcast meta_a.shape meta_b.shape in
            let res_dtype = Nx_core.Dtype.uint8 in
            let out_var, ir_dtype = allocate_buffer state res_dtype res_shape in
            add_node state
              (Any_Node
                 (binary ~op:Cmplt ~a_var:var_a ~b_var:var_b ~out_var
                    ~dtype:ir_dtype));
            continue k
              (create_symbolic_tensor state out_var res_dtype res_shape))
    | E_cmpne { a; b } ->
        Some
          (fun k ->
            let var_a, meta_a = get_var_and_meta state a in
            let var_b, meta_b = get_var_and_meta state b in
            let res_shape = Shape.broadcast meta_a.shape meta_b.shape in
            let res_dtype = Nx_core.Dtype.uint8 in
            let out_var, ir_dtype = allocate_buffer state res_dtype res_shape in
            add_node state
              (Any_Node
                 (binary ~op:Cmpne ~a_var:var_a ~b_var:var_b ~out_var
                    ~dtype:ir_dtype));
            continue k
              (create_symbolic_tensor state out_var res_dtype res_shape))
    | E_neg { t_in } -> Some (fun k -> continue k (handle_unary state Neg t_in))
    | E_log2 { t_in } ->
        Some (fun k -> continue k (handle_unary state Log2 t_in))
    | E_exp2 { t_in } ->
        Some (fun k -> continue k (handle_unary state Exp2 t_in))
    | E_sin { t_in } -> Some (fun k -> continue k (handle_unary state Sin t_in))
    | E_sqrt { t_in } ->
        Some (fun k -> continue k (handle_unary state Sqrt t_in))
    | E_recip { t_in } ->
        Some (fun k -> continue k (handle_unary state Recip t_in))
    | E_reduce_sum { t_in; axes; keepdims } ->
        Some
          (fun k ->
            continue k (handle_reduction state Reduce_Sum t_in axes keepdims))
    | E_reduce_max { t_in; axes; keepdims } ->
        Some
          (fun k ->
            continue k (handle_reduction state Reduce_Max t_in axes keepdims))
    | E_reduce_prod { t_in; axes; keepdims } ->
        Some
          (fun k ->
            continue k (handle_reduction state Reduce_Prod t_in axes keepdims))
    | E_reshape { t_in; new_shape } ->
        Some
          (fun k ->
            let var_in, _ = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Reshape
                    {
                      in_var = var_in;
                      new_shape;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            record_metadata state out_var dt new_shape;
            continue k (create_symbolic_tensor state out_var dt new_shape))
    | E_expand { t_in; new_target_shape } ->
        Some
          (fun k ->
            let var_in, _ = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_var = Var.fresh () in
            add_node state
              (Any_Node
                 (Expand
                    {
                      in_var = var_in;
                      new_target_shape;
                      out_var;
                      dtype = nx_dtype_to_ir_dtype dt;
                    }));
            record_metadata state out_var dt new_target_shape;
            continue k
              (create_symbolic_tensor state out_var dt new_target_shape))
    | E_permute { t_in; axes } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_shape =
              Array.init (Array.length axes) (fun i -> meta_in.shape.(axes.(i)))
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
            record_metadata state out_var dt out_shape;
            continue k (create_symbolic_tensor state out_var dt out_shape))
    | E_where { condition; if_true; if_false } ->
        Some
          (fun k ->
            let cond_var, meta_cond = get_var_and_meta state condition in
            let x_var, meta_x = get_var_and_meta state if_true in
            let y_var, meta_y = get_var_and_meta state if_false in
            let res_dtype = dtype if_true in
            let res_shape =
              Shape.broadcast
                (Shape.broadcast meta_cond.shape meta_x.shape)
                meta_y.shape
            in
            let out_var, ir_dtype = allocate_buffer state res_dtype res_shape in
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
            continue k
              (create_symbolic_tensor state out_var res_dtype res_shape))
    | E_cast { t_in; target_dtype } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let shape = meta_in.shape in
            let out_var, ir_dtype = allocate_buffer state target_dtype shape in
            add_node state
              (Any_Node
                 (Cast
                    {
                      in_var = var_in;
                      target_dtype = nx_dtype_to_ir_any_dtype target_dtype;
                      out_var;
                      dtype = ir_dtype;
                    }));
            continue k (create_symbolic_tensor state out_var target_dtype shape))
    | E_contiguous { t_in } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let shape = meta_in.shape in
            let out_var, ir_dtype = allocate_buffer state dt shape in
            add_node state
              (Any_Node
                 (Contiguous { in_var = var_in; out_var; dtype = ir_dtype }));
            continue k (create_symbolic_tensor state out_var dt shape))
    | E_copy { t_in } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let shape = meta_in.shape in
            let out_var, ir_dtype = allocate_buffer state dt shape in
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
            continue k (create_symbolic_tensor state out_var dt shape))
    | E_pad { t_in; padding_config; _ } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_shape =
              Array.mapi
                (fun i dim ->
                  let low, high = padding_config.(i) in
                  dim + low + high)
                meta_in.shape
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
            record_metadata state out_var dt out_shape;
            continue k (create_symbolic_tensor state out_var dt out_shape))
    | E_shrink { t_in; limits } ->
        Some
          (fun k ->
            let var_in, meta_in = get_var_and_meta state t_in in
            let dt = dtype t_in in
            let out_shape =
              Array.mapi
                (fun i _ ->
                  let low, high = limits.(i) in
                  high - low)
                meta_in.shape
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
            record_metadata state out_var dt out_shape;
            continue k (create_symbolic_tensor state out_var dt out_shape))
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
            record_metadata state out_var dt meta_in.shape;
            continue k (create_symbolic_tensor state out_var dt meta_in.shape))
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            let vars_and_metas = List.map (get_var_and_meta state) t_list in
            let in_vars = List.map fst vars_and_metas |> Array.of_list in
            let first_meta = List.hd (List.map snd vars_and_metas) in
            let dt = dtype (List.hd t_list) in
            let out_shape = Array.copy first_meta.shape in
            out_shape.(axis) <-
              List.fold_left
                (fun acc ((_, meta) : Var.t * var_metadata) ->
                  acc + meta.shape.(axis))
                0 vars_and_metas;
            let out_var, ir_dtype = allocate_buffer state dt out_shape in
            add_node state
              (Any_Node (cat ~in_vars ~axis ~out_var ~dtype:ir_dtype));
            continue k (create_symbolic_tensor state out_var dt out_shape))
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
            let shape = meta_ctr.shape in
            let out_var, ir_dtype = allocate_buffer state dt shape in
            add_node state
              (Any_Node
                 (Threefry { ctr_var; key_var; out_var; dtype = ir_dtype }));
            continue k (create_symbolic_tensor state out_var dt shape))
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            let data_var, meta_data = get_var_and_meta state data in
            let indices_var, meta_indices = get_var_and_meta state indices in
            let dt = dtype data in
            let out_shape = Array.copy meta_data.shape in
            out_shape.(axis) <- meta_indices.shape.(0);
            let out_var, ir_dtype = allocate_buffer state dt out_shape in
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
            continue k (create_symbolic_tensor state out_var dt out_shape))
    | E_scatter { data_template; indices; updates; axis } ->
        Some
          (fun k ->
            let _template_var, meta_template =
              get_var_and_meta state data_template
            in
            let indices_var, _ = get_var_and_meta state indices in
            let updates_var, _ = get_var_and_meta state updates in
            let dt = dtype data_template in
            let shape = meta_template.shape in
            let out_var, ir_dtype = allocate_buffer state dt shape in
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
            continue k (create_symbolic_tensor state out_var dt shape))
    | E_fft { t = _; axes = _; s = _ } ->
        Some
          (fun _k ->
            (* FFT operations are not supported in JIT yet *)
            failwith "JIT: FFT operations not yet supported")
    | E_ifft { t = _; axes = _; s = _ } ->
        Some
          (fun _k ->
            (* IFFT operations are not supported in JIT yet *)
            failwith "JIT: IFFT operations not yet supported")
    | E_rfft { t = _; axes = _; s = _ } ->
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

let backend_module (device : jit_device) =
  match device with
  | `metal ->
      let module M = Rune_jit_metal_or_missing in
      (module M : Rune_jit.Backend_intf.S)
  | `llvm ->
      let module L = Rune_jit_llvm in
      (module L : Rune_jit.Backend_intf.S)

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
    | Native_tensor cpu_t -> Nx_c.data cpu_t
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
      Nx_rune.op_const_array cpu_ctx out_ba
  | Symbolic_tensor _ -> assert false

(* ───── Main JIT Function ───── *)

let jit ?(device : jit_device = `metal)
    (f : ('a, 'b) Nx_rune.t -> ('c, 'd) Nx_rune.t) =
  (* Separate caches per backend device, with concrete native types. *)
  let metal_cache =
    let module M = Rune_jit_metal_or_missing in
    (Hashtbl.create 8
      : (int array, M.callable_kernel_native compiled_state) Hashtbl.t)
  in
  let llvm_cache =
    let module L = Rune_jit_llvm in
    (Hashtbl.create 8
      : (int array, L.callable_kernel_native compiled_state) Hashtbl.t)
  in

  fun (input : ('a, 'b) Nx_rune.t) ->
    (* All inputs are expected to be CPU tensors outside JIT. *)
    (match input with
    | Native_tensor _ -> ()
    | Symbolic_tensor _ -> failwith "JIT: Cannot execute with symbolic tensor");

    let input_shape = get_shape (view input) in
    match device with
    | `metal ->
        let module M = Rune_jit_metal_or_missing in
        let backend =
          (module M : Rune_jit.Backend_intf.S
            with type callable_kernel_native = M.callable_kernel_native)
        in
        begin
          match Hashtbl.find_opt metal_cache input_shape with
          | Some state -> execute_compiled_fn ~backend state input
          | None -> (
              try
                let _ = M.Device_info.get_default () in
                let ctx = Nx_rune.create_context () in
                let graph, symbolic_result = trace ctx f input in
                Printf.eprintf
                  "JIT: Compiling graph for shape %s with %d nodes\n"
                  (Array.fold_left
                     (fun acc x -> acc ^ " " ^ string_of_int x)
                     "[" input_shape
                  ^ " ]")
                  (List.length graph.nodes);
                let executable = compile_graph ~backend graph in
                let state : M.callable_kernel_native compiled_state =
                  {
                    executable;
                    input_vars = graph.input_vars;
                    output_vars = graph.output_vars;
                    output_shape = get_shape (view symbolic_result);
                    output_dtype =
                      nx_dtype_to_ir_any_dtype (dtype symbolic_result);
                  }
                in
                Hashtbl.add metal_cache input_shape state;
                execute_compiled_fn ~backend state input
              with e ->
                Printf.eprintf
                  "JIT: Backend %s unavailable or compilation failed (%s); falling back to eager\n"
                  M.name (Printexc.to_string e);
                f input)
        end
    | `llvm ->
        let module L = Rune_jit_llvm in
        let backend =
          (module L : Rune_jit.Backend_intf.S
            with type callable_kernel_native = L.callable_kernel_native)
        in
        begin
          match Hashtbl.find_opt llvm_cache input_shape with
          | Some state -> execute_compiled_fn ~backend state input
          | None -> (
              try
                let _ = L.Device_info.get_default () in
                let ctx = Nx_rune.create_context () in
                let graph, symbolic_result = trace ctx f input in
                Printf.eprintf
                  "JIT: Compiling graph for shape %s with %d nodes\n"
                  (Array.fold_left
                     (fun acc x -> acc ^ " " ^ string_of_int x)
                     "[" input_shape
                  ^ " ]")
                  (List.length graph.nodes);
                let executable = compile_graph ~backend graph in
                let state : L.callable_kernel_native compiled_state =
                  {
                    executable;
                    input_vars = graph.input_vars;
                    output_vars = graph.output_vars;
                    output_shape = get_shape (view symbolic_result);
                    output_dtype =
                      nx_dtype_to_ir_any_dtype (dtype symbolic_result);
                  }
                in
                Hashtbl.add llvm_cache input_shape state;
                execute_compiled_fn ~backend state input
              with e ->
                Printf.eprintf
                  "JIT: Backend %s unavailable or compilation failed (%s); falling back to eager\n"
                  L.name (Printexc.to_string e);
                f input)
        end
