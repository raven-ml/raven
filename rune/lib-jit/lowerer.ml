(* lowerer.ml *)

open Ir

(* ───── helpers ───── *)

let const_to_string : type a. a Dtype.t -> a -> string =
 fun dt v ->
  match dt with
  | Dtype.Float32 -> Printf.sprintf "%gf" v
  | Dtype.Int32 -> Int32.to_string v
  | Dtype.Uint8 -> string_of_int v
  | Dtype.Bool -> if v then "true" else "false"
  | Dtype.Unit -> "0"

(* ───── lowering context ───── *)

type lowering_ctx = {
  meta : (Var.t, var_metadata) Hashtbl.t; (* one shared table *)
  scalar_map : (Var.t, Var.t) Hashtbl.t; (* HL scalar → LL scalar *)
  buffer_map : (Var.t, Var.t) Hashtbl.t; (* HL buffer → LL buffer *)
  instrs : Lowered.instruction list ref;
}

let new_ctx (graph_meta : (Var.t, var_metadata) Hashtbl.t)
    (kernel_meta : (Var.t, var_metadata) Hashtbl.t) =
  (* meta starts as a shallow copy of both sources *)
  let meta = Hashtbl.copy graph_meta in
  Hashtbl.iter (Hashtbl.replace meta) kernel_meta;
  {
    meta;
    scalar_map = Hashtbl.create 16;
    buffer_map = Hashtbl.create 16;
    instrs = ref [];
  }

let add_instr ctx i = ctx.instrs := i :: !(ctx.instrs)
let ensure_meta ctx v meta = Hashtbl.replace ctx.meta v meta
let meta_of ctx v = Hashtbl.find_opt ctx.meta v

(* ───── mapping helpers ───── *)

let ll_of_hl ctx hl ~buffer =
  let tbl = if buffer then ctx.buffer_map else ctx.scalar_map in
  match Hashtbl.find_opt tbl hl with
  | Some ll -> ll
  | None ->
      let ll = Var.fresh () in
      Hashtbl.add tbl hl ll;
      Option.iter (ensure_meta ctx ll) (meta_of ctx hl);
      ll

(* ───── frequently used snippets ───── *)

let gtid ctx =
  let v = Var.fresh () in
  ensure_meta ctx v
    { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |]; device = None };
  add_instr ctx
    (Lowered.L_Special { dst = v; kind = Special_index_kind.Global_task_idx 0 });
  v

let load_scalar ctx ~hl_buffer ~idx ~dtype =
  let buf = ll_of_hl ctx hl_buffer ~buffer:true in
  let dst = Var.fresh () in
  ensure_meta ctx dst
    { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
  add_instr ctx
    (Lowered.L_Load
       { dst; buf; idx; dtype = Dtype.Any_Dtype dtype; valid = None });
  dst

(* ───── node lowering ───── *)

let lower_node ctx (Any_Node n) kernel_outs =
  let open Lowered in
  match n with
  | Placeholder { out_var; _ } ->
      ignore (ll_of_hl ctx out_var ~buffer:true);
      Ok ()
  | Buffer { dtype; size_in_elements; device = _; out_var } ->
      let ll = ll_of_hl ctx out_var ~buffer:true in
      add_instr ctx
        (L_Buffer
           { dtype = Dtype.Any_Dtype dtype; size = size_in_elements; out = ll });
      Ok ()
  | Const_Scalar { value; dtype; out_var } ->
      let ll = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx
        (L_Const
           {
             dtype = Dtype.Any_Dtype dtype;
             value = const_to_string dtype value;
             out = ll;
           });
      Ok ()
  | Binop { op; a_var; b_var; out_var; dtype } ->
      let idx = gtid ctx in
      let a_ll = load_scalar ctx ~hl_buffer:a_var ~idx ~dtype in
      let b_ll = load_scalar ctx ~hl_buffer:b_var ~idx ~dtype in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_ALU
           {
             dst;
             op = Lowered.Binary op;
             args = [ a_ll; b_ll ];
             dtype = Dtype.Any_Dtype dtype;
           });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | Reduce_Axis { in_var; reduce_op_kind; out_var; dtype; _ } ->
      (* look up shape once *)
      let shape =
        match meta_of ctx in_var with
        | Some m -> m.shape
        | None ->
            failwith
              ("No metadata for var " ^ Var.to_string in_var ^ " during Reduce")
      in
      let total = Array.fold_left ( * ) 1 shape in
      let idx = Var.fresh () in
      ensure_meta ctx idx
        { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |]; device = None };

      let ub = Var.fresh () in
      ensure_meta ctx ub
        { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_Const
           {
             dtype = Dtype.Any_Dtype Dtype.Int32;
             value = string_of_int total;
             out = ub;
           });
      add_instr ctx (L_Range { idx; bound = ub });

      let acc = Var.fresh () in
      ensure_meta ctx acc
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      let identity =
        match (dtype, reduce_op_kind) with
        | Dtype.Float32, Reduce_Sum -> "0.0"
        | Dtype.Float32, Reduce_Max -> "-INFINITY"
        | Dtype.Float32, Reduce_Prod -> "1.0"
        | Dtype.Int32, Reduce_Sum -> "0"
        | Dtype.Int32, Reduce_Max -> string_of_int min_int
        | Dtype.Int32, Reduce_Prod -> "1"
        | Dtype.Uint8, Reduce_Sum -> "0"
        | Dtype.Uint8, Reduce_Max -> "0"
        | Dtype.Uint8, Reduce_Prod -> "1"
        | Dtype.Bool, Reduce_Sum -> "false"
        | Dtype.Bool, Reduce_Max -> "true"
        | Dtype.Bool, Reduce_Prod -> "true"
        | Dtype.Unit, _ -> "0"
      in
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = identity; out = acc });

      let cur = load_scalar ctx ~hl_buffer:in_var ~idx ~dtype in
      let op =
        match reduce_op_kind with
        | Reduce_Sum -> Lowered.Binary Add
        | Reduce_Max -> Lowered.Binary Max
        | Reduce_Prod -> Lowered.Binary Mul
      in
      add_instr ctx
        (L_ALU
           { dst = acc; op; args = [ acc; cur ]; dtype = Dtype.Any_Dtype dtype });
      add_instr ctx L_EndRange;

      if List.mem out_var kernel_outs then (
        let ob = ll_of_hl ctx out_var ~buffer:true in
        let z = Var.fresh () in
        ensure_meta ctx z
          {
            dtype = Dtype.Any_Dtype Dtype.Int32;
            shape = [| 1 |];
            device = None;
          };
        add_instr ctx
          (L_Const { dtype = Dtype.Any_Dtype Dtype.Int32; value = "0"; out = z });
        add_instr ctx (L_Store { buf = ob; idx = z; src = acc; valid = None }))
      else Hashtbl.replace ctx.scalar_map out_var acc;
      Ok ()
  | Unary { op; in_var; out_var; dtype } ->
      let idx = gtid ctx in
      let in_ll = load_scalar ctx ~hl_buffer:in_var ~idx ~dtype in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_ALU
           {
             dst;
             op = Lowered.Unary op;
             args = [ in_ll ];
             dtype = Dtype.Any_Dtype dtype;
           });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | Ternary
      {
        op = Where;
        a_var = cond_var;
        b_var = x_var;
        c_var = y_var;
        out_var;
        dtype;
      } ->
      let idx = gtid ctx in
      let cond_ll =
        load_scalar ctx ~hl_buffer:cond_var ~idx ~dtype:Dtype.Bool
      in
      let x_ll = load_scalar ctx ~hl_buffer:x_var ~idx ~dtype in
      let y_ll = load_scalar ctx ~hl_buffer:y_var ~idx ~dtype in
      (* For now, simple implementation using compare and multiply *)
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      (* This is a simplification - a real implementation would need a ternary
         op *)
      add_instr ctx
        (L_ALU
           {
             dst;
             op = Lowered.Ternary Where;
             args = [ cond_ll; x_ll; y_ll ];
             dtype = Dtype.Any_Dtype dtype;
           });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | Expand { in_var; out_var; _ }
  | Reshape { in_var; out_var; _ }
  | Permute { in_var; out_var; _ }
  | Pad { in_var; out_var; _ }
  | Shrink { in_var; out_var; _ }
  | Flip { in_var; out_var; _ }
  | Contiguous { in_var; out_var; _ }
  | Copy { in_var; target_device = _; out_var; _ } ->
      (match Hashtbl.find_opt ctx.scalar_map in_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx in_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Cast { in_var; target_dtype = _; out_var; dtype = _ } ->
      (* For now, treat cast as a view operation *)
      (match Hashtbl.find_opt ctx.scalar_map in_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx in_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Cat { in_vars; axis = _; out_var; dtype = _ } ->
      (* Simplified cat - just map to first input for now *)
      (if Array.length in_vars > 0 then
         match Hashtbl.find_opt ctx.scalar_map in_vars.(0) with
         | Some s -> Hashtbl.replace ctx.scalar_map out_var s
         | None ->
             let b = ll_of_hl ctx in_vars.(0) ~buffer:true in
             Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Assign { target_var; updates = _; out_var; dtype = _ } ->
      (* Simplified assign - just map to target for now *)
      (match Hashtbl.find_opt ctx.scalar_map target_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx target_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Threefry { ctr_var = _; key_var = _; out_var; dtype } ->
      (* Simplified threefry - needs proper implementation *)
      let idx = gtid ctx in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = "0"; out = dst });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | Gather { src_var; indices_var = _; axis = _; out_var; dtype = _ } ->
      (* Simplified gather - just map to source for now *)
      (match Hashtbl.find_opt ctx.scalar_map src_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx src_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Scatter
      { indices_var = _; updates_var = _; axis = _; shape = _; out_var; dtype }
    ->
      (* Simplified scatter - needs proper implementation *)
      let idx = gtid ctx in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = "0"; out = dst });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | Ternary { op = Mulacc; a_var; b_var; c_var; out_var; dtype } ->
      let idx = gtid ctx in
      let a_ll = load_scalar ctx ~hl_buffer:a_var ~idx ~dtype in
      let b_ll = load_scalar ctx ~hl_buffer:b_var ~idx ~dtype in
      let c_ll = load_scalar ctx ~hl_buffer:c_var ~idx ~dtype in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_ALU
           {
             dst;
             op = Lowered.Ternary Mulacc;
             args = [ a_ll; b_ll; c_ll ];
             dtype = Dtype.Any_Dtype dtype;
           });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | View { in_var; shape_tracker = _; out_var; _ }
  | Valid { in_var; shape_tracker = _; out_var; _ }
  | Detach { in_var; out_var; _ }
  | Contiguous_Backward { in_var; out_var; _ }
  | Bitcast { in_var; target_dtype = _; out_var; _ } ->
      (match Hashtbl.find_opt ctx.scalar_map in_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx in_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Vconst { values; dtype; out_var } ->
      (* Vector constant - for now treat as buffer with preset values *)
      let ll = ll_of_hl ctx out_var ~buffer:true in
      add_instr ctx
        (L_Buffer
           {
             dtype = Dtype.Any_Dtype dtype;
             size = Array.length values;
             out = ll;
           });
      (* TODO: Initialize buffer with values *)
      Ok ()
  | Buffer_View { buffer_var; size = _; offset = _; dtype = _; out_var } ->
      (* For now, just alias the buffers *)
      let b = ll_of_hl ctx buffer_var ~buffer:true in
      Hashtbl.replace ctx.buffer_map out_var b;
      Ok ()
  | Index { in_var; idx_var; valid_var = _; out_var; dtype } ->
      (* Simplified index - needs proper implementation *)
      let base_idx = gtid ctx in
      let idx =
        load_scalar ctx ~hl_buffer:idx_var ~idx:base_idx ~dtype:Dtype.Int32
      in
      let in_ll = load_scalar ctx ~hl_buffer:in_var ~idx ~dtype in
      Hashtbl.replace ctx.scalar_map out_var in_ll;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx
           (L_Store { buf = out_buf; idx = base_idx; src = in_ll; valid = None }));
      Ok ()
  | Gep { in_var; indices; out_var; dtype } ->
      (* Get element pointer - for vectors *)
      let ll = ll_of_hl ctx in_var ~buffer:false in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_Gep { dst; src = ll; indices; dtype = Dtype.Any_Dtype dtype });
      Hashtbl.replace ctx.scalar_map out_var dst;
      Ok ()
  | Vectorize { in_vars; out_var; dtype } ->
      (* Build vector from scalars *)
      let srcs = Array.map (fun v -> ll_of_hl ctx v ~buffer:false) in_vars in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        {
          dtype = Dtype.Any_Dtype dtype;
          shape = [| Array.length in_vars |];
          device = None;
        };
      add_instr ctx (L_Vectorize { dst; srcs; dtype = Dtype.Any_Dtype dtype });
      Hashtbl.replace ctx.scalar_map out_var dst;
      Ok ()
  | Wmma { a_var; b_var; c_var; m; n; k; out_var; dtype } ->
      (* Tensor core operations *)
      let a_ll = ll_of_hl ctx a_var ~buffer:true in
      let b_ll = ll_of_hl ctx b_var ~buffer:true in
      let c_ll = ll_of_hl ctx c_var ~buffer:true in
      let dst = ll_of_hl ctx out_var ~buffer:true in
      add_instr ctx
        (L_Wmma
           {
             dst;
             a = a_ll;
             b = b_ll;
             c = c_ll;
             m;
             n;
             k;
             dtype = Dtype.Any_Dtype dtype;
           });
      Ok ()
  | Define_Var { sym_var; out_var; dtype = _ } ->
      let ll = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx (L_Define_Var { sym_var; out = ll });
      Ok ()
  | Bind { sym_var; value; out_var; dtype = _ } ->
      (* Bind symbolic var to value *)
      let sym_ll = ll_of_hl ctx sym_var ~buffer:false in
      let dst = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx
        (L_Const
           {
             dtype = Dtype.Any_Dtype Dtype.Int32;
             value = string_of_int value;
             out = dst;
           });
      add_instr ctx (L_Assign { dst = sym_ll; src = dst });
      Ok ()
  | Multi { device_vars; axis = _; real_mask = _; out_var; dtype = _ } ->
      (* Multi-device tensor - for now just use first device *)
      (if Array.length device_vars > 0 then
         match Hashtbl.find_opt ctx.scalar_map device_vars.(0) with
         | Some s -> Hashtbl.replace ctx.scalar_map out_var s
         | None ->
             let b = ll_of_hl ctx device_vars.(0) ~buffer:true in
             Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Fuse { in_var; out_var; dtype = _ } ->
      (* Fusion marker - pass through *)
      (match Hashtbl.find_opt ctx.scalar_map in_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx in_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()
  | Unroll { loop_var; unroll_factor; out_var; dtype = _ } ->
      (* Loop unroll directive *)
      let ll = ll_of_hl ctx loop_var ~buffer:false in
      let dst = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx (L_Unroll { idx = ll; iterations = unroll_factor });
      add_instr ctx (L_Assign { dst; src = ll });
      Ok ()
  | Contract { in_vars = _; contraction_axes = _; out_var; dtype } ->
      (* Tensor contraction - simplified for now *)
      let idx = gtid ctx in
      let dst = Var.fresh () in
      ensure_meta ctx dst
        { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |]; device = None };
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = "0"; out = dst });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx (L_Store { buf = out_buf; idx; src = dst; valid = None }));
      Ok ()
  | Sink { deps = _; dtype = _ } ->
      (* Sink has no output, just ensures dependencies are computed *)
      Ok ()
  | Kernel
      { ast = _; input_vars = _; output_vars = _; metadata = _; out_var; dtype }
    ->
      (* Kernel wrapper - simplified *)
      let ll = ll_of_hl ctx out_var ~buffer:true in
      add_instr ctx
        (L_Buffer { dtype = Dtype.Any_Dtype dtype; size = 1; out = ll });
      Ok ()
  | Unique { id; out_var; dtype } ->
      (* Unique identifier generation *)
      let ll = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx
        (L_Const
           { dtype = Dtype.Any_Dtype dtype; value = string_of_int id; out = ll });
      Ok ()
  | Device { device_name = _; out_var; dtype } ->
      (* Device marker - for now just create a dummy value *)
      let ll = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = "0"; out = ll });
      Ok ()
  | Custom { op_name; in_vars; attributes; out_var; dtype = _ } ->
      (* Custom operation *)
      let args = Array.map (fun v -> ll_of_hl ctx v ~buffer:false) in_vars in
      let dst = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx
        (L_Custom { dst = Some dst; op_name; args; attributes; inline = false });
      Ok ()
  | Noop { in_var = _; out_var; dtype } ->
      (* No operation - just create dummy output *)
      let ll = ll_of_hl ctx out_var ~buffer:false in
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = "0"; out = ll });
      Ok ()

(* ───── top-level entry ───── *)

let lower_kernel ~(kernel_spec : Scheduler.kernel_spec_t)
    ~original_graph_vars_metadata =
  let ( let* ) = Result.bind in
  let ( let+ ) = fun r f -> Result.map f r in
  let ctx = new_ctx original_graph_vars_metadata kernel_spec.vars_metadata in
  let open Result in
  let+ () =
    List.fold_left
      (fun acc n ->
        let* _ = acc in
        lower_node ctx n kernel_spec.outputs)
      (Ok ()) kernel_spec.nodes
  in
  {
    Lowered.instructions = List.rev !(ctx.instrs);
    vars_metadata = ctx.meta;
    kernel_input_vars =
      List.map (fun v -> ll_of_hl ctx v ~buffer:true) kernel_spec.inputs;
    kernel_output_vars =
      List.map (fun v -> ll_of_hl ctx v ~buffer:true) kernel_spec.outputs;
    symbolic_vars = [];
  }
