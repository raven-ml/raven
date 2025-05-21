(* lowerer.ml *)

open Ir

(* ────────── helpers ────────── *)

let const_to_string : type a. a Dtype.t -> a -> string =
 fun dt v ->
  match dt with
  | Dtype.Float32 -> Printf.sprintf "%gf" v
  | Dtype.Int32 -> string_of_int v
  | Dtype.Uint8 -> string_of_int v
  | Dtype.Bool -> if v then "true" else "false"
  | Dtype.Unit -> "0"

(* ────────── lowering context ────────── *)

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

(* ────────── mapping helpers ────────── *)

let ll_of_hl ctx hl ~buffer =
  let tbl = if buffer then ctx.buffer_map else ctx.scalar_map in
  match Hashtbl.find_opt tbl hl with
  | Some ll -> ll
  | None ->
      let ll = Var.fresh () in
      Hashtbl.add tbl hl ll;
      Option.iter (ensure_meta ctx ll) (meta_of ctx hl);
      ll

(* ────────── frequently used snippets ────────── *)

let gtid ctx =
  let v = Var.fresh () in
  ensure_meta ctx v { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };
  add_instr ctx
    (Lowered.L_SpecialIndex
       { dst = v; kind = Special_index_kind.Global_task_idx 0 });
  v

let load_scalar ctx ~hl_buffer ~idx ~dtype =
  let buf = ll_of_hl ctx hl_buffer ~buffer:true in
  let dst = Var.fresh () in
  ensure_meta ctx dst { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |] };
  add_instr ctx
    (Lowered.L_Load
       { dst; buf; idxs = [ idx ]; mask = None; dtype = Dtype.Any_Dtype dtype });
  dst

(* ────────── node lowering ────────── *)

let lower_node ctx (Any_Node n) kernel_outs =
  let open Lowered in
  match n with
  | Placeholder { out_var; _ } ->
      ignore (ll_of_hl ctx out_var ~buffer:true);
      Ok ()
  | Buffer { dtype; size_in_elements; out_var } ->
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
      ensure_meta ctx dst { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |] };
      add_instr ctx
        (L_ALU
           {
             dst;
             op = Bin op;
             args = [ a_ll; b_ll ];
             dtype = Dtype.Any_Dtype dtype;
           });
      Hashtbl.replace ctx.scalar_map out_var dst;
      (if List.mem out_var kernel_outs then
         let out_buf = ll_of_hl ctx out_var ~buffer:true in
         add_instr ctx
           (L_Store { buf = out_buf; idxs = [ idx ]; src = dst; mask = None }));
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
        { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };

      let ub = Var.fresh () in
      ensure_meta ctx ub
        { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };
      add_instr ctx
        (L_Const
           {
             dtype = Dtype.Any_Dtype Dtype.Int32;
             value = string_of_int total;
             out = ub;
           });
      add_instr ctx (L_Range { idx; upper = ub });

      let acc = Var.fresh () in
      ensure_meta ctx acc { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |] };
      let identity =
        match (dtype, reduce_op_kind) with
        | Dtype.Float32, Reduce_Sum -> "0.0"
        | Dtype.Float32, Reduce_Max -> "-INFINITY"
        | Dtype.Int32, Reduce_Sum -> "0"
        | Dtype.Int32, Reduce_Max -> string_of_int min_int
        | Dtype.Uint8, Reduce_Sum -> "0"
        | Dtype.Uint8, Reduce_Max -> "0"
        | Dtype.Bool, Reduce_Sum -> "false"
        | Dtype.Bool, Reduce_Max -> "true"
        | Dtype.Unit, _ -> "0"
      in
      add_instr ctx
        (L_Const { dtype = Dtype.Any_Dtype dtype; value = identity; out = acc });

      let cur = load_scalar ctx ~hl_buffer:in_var ~idx ~dtype in
      let op =
        match reduce_op_kind with
        | Reduce_Sum -> Bin Add
        | Reduce_Max -> Bin Max
      in
      add_instr ctx
        (L_ALU
           { dst = acc; op; args = [ acc; cur ]; dtype = Dtype.Any_Dtype dtype });
      add_instr ctx L_EndRange;

      if List.mem out_var kernel_outs then (
        let ob = ll_of_hl ctx out_var ~buffer:true in
        let z = Var.fresh () in
        ensure_meta ctx z
          { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };
        add_instr ctx
          (L_Const { dtype = Dtype.Any_Dtype Dtype.Int32; value = "0"; out = z });
        add_instr ctx
          (L_Store { buf = ob; idxs = [ z ]; src = acc; mask = None }))
      else Hashtbl.replace ctx.scalar_map out_var acc;
      Ok ()
  | Expand { in_var; out_var; _ }
  | Reshape { in_var; out_var; _ }
  | Permute { in_var; out_var; _ } ->
      (match Hashtbl.find_opt ctx.scalar_map in_var with
      | Some s -> Hashtbl.replace ctx.scalar_map out_var s
      | None ->
          let b = ll_of_hl ctx in_var ~buffer:true in
          Hashtbl.replace ctx.buffer_map out_var b);
      Ok ()

(* ────────── top-level entry ────────── *)

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
  }
