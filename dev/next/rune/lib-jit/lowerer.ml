open Ir

type lowering_ctx = {
  (* Maps high-level Var.t (from kernel_spec) to low-level Var.t (for lowered
     instructions) This is primarily for scalar values / registers. *)
  mutable var_map : (Var.t, Var.t) Hashtbl.t;
  (* Maps high-level Var.t representing buffers to low-level Var.t (which will
     be kernel args or LI_Buffer outputs) *)
  mutable buffer_var_map : (Var.t, Var.t) Hashtbl.t;
  (* List of lowered instructions, built in reverse order. *)
  mutable instructions : Lowered.any_instruction list;
  (* Metadata for all variables in the lowered graph. *)
  mutable lowered_vars_metadata : (Var.t, var_metadata) Hashtbl.t;
  (* Original graph metadata, for looking up shapes/dtypes of inputs not
     produced in this kernel. *)
  original_graph_vars_metadata : (Var.t, var_metadata) Hashtbl.t;
  (* Metadata for vars defined *within* the current kernel spec. *)
  kernel_spec_vars_metadata : (Var.t, var_metadata) Hashtbl.t;
}

let new_lowering_ctx original_graph_vars_metadata kernel_spec_vars_metadata :
    lowering_ctx =
  {
    var_map = Hashtbl.create 16;
    buffer_var_map = Hashtbl.create 16;
    instructions = [];
    lowered_vars_metadata = Hashtbl.create 16;
    original_graph_vars_metadata;
    kernel_spec_vars_metadata;
  }

(* Get or create a low-level variable for a high-level variable. If `is_buffer`
   is true, it treats `hl_var` as representing a buffer. Otherwise, it's treated
   as a scalar value. *)
let get_or_create_ll_var (ctx : lowering_ctx) (hl_var : Var.t)
    (is_buffer : bool) : Var.t =
  let target_map = if is_buffer then ctx.buffer_var_map else ctx.var_map in
  match Hashtbl.find_opt target_map hl_var with
  | Some ll_var -> ll_var
  | None ->
      let ll_var = Var.fresh () in
      Hashtbl.add target_map hl_var ll_var;
      (* Copy metadata - IMPORTANT: Use original_graph_vars_metadata for
         external vars, kernel_spec_vars_metadata for vars produced within this
         kernel_spec *)
      let meta_src =
        if Hashtbl.mem ctx.kernel_spec_vars_metadata hl_var then
          ctx.kernel_spec_vars_metadata
        else ctx.original_graph_vars_metadata
      in
      (match Hashtbl.find_opt meta_src hl_var with
      | Some meta -> Hashtbl.add ctx.lowered_vars_metadata ll_var meta
      | None ->
          (* This indicates an issue or an unhandled var type *)
          Printf.eprintf
            "Warning: No metadata for hl_var %s during ll_var creation\n"
            (Var.to_string hl_var));
      ll_var

let add_instr (ctx : lowering_ctx) (instr : ('a, 'b) Lowered.instruction_t) :
    unit =
  ctx.instructions <- Lowered.Any_Instruction instr :: ctx.instructions

(* Helper to get metadata for a high-level variable. *)
let get_hl_var_metadata (ctx : lowering_ctx) (hl_var : Var.t) :
    var_metadata option =
  match Hashtbl.find_opt ctx.kernel_spec_vars_metadata hl_var with
  | Some meta -> Some meta
  | None -> Hashtbl.find_opt ctx.original_graph_vars_metadata hl_var

(* --- Indexing Strategy (Simplified for now) --- This is the most complex part
   and will need significant expansion. For now, assume element-wise operations
   where output index = input index = global_thread_id. We'll need a more robust
   way to handle broadcasts, reductions, views, etc. *)
let get_global_thread_id_0d (ctx : lowering_ctx) : Var.t =
  (* For simplicity, assume a 1D global ID for now. This needs to be N-D. *)
  let ll_gtid_var = Var.fresh () in
  let gtid_meta = { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] } in
  (* gtid is scalar int *)
  Hashtbl.add ctx.lowered_vars_metadata ll_gtid_var gtid_meta;
  add_instr ctx
    (Lowered.LI_Special_Index
       {
         name_hint = "gtid_x";
         kind = Special_index_kind.Global_task_idx 0;
         out_var = ll_gtid_var;
       });
  ll_gtid_var

(* Helper to create load instruction and associated metadata *)
let create_load_instr (ctx : lowering_ctx) ~(buffer_hl_var : Var.t)
    ~(ll_idx_var : Var.t) ~(dtype : ('elt, 'ph) Dtype.t) : Var.t =
  let ll_in_buf = get_or_create_ll_var ctx buffer_hl_var true in
  let ll_val = Var.fresh () in
  get_hl_var_metadata ctx buffer_hl_var
  |> Option.iter (fun m ->
         Hashtbl.add ctx.lowered_vars_metadata ll_val { m with shape = [| 1 |] });
  add_instr ctx
    (Lowered.LI_Load
       {
         buffer_source_var = ll_in_buf;
         indices_vars = [ ll_idx_var ];
         valid_mask_var = None;
         out_var = ll_val;
         dtype;
       });
  ll_val

(* Lowers a single high-level node. *)
let lower_node (ctx : lowering_ctx) (node : any_node)
    (kernel_spec_outputs : Var.t list) : (unit, string) result =
  match node with
  | Any_Node (Placeholder { out_var = hl_out_var; _ }) ->
      let _ll_buf_var = get_or_create_ll_var ctx hl_out_var true in
      (* This HL var will be part of kernel_spec.inputs, mapping handled at end
         of lower_kernel *)
      Ok ()
  | Any_Node (Const_Scalar { value; dtype; out_var = hl_out_var }) ->
      let ll_out_scalar_var = get_or_create_ll_var ctx hl_out_var false in
      add_instr ctx
        (Lowered.LI_Const_Scalar { value; dtype; out_var = ll_out_scalar_var });
      Ok ()
  | Any_Node (Buffer { dtype; size_in_elements; out_var = hl_out_var }) ->
      let ll_smem_buf_var = get_or_create_ll_var ctx hl_out_var true in
      add_instr ctx
        (Lowered.LI_Buffer
           { dtype; size_in_elements; out_var = ll_smem_buf_var });
      Ok ()
  | Any_Node
      (Add
         { in_a_var = hl_in_a; in_b_var = hl_in_b; out_var = hl_out_var; dtype })
    ->
      let ll_idx_var = get_global_thread_id_0d ctx in
      let ll_val_a =
        create_load_instr ctx ~buffer_hl_var:hl_in_a ~ll_idx_var ~dtype
      in
      let ll_val_b =
        create_load_instr ctx ~buffer_hl_var:hl_in_b ~ll_idx_var ~dtype
      in

      let ll_res_val = Var.fresh () in
      get_hl_var_metadata ctx hl_out_var
      |> Option.iter (fun m ->
             Hashtbl.add ctx.lowered_vars_metadata ll_res_val
               { m with shape = [| 1 |] });
      add_instr ctx
        (Lowered.LI_Scalar_ALU
           {
             op_type = Lowered.Scalar_Add;
             inputs_vars = [ ll_val_a; ll_val_b ];
             out_var = ll_res_val;
             dtype;
           });
      Hashtbl.replace ctx.var_map hl_out_var ll_res_val;

      (if List.mem hl_out_var kernel_spec_outputs then
         let ll_out_buf = get_or_create_ll_var ctx hl_out_var true in
         add_instr ctx
           (Lowered.LI_Store
              {
                buffer_target_var = ll_out_buf;
                indices_vars = [ ll_idx_var ];
                scalar_value_to_store_var = ll_res_val;
                valid_mask_var = None;
              }));
      Ok ()
  | Any_Node
      (Reduce_Axis
         {
           in_var = hl_in_var;
           reduce_op_kind;
           axes = _;
           out_var = hl_out_var;
           dtype;
         }) ->
      let _ll_in_buf = get_or_create_ll_var ctx hl_in_var true in
      (* Ensure mapped *)
      let in_meta =
        match get_hl_var_metadata ctx hl_in_var with
        | Some m -> m
        | None ->
            failwith
              ("Input metadata missing for Reduce: " ^ Var.to_string hl_in_var)
      in
      let total_elements_to_reduce = Array.fold_left ( * ) 1 in_meta.shape in

      let ll_loop_idx_upper_bound = Var.fresh () in
      Hashtbl.add ctx.lowered_vars_metadata ll_loop_idx_upper_bound
        { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };
      add_instr ctx
        (Lowered.LI_Const_Scalar
           {
             value = total_elements_to_reduce;
             dtype = Dtype.Int32;
             out_var = ll_loop_idx_upper_bound;
           });

      let ll_loop_idx = Var.fresh () in
      Hashtbl.add ctx.lowered_vars_metadata ll_loop_idx
        { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };
      add_instr ctx
        (Lowered.LI_Range
           {
             name_hint = "reduce_loop";
             upper_bound_exclusive = ll_loop_idx_upper_bound;
             out_var = ll_loop_idx;
           });

      let ll_acc = Var.fresh () in
      let acc_meta = { dtype = Dtype.Any_Dtype dtype; shape = [| 1 |] } in
      Hashtbl.add ctx.lowered_vars_metadata ll_acc acc_meta;

      let () =
        match dtype with
        | Dtype.Float32 ->
            let identity_val =
              match reduce_op_kind with
              | Reduce_Sum -> 0.0
              | Reduce_Max -> Float.neg_infinity
            in
            add_instr ctx
              (Lowered.LI_Const_Scalar
                 {
                   value = identity_val;
                   dtype = Dtype.Float32;
                   out_var = ll_acc;
                 })
        | Dtype.Int32 ->
            let identity_val =
              match reduce_op_kind with
              | Reduce_Sum -> 0
              | Reduce_Max -> min_int
            in
            add_instr ctx
              (Lowered.LI_Const_Scalar
                 { value = identity_val; dtype = Dtype.Int32; out_var = ll_acc })
        | Dtype.Uint8 ->
            let identity_val =
              match reduce_op_kind with Reduce_Sum -> 0 | Reduce_Max -> 0
            in
            add_instr ctx
              (Lowered.LI_Const_Scalar
                 { value = identity_val; dtype = Dtype.Uint8; out_var = ll_acc })
        | Dtype.Bool ->
            let identity_val =
              match reduce_op_kind with
              | Reduce_Sum -> false
              | Reduce_Max -> true
            in
            add_instr ctx
              (Lowered.LI_Const_Scalar
                 { value = identity_val; dtype = Dtype.Bool; out_var = ll_acc })
        | Dtype.Unit -> failwith "Cannot create accumulator for Unit type"
      in

      let ll_current_val =
        create_load_instr ctx ~buffer_hl_var:hl_in_var ~ll_idx_var:ll_loop_idx
          ~dtype
      in

      let alu_op_type =
        match reduce_op_kind with
        | Reduce_Sum -> Lowered.Scalar_Add
        | Reduce_Max -> Lowered.Scalar_Max
      in
      add_instr ctx
        (Lowered.LI_Scalar_ALU
           {
             op_type = alu_op_type;
             inputs_vars = [ ll_acc; ll_current_val ];
             out_var = ll_acc;
             dtype;
           });

      add_instr ctx Lowered.LI_End_Range;

      (* Close the loop *)
      if List.mem hl_out_var kernel_spec_outputs then (
        let ll_out_buf = get_or_create_ll_var ctx hl_out_var true in
        let ll_const_idx_0 = Var.fresh () in
        Hashtbl.add ctx.lowered_vars_metadata ll_const_idx_0
          { dtype = Dtype.Any_Dtype Dtype.Int32; shape = [| 1 |] };
        add_instr ctx
          (Lowered.LI_Const_Scalar
             { value = 0; dtype = Dtype.Int32; out_var = ll_const_idx_0 });
        add_instr ctx
          (Lowered.LI_Store
             {
               buffer_target_var = ll_out_buf;
               indices_vars = [ ll_const_idx_0 ];
               scalar_value_to_store_var = ll_acc;
               valid_mask_var = None;
             }))
      else Hashtbl.replace ctx.var_map hl_out_var ll_acc;
      Ok ()
  | Any_Node (Expand { in_var = hl_in_var; out_var = hl_out_var; _ })
  | Any_Node (Reshape { in_var = hl_in_var; out_var = hl_out_var; _ })
  | Any_Node (Permute { in_var = hl_in_var; out_var = hl_out_var; _ }) ->
      (match Hashtbl.find_opt ctx.var_map hl_in_var with
      | Some ll_in_scalar_var ->
          Hashtbl.replace ctx.var_map hl_out_var ll_in_scalar_var
      | None ->
          let ll_in_buf_var = get_or_create_ll_var ctx hl_in_var true in
          Hashtbl.replace ctx.buffer_var_map hl_out_var ll_in_buf_var);
      Ok ()
  | Any_Node (Mul { out_var; _ }) ->
      Error
        (Printf.sprintf
           "Lowering for Mul not yet implemented (node producing %s)"
           (Var.to_string out_var))

let lower_kernel ~(kernel_spec : Scheduler.kernel_spec_t)
    ~(original_graph_vars_metadata : (Var.t, var_metadata) Hashtbl.t) :
    (Ir.Lowered.graph_t, string) result =
  let ctx =
    new_lowering_ctx original_graph_vars_metadata kernel_spec.vars_metadata
  in

  let result =
    List.fold_left
      (fun acc_res node ->
        match acc_res with
        | Error e -> Error e
        | Ok () -> lower_node ctx node kernel_spec.outputs)
      (Ok ()) kernel_spec.nodes
  in

  match result with
  | Error e -> Error e
  | Ok () ->
      let final_instructions = List.rev ctx.instructions in
      let map_hl_to_ll_buffer_var category_name hl_var =
        match Hashtbl.find_opt ctx.buffer_var_map hl_var with
        | Some ll_var -> ll_var
        | None ->
            (* This can happen if an input/output var is not used in any node
               that maps it to a buffer_var, e.g. if it's only used in a view op
               that propagates the var mapping. Ensure it is mapped if it's a
               buffer. The Placeholder node handles inputs. Outputs might not be
               explicitly mapped if they are results of view ops on other
               outputs. *)
            if
              Hashtbl.mem ctx.original_graph_vars_metadata hl_var
              || Hashtbl.mem kernel_spec.vars_metadata hl_var
            then
              get_or_create_ll_var ctx hl_var
                true (* Ensure it's created if metadata exists *)
            else (
              Printf.eprintf
                "Lowerer Warning: HL %s var %s not found in buffer_var_map and \
                 no metadata.\n"
                category_name (Var.to_string hl_var);
              Var.fresh ()
              (* Should ideally be an error or ensure all such vars are mapped
                 by nodes *))
      in
      let final_ll_kernel_inputs =
        List.map (map_hl_to_ll_buffer_var "input") kernel_spec.inputs
      in
      let final_ll_kernel_outputs =
        List.map (map_hl_to_ll_buffer_var "output") kernel_spec.outputs
      in

      Ok
        {
          Lowered.instructions = final_instructions;
          vars_metadata = ctx.lowered_vars_metadata;
          kernel_input_vars = final_ll_kernel_inputs;
          kernel_output_vars = final_ll_kernel_outputs;
        }
