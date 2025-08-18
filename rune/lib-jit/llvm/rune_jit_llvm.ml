(* rune_jit_llvm.ml - LLVM JIT backend for Rune *)

open Rune_jit

(* ───── Opaque native handles ───── *)

type device_info = {
  context : Llvm.llcontext;
  module_ : Llvm.llmodule;
  engine : Llvm_executionengine.llexecutionengine;
  target_data : Llvm_target.DataLayout.t;
}

type device_buffer_native = {
  ptr : nativeint;
  size_bytes : int;
  dtype : Ir.Dtype.any;
}

type compiled_artifact_native = {
  (* Don't store module_ as it's owned by execution engine *)
  function_names : string list;
}

type callable_kernel_native = { name : string }

(* ───── Public record wrappers ───── *)

type nonrec 'a device_buffer =
  ('a, device_buffer_native) Backend_intf.device_buffer

type nonrec any_device_buffer =
  device_buffer_native Backend_intf.any_device_buffer

type nonrec compiled_artifact =
  compiled_artifact_native Backend_intf.compiled_artifact

type nonrec callable_kernel =
  callable_kernel_native Backend_intf.callable_kernel

(* ───── Helper functions ───── *)

let dtype_to_llvm_type context (Ir.Dtype.Any_Dtype dt) =
  match dt with
  | Ir.Dtype.Float32 -> Llvm.float_type context
  | Ir.Dtype.Int32 -> Llvm.i32_type context
  | Ir.Dtype.Bool -> Llvm.i1_type context
  | Ir.Dtype.Uint8 -> Llvm.i8_type context
  | Ir.Dtype.Unit -> Llvm.void_type context

let foreign_malloc =
  Foreign.foreign "malloc" Ctypes.(size_t @-> returning (ptr void))

(* let foreign_free = Foreign.foreign "free" Ctypes.(ptr void @-> returning
   void) *)
let foreign_memcpy =
  Foreign.foreign "memcpy"
    Ctypes.(ptr void @-> ptr void @-> size_t @-> returning (ptr void))

(* ───── Device_info ───── *)

module Device_info = struct
  (* Global singleton for the device info *)
  let global_device_info = ref None

  let get_default () =
    match !global_device_info with
    | Some info -> info
    | None ->
        let context = Llvm.global_context () in
        (* Don't create a module here - each kernel will have its own *)

        (* Initialize LLVM targets *)
        ignore (Llvm_all_backends.initialize ());

        (* Create execution engine with a dummy module *)
        let dummy_module = Llvm.create_module context "dummy" in
        let engine = Llvm_executionengine.create dummy_module in

        let target_data = Llvm_executionengine.data_layout engine in

        let info = { context; module_ = dummy_module; engine; target_data } in
        global_device_info := Some info;
        info

  let max_shared_memory _ =
    (* LLVM JIT doesn't have shared memory constraints like GPUs *)
    1024 * 1024 * 16 (* 16MB arbitrary limit *)

  let max_workgroup_size _ =
    (* CPU doesn't have workgroup constraints *)
    [| 1024; 1024; 1024 |]

  let supports_dtype _ _ = true
  let renderer_float4_str _ = None (* LLVM IR doesn't have built-in float4 *)
  let renderer_smem_prefix _ = "" (* No special prefix for shared memory *)
  let renderer_barrier_str _ = "" (* No barriers needed for CPU *)
end

(* ───── Renderer ───── *)

module Renderer = struct
  (* Global storage for lowered IR - shared with Compiler *)
  let pending_kernels : (string, Ir.Lowered.graph_t) Hashtbl.t =
    Hashtbl.create 10

  let render ~device_info:_ ~lowered_ir ~kernel_name =
    (* Store the lowered IR for the Compiler to use *)
    Hashtbl.add pending_kernels kernel_name lowered_ir;
    (* Return kernel name as "source code" *)
    kernel_name
end

(* ───── Compiler ───── *)

module Compiler = struct
  type compile_options = {
    optimization_level : int;
    fast_math : bool;
    device_info : device_info;
  }

  let default_options device_info =
    { optimization_level = 2; fast_math = true; device_info }

  let compile_instruction context builder vars_table = function
    | Ir.Lowered.L_Const { dtype; value; out } ->
        let lltype = dtype_to_llvm_type context dtype in
        let llvalue =
          match dtype with
          | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
              Llvm.const_float lltype (float_of_string value)
          | Ir.Dtype.Any_Dtype Ir.Dtype.Int32 ->
              Llvm.const_int lltype (int_of_string value)
          | Ir.Dtype.Any_Dtype Ir.Dtype.Uint8 ->
              Llvm.const_int lltype (int_of_string value)
          | Ir.Dtype.Any_Dtype Ir.Dtype.Bool ->
              Llvm.const_int lltype (if value = "true" then 1 else 0)
          | _ -> failwith "Unsupported constant type"
        in
        Hashtbl.add vars_table out llvalue
    | Ir.Lowered.L_ALU { dst; op; args; dtype } ->
        let lltype = dtype_to_llvm_type context dtype in
        let get_var v =
          try Hashtbl.find vars_table v
          with Not_found ->
            failwith (Printf.sprintf "Variable v%d not found" v)
        in
        let llvalue =
          match (op, args) with
          | Ir.Lowered.Binary Ir.Add, [ a; b ] -> (
              let a_val = get_var a in
              let b_val = get_var b in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fadd a_val b_val "add" builder
              | _ -> Llvm.build_add a_val b_val "add" builder)
          | Ir.Lowered.Binary Ir.Mul, [ a; b ] -> (
              let a_val = get_var a in
              let b_val = get_var b in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fmul a_val b_val "mul" builder
              | _ -> Llvm.build_mul a_val b_val "mul" builder)
          | Ir.Lowered.Binary Ir.Sub, [ a; b ] -> (
              let a_val = get_var a in
              let b_val = get_var b in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fsub a_val b_val "sub" builder
              | _ -> Llvm.build_sub a_val b_val "sub" builder)
          | Ir.Lowered.Binary Ir.Div, [ a; b ] -> (
              let a_val = get_var a in
              let b_val = get_var b in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fdiv a_val b_val "div" builder
              | _ -> Llvm.build_sdiv a_val b_val "div" builder)
          | Ir.Lowered.Binary Ir.Cmplt, [ a; b ] -> (
              let a_val = get_var a in
              let b_val = get_var b in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fcmp Llvm.Fcmp.Olt a_val b_val "cmplt" builder
              | _ -> Llvm.build_icmp Llvm.Icmp.Slt a_val b_val "cmplt" builder)
          | Ir.Lowered.Unary Ir.Neg, [ a ] -> (
              let a_val = get_var a in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fneg a_val "neg" builder
              | _ -> Llvm.build_neg a_val "neg" builder)
          | Ir.Lowered.Unary Ir.Sqrt, [ a ] ->
              let a_val = get_var a in
              let fn_type = Llvm.function_type lltype [| lltype |] in
              let sqrt_fn =
                Llvm.declare_function "llvm.sqrt.f32" fn_type
                  (Llvm.global_parent
                     (Llvm.block_parent (Llvm.insertion_block builder)))
              in
              Llvm.build_call fn_type sqrt_fn [| a_val |] "sqrt" builder
          | Ir.Lowered.Ternary Ir.Where, [ cond; true_val; false_val ] ->
              let cond_val = get_var cond in
              let true_val_ll = get_var true_val in
              let false_val_ll = get_var false_val in
              (* Ensure condition is i1 for select instruction *)
              let cond_i1 =
                if Llvm.type_of cond_val = Llvm.i1_type context then cond_val
                else if Llvm.type_of cond_val = Llvm.i8_type context then
                  (* For i8, compare with 0 *)
                  Llvm.build_icmp Llvm.Icmp.Ne cond_val
                    (Llvm.const_int (Llvm.i8_type context) 0)
                    "cond_bool" builder
                else
                  (* For other integer types, compare with 0 *)
                  Llvm.build_icmp Llvm.Icmp.Ne cond_val
                    (Llvm.const_int (Llvm.type_of cond_val) 0)
                    "cond_bool" builder
              in
              Llvm.build_select cond_i1 true_val_ll false_val_ll "where" builder
          | Ir.Lowered.Ternary Ir.Mulacc, [ a; b; c ] -> (
              let a_val = get_var a in
              let b_val = get_var b in
              let c_val = get_var c in
              let mul_result =
                match dtype with
                | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                    Llvm.build_fmul a_val b_val "mul_tmp" builder
                | _ -> Llvm.build_mul a_val b_val "mul_tmp" builder
              in
              match dtype with
              | Ir.Dtype.Any_Dtype Ir.Dtype.Float32 ->
                  Llvm.build_fadd mul_result c_val "mulacc" builder
              | _ -> Llvm.build_add mul_result c_val "mulacc" builder)
          | _ -> failwith "Unsupported ALU operation"
        in
        Hashtbl.add vars_table dst llvalue
    | Ir.Lowered.L_Load { dst; buf; idx; dtype; valid = _ } ->
        let ptr_val =
          try Hashtbl.find vars_table buf
          with Not_found ->
            failwith (Printf.sprintf "Buffer v%d not found" buf)
        in
        let idx_val =
          try Hashtbl.find vars_table idx
          with Not_found ->
            failwith (Printf.sprintf "Index v%d not found" idx)
        in
        (* Use uint8 for bools since that's how they're stored *)
        let actual_lltype =
          match dtype with
          | Ir.Dtype.Any_Dtype Ir.Dtype.Bool -> Llvm.i8_type context
          | _ -> dtype_to_llvm_type context dtype
        in
        (* Convert index to i64 for GEP if needed *)
        let idx_64 =
          if Llvm.type_of idx_val = Llvm.i32_type context then
            Llvm.build_sext idx_val (Llvm.i64_type context) "idx_i64" builder
          else idx_val
        in
        let gep =
          Llvm.build_gep actual_lltype ptr_val [| idx_64 |] "gep" builder
        in
        let loaded = Llvm.build_load actual_lltype gep "load" builder in
        (* Convert uint8 to bool if needed *)
        let final_value =
          match dtype with
          | Ir.Dtype.Any_Dtype Ir.Dtype.Bool ->
              (* Compare i8 != 0 for bool conversion *)
              Llvm.build_icmp Llvm.Icmp.Ne loaded
                (Llvm.const_int (Llvm.i8_type context) 0)
                "to_bool" builder
          | _ -> loaded
        in
        Hashtbl.add vars_table dst final_value
    | Ir.Lowered.L_Store { buf; idx; src; valid = _ } ->
        let ptr_val =
          try Hashtbl.find vars_table buf
          with Not_found ->
            failwith (Printf.sprintf "Buffer v%d not found" buf)
        in
        let idx_val =
          try Hashtbl.find vars_table idx
          with Not_found ->
            failwith (Printf.sprintf "Index v%d not found" idx)
        in
        let src_val =
          try Hashtbl.find vars_table src
          with Not_found ->
            failwith (Printf.sprintf "Source v%d not found" src)
        in
        let elem_type = Llvm.type_of src_val in
        (* Convert index to i64 for GEP if needed *)
        let idx_64 =
          if Llvm.type_of idx_val = Llvm.i32_type context then
            Llvm.build_sext idx_val (Llvm.i64_type context) "idx_i64" builder
          else idx_val
        in
        let gep = Llvm.build_gep elem_type ptr_val [| idx_64 |] "gep" builder in
        ignore (Llvm.build_store src_val gep builder)
    | Ir.Lowered.L_Range { idx; bound } ->
        (* This is handled by process_instructions in compile_kernel *)
        let _bound_val =
          try Hashtbl.find vars_table bound
          with Not_found ->
            (* If bound is not found, it might be a constant *)
            let zero = Llvm.const_int (Llvm.i32_type context) 0 in
            Hashtbl.add vars_table bound zero;
            zero
        in
        (* Store loop counter - will be overwritten by proper loop handling *)
        let zero = Llvm.const_int (Llvm.i32_type context) 0 in
        Hashtbl.add vars_table idx zero
    | Ir.Lowered.L_Special { dst; kind } ->
        (* Handle special indices like thread/block IDs *)
        let value =
          match kind with
          | Ir.Special_index_kind.Global_task_idx _
          | Ir.Special_index_kind.Local_thread_idx _
          | Ir.Special_index_kind.Workgroup_idx _ ->
              (* For CPU, just use 0 *)
              Llvm.const_int (Llvm.i32_type context) 0
        in
        Hashtbl.add vars_table dst value
    | Ir.Lowered.L_Buffer { dtype; size; out } ->
        (* Allocate a buffer - for now just create a pointer *)
        let lltype = dtype_to_llvm_type context dtype in
        let alloca =
          Llvm.build_array_alloca lltype
            (Llvm.const_int (Llvm.i32_type context) size)
            "buffer" builder
        in
        Hashtbl.add vars_table out alloca
    | Ir.Lowered.L_EndRange | Ir.Lowered.L_EndIf | Ir.Lowered.L_Barrier ->
        (* Control flow markers - ignore for now *)
        ()
    | _ -> () (* Skip other instructions for now *)

  (* Global counter and mapping for unique kernel names *)
  let kernel_counter = ref 0

  let kernel_name_map =
    Hashtbl.create 32 (* Maps original names to unique names *)

  let compile_kernel device_info lowered_ir original_kernel_name =
    (* Make kernel name unique to avoid conflicts in execution engine *)
    let kernel_name =
      Printf.sprintf "%s_%d" original_kernel_name !kernel_counter
    in
    incr kernel_counter;
    (* Store the mapping *)
    Hashtbl.add kernel_name_map original_kernel_name kernel_name;

    let context = device_info.context in
    let module_ = Llvm.create_module context kernel_name in

    (* Create function signature - takes pointers and size as arguments *)
    let ptr_type = Llvm.pointer_type context in
    let i32_type = Llvm.i32_type context in
    (* Combine input and output vars as parameters *)
    let all_params =
      lowered_ir.Ir.Lowered.kernel_input_vars
      @ lowered_ir.Ir.Lowered.kernel_output_vars
    in
    let num_params = List.length all_params in
    (* Add extra parameter for size *)
    let param_types =
      Array.init (num_params + 1) (fun i ->
          if i < num_params then ptr_type else i32_type)
    in
    let fn_type = Llvm.function_type (Llvm.void_type context) param_types in
    let fn = Llvm.declare_function kernel_name fn_type module_ in

    (* Create entry block *)
    let entry_bb = Llvm.append_block context "entry" fn in
    let builder = Llvm.builder context in
    Llvm.position_at_end entry_bb builder;

    (* Map all vars to function parameters *)
    let vars_table = Hashtbl.create 256 in
    List.iteri
      (fun i var ->
        let param = Llvm.param fn i in
        Hashtbl.add vars_table var param)
      all_params;

    (* Get the size parameter (last parameter) *)
    let size_param = Llvm.param fn num_params in

    (* Track if we have L_Special for global task index *)
    let has_global_task_idx = ref None in
    List.iter
      (function
        | Ir.Lowered.L_Special
            { dst; kind = Ir.Special_index_kind.Global_task_idx _ } ->
            has_global_task_idx := Some dst
        | _ -> ())
      lowered_ir.instructions;

    (* Track loop context for proper control flow *)
    let loop_stack = ref [] in

    (* Process instructions with proper loop handling *)
    let rec process_instructions instrs =
      match instrs with
      | [] -> ()
      | Ir.Lowered.L_Range { idx; bound } :: rest ->
          (* Create loop structure *)
          let bound_val =
            try Hashtbl.find vars_table bound
            with Not_found ->
              (* Bound should have been added as a constant earlier *)
              failwith (Printf.sprintf "Bound variable v%d not found" bound)
          in

          (* Create basic blocks for loop *)
          let loop_bb = Llvm.append_block context "loop" fn in
          let body_bb = Llvm.append_block context "loop_body" fn in
          let end_bb = Llvm.append_block context "loop_end" fn in

          (* Initialize loop counter *)
          let counter_ptr =
            Llvm.build_alloca (Llvm.i32_type context) "counter" builder
          in
          let zero = Llvm.const_int (Llvm.i32_type context) 0 in
          ignore (Llvm.build_store zero counter_ptr builder);
          ignore (Llvm.build_br loop_bb builder);

          (* Loop header *)
          Llvm.position_at_end loop_bb builder;
          let current_idx =
            Llvm.build_load (Llvm.i32_type context) counter_ptr "idx" builder
          in
          Hashtbl.add vars_table idx current_idx;
          let cond =
            Llvm.build_icmp Llvm.Icmp.Slt current_idx bound_val "loop_cond"
              builder
          in
          ignore (Llvm.build_cond_br cond body_bb end_bb builder);

          (* Loop body *)
          Llvm.position_at_end body_bb builder;

          (* Find matching EndRange and process body *)
          let rec find_body acc = function
            | [] -> failwith "L_Range without matching L_EndRange"
            | Ir.Lowered.L_EndRange :: rest -> (List.rev acc, rest)
            | instr :: rest -> find_body (instr :: acc) rest
          in
          let body_instrs, rest_after_loop = find_body [] rest in

          (* Push loop context *)
          loop_stack := (counter_ptr, loop_bb) :: !loop_stack;

          (* Process body instructions *)
          process_instructions body_instrs;

          (* Pop loop context *)
          loop_stack := List.tl !loop_stack;

          (* Increment counter and branch back *)
          let next_idx =
            Llvm.build_add current_idx
              (Llvm.const_int (Llvm.i32_type context) 1)
              "next_idx" builder
          in
          ignore (Llvm.build_store next_idx counter_ptr builder);
          ignore (Llvm.build_br loop_bb builder);

          (* Continue after loop *)
          Llvm.position_at_end end_bb builder;
          process_instructions rest_after_loop
      | instr :: rest ->
          compile_instruction context builder vars_table instr;
          process_instructions rest
    in

    (* If we have a global task index, wrap everything in a loop *)
    let process_with_loop global_idx_var =
      (* Create loop structure *)
      let loop_bb = Llvm.append_block context "main_loop" fn in
      let body_bb = Llvm.append_block context "main_body" fn in
      let end_bb = Llvm.append_block context "main_end" fn in

      (* Initialize loop counter *)
      let counter_ptr = Llvm.build_alloca i32_type "main_counter" builder in
      let zero = Llvm.const_int i32_type 0 in
      ignore (Llvm.build_store zero counter_ptr builder);
      ignore (Llvm.build_br loop_bb builder);

      (* Loop header *)
      Llvm.position_at_end loop_bb builder;
      let current_idx =
        Llvm.build_load i32_type counter_ptr "main_idx" builder
      in
      let cond =
        Llvm.build_icmp Llvm.Icmp.Slt current_idx size_param "main_cond" builder
      in
      ignore (Llvm.build_cond_br cond body_bb end_bb builder);

      (* Loop body *)
      Llvm.position_at_end body_bb builder;

      (* Store the current index as the global task index *)
      Hashtbl.add vars_table global_idx_var current_idx;

      (* Process all instructions *)
      let rec process_body_instructions = function
        | [] -> ()
        | Ir.Lowered.L_Special
            { dst = _; kind = Ir.Special_index_kind.Global_task_idx _ }
          :: rest ->
            (* Skip - we already handled it *)
            process_body_instructions rest
        | instr :: rest ->
            compile_instruction context builder vars_table instr;
            process_body_instructions rest
      in
      process_body_instructions lowered_ir.instructions;

      (* Increment counter and branch back *)
      let next_idx =
        Llvm.build_add current_idx
          (Llvm.const_int i32_type 1)
          "next_main_idx" builder
      in
      ignore (Llvm.build_store next_idx counter_ptr builder);
      ignore (Llvm.build_br loop_bb builder);

      (* Continue after loop *)
      Llvm.position_at_end end_bb builder
    in

    (* If we have a global task index, wrap in a loop, otherwise process
       normally *)
    (match !has_global_task_idx with
    | Some global_idx_var -> process_with_loop global_idx_var
    | None -> process_instructions lowered_ir.instructions);

    (* Return void *)
    ignore (Llvm.build_ret_void builder);

    (* Verify and optimize *)
    (try Llvm_analysis.assert_valid_function fn
     with e ->
       Printf.eprintf "Function validation failed: %s\n" (Printexc.to_string e);
       Printf.eprintf "Module:\n%s\n" (Llvm.string_of_llmodule module_);
       raise e);

    (* Return the module and the unique function name - module will be added to
       engine *)
    (module_, kernel_name)

  let compile ~device_info ~source_code ~options:_ =
    (* source_code is the kernel name from Renderer.render *)
    let original_kernel_name = source_code in
    (* Get the lowered IR that Renderer stored *)
    match Hashtbl.find_opt Renderer.pending_kernels original_kernel_name with
    | None ->
        Error
          (Printf.sprintf "No lowered IR found for kernel '%s'"
             original_kernel_name)
    | Some lowered_ir ->
        (* Compile the lowered IR with unique name *)
        let module_, unique_fn_name =
          compile_kernel device_info lowered_ir original_kernel_name
        in
        Hashtbl.remove Renderer.pending_kernels original_kernel_name;
        (* Add the compiled module to the execution engine - it takes
           ownership *)
        Llvm_executionengine.add_module module_ device_info.engine;
        (* Store both names - unique for internal use, original as entry
           point *)
        let native_artifact = { function_names = [ unique_fn_name ] } in
        (* The entry point is the original name that will be requested *)
        let entry_points = [ original_kernel_name ] in
        Ok { Backend_intf.native_artifact; entry_points }
end

(* ───── Runtime ───── *)

module Runtime = struct
  let allocate_buffer ~device_info:_ ~size_in_bytes ~dtype =
    let ptr =
      if size_in_bytes > 0 then
        let raw_ptr = foreign_malloc (Unsigned.Size_t.of_int size_in_bytes) in
        Ctypes.raw_address_of_ptr raw_ptr
      else Nativeint.zero
    in
    let native_buffer =
      { ptr; size_bytes = size_in_bytes; dtype = Ir.Dtype.Any_Dtype dtype }
    in
    Ok { Backend_intf.native_buffer; size_in_bytes; dtype }

  let copy_to_device ~dest_buffer ~host_data ~host_data_offset_bytes
      ~copy_size_bytes =
    (if copy_size_bytes > 0 then
       let dest_ptr =
         Ctypes.(ptr_of_raw_address dest_buffer.Backend_intf.native_buffer.ptr)
       in
       let src_ptr =
         Ctypes.ptr_of_raw_address
           (Nativeint.add host_data (Nativeint.of_int host_data_offset_bytes))
       in
       ignore
         (foreign_memcpy dest_ptr src_ptr
            (Unsigned.Size_t.of_int copy_size_bytes)));
    Ok ()

  let copy_from_device ~src_buffer ~host_dest_ptr ~device_data_offset_bytes
      ~copy_size_bytes =
    (if copy_size_bytes > 0 then
       let src_ptr =
         Ctypes.ptr_of_raw_address
           (Nativeint.add src_buffer.Backend_intf.native_buffer.ptr
              (Nativeint.of_int device_data_offset_bytes))
       in
       let dest_ptr = Ctypes.ptr_of_raw_address host_dest_ptr in
       ignore
         (foreign_memcpy dest_ptr src_ptr
            (Unsigned.Size_t.of_int copy_size_bytes)));
    Ok ()

  let get_kernel ~artifact:_ ~kernel_name =
    (* Check if we have a mapping for this kernel name *)
    match Hashtbl.find_opt Compiler.kernel_name_map kernel_name with
    | Some unique_name ->
        (* Use the unique name for execution *)
        let native_kernel = { name = unique_name } in
        Ok { Backend_intf.native_kernel; name = kernel_name }
    | None -> Error (Printf.sprintf "Kernel '%s' not found" kernel_name)

  let launch_kernel ?local_dims:_ ~device_info ~global_dims ~args kernel =
    (* Get function pointer from JIT *)
    let engine = device_info.engine in

    (* Convert device buffers to pointers *)
    let ptr_list =
      List.map
        (fun arg ->
          match arg with
          | Backend_intf.Any_Device_Buffer buf ->
              Ctypes.ptr_of_raw_address buf.native_buffer.ptr)
        args
    in

    (* Get the size from global_dims - use first dimension *)
    let size = global_dims.(0) in

    (* Call function based on number of arguments (plus size parameter) *)
    let open Ctypes in
    (match ptr_list with
    | [ p1 ] ->
        let fn_ptr =
          Llvm_executionengine.get_function_address
            kernel.Backend_intf.native_kernel.name
            (Foreign.funptr (ptr void @-> int @-> returning void))
            engine
        in
        fn_ptr p1 size
    | [ p1; p2 ] ->
        let fn_ptr =
          Llvm_executionengine.get_function_address
            kernel.Backend_intf.native_kernel.name
            (Foreign.funptr (ptr void @-> ptr void @-> int @-> returning void))
            engine
        in
        fn_ptr p1 p2 size
    | [ p1; p2; p3 ] ->
        let fn_ptr =
          Llvm_executionengine.get_function_address
            kernel.Backend_intf.native_kernel.name
            (Foreign.funptr
               (ptr void @-> ptr void @-> ptr void @-> int @-> returning void))
            engine
        in
        fn_ptr p1 p2 p3 size
    | [ p1; p2; p3; p4 ] ->
        let fn_ptr =
          Llvm_executionengine.get_function_address
            kernel.Backend_intf.native_kernel.name
            (Foreign.funptr
               (ptr void @-> ptr void @-> ptr void @-> ptr void @-> int
              @-> returning void))
            engine
        in
        fn_ptr p1 p2 p3 p4 size
    | [ p1; p2; p3; p4; p5 ] ->
        let fn_ptr =
          Llvm_executionengine.get_function_address
            kernel.Backend_intf.native_kernel.name
            (Foreign.funptr
               (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void
              @-> int @-> returning void))
            engine
        in
        fn_ptr p1 p2 p3 p4 p5 size
    | [ p1; p2; p3; p4; p5; p6 ] ->
        let fn_ptr =
          Llvm_executionengine.get_function_address
            kernel.Backend_intf.native_kernel.name
            (Foreign.funptr
               (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void
              @-> ptr void @-> int @-> returning void))
            engine
        in
        fn_ptr p1 p2 p3 p4 p5 p6 size
    | _ ->
        failwith
          (Printf.sprintf "Unsupported number of arguments: %d"
             (List.length args)));
    Ok ()

  let synchronize ~device_info:_ =
    (* CPU execution is synchronous *)
    ()
end

(* Module name *)
let name = "llvm"
