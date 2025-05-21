open Ir

let foreign_memcpy =
  Foreign.foreign "memcpy"
    Ctypes.(ptr void @-> ptr void @-> size_t @-> returning (ptr void))

let memcpy_bytes ~dst ~src ~len =
  if len > 0 then ignore (foreign_memcpy dst src (Unsigned.Size_t.of_int len))

module Device_info = struct
  type t = {
    device : Metal.Device.t;
    queue : Metal.CommandQueue.t;
    attributes : Metal.Device.attributes;
  }

  let get_default () =
    let device = Metal.Device.create_system_default () in
    let queue = Metal.CommandQueue.on_device device in
    let attributes = Metal.Device.get_attributes device in
    { device; queue; attributes }

  let max_shared_memory t =
    Unsigned.ULong.to_int t.attributes.max_threadgroup_memory_length

  let max_workgroup_size t =
    let m = t.attributes.max_threads_per_threadgroup in
    [| m.width; m.height; m.depth |]

  let supports_dtype _t (Dtype.Any_Dtype dt) =
    match dt with
    | Dtype.Float32 | Dtype.Int32 | Dtype.Bool | Dtype.Uint8 | Dtype.Unit ->
        true

  let renderer_float4_str _ = Some "float4"
  let renderer_smem_prefix _ = "threadgroup "

  let renderer_barrier_str _ =
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
end

type device_buffer_native = Metal.Buffer.t

type ('a_elt, 'b_layout_phantom) device_buffer = {
  native_buffer : device_buffer_native;
  size_in_bytes : int;
  dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
  device_info : Device_info.t;
}

type any_device_buffer =
  | Any_Device_Buffer : ('a, 'b) device_buffer -> any_device_buffer
[@@unboxed]

type compiled_artifact_native = Metal.Library.t

type compiled_artifact = {
  native_artifact : compiled_artifact_native;
  entry_points : string list;
}

type callable_kernel_native = Metal.ComputePipelineState.t

type callable_kernel = {
  native_kernel : callable_kernel_native;
  name : string;
  device_info : Device_info.t;
}

module Renderer = struct
  let render_dtype (Dtype.Any_Dtype dt) : string =
    match dt with
    | Dtype.Float32 -> "float"
    | Dtype.Int32 -> "int"
    | Dtype.Bool -> "bool"
    | Dtype.Uint8 -> "uchar"
    | Dtype.Unit -> "void"

  let render_const_value (type a b) (dt : (a, b) Dtype.t) (v : a) : string =
    match dt with
    | Dtype.Float32 -> Printf.sprintf "%gf" v
    | Dtype.Int32 -> string_of_int v
    | Dtype.Bool -> if v then "true" else "false"
    | Dtype.Uint8 -> string_of_int v
    | Dtype.Unit -> "/*unit_val*/"

  let default_value_str_for_dtype (Dtype.Any_Dtype dt) =
    match dt with
    | Dtype.Float32 -> "0.0f"
    | Dtype.Int32 -> "0"
    | Dtype.Bool -> "false"
    | Dtype.Uint8 -> "0"
    | Dtype.Unit -> ""

  let render_scalar_alu_op (op : Ir.Lowered.scalar_alu_op_type) : string =
    match op with
    | Ir.Lowered.Scalar_Add -> "+"
    | Ir.Lowered.Scalar_Mul -> "*"
    | Ir.Lowered.Scalar_Max -> "fmax"
    | Ir.Lowered.Scalar_CmpLt -> "<"

  let render_special_index_kind (k : Ir.Special_index_kind.t) : string =
    match k with
    | Ir.Special_index_kind.Global_task_idx dim ->
        Printf.sprintf "gtid.%c" [| 'x'; 'y'; 'z' |].(dim)
    | Ir.Special_index_kind.Local_thread_idx dim ->
        Printf.sprintf "lid.%c" [| 'x'; 'y'; 'z' |].(dim)
    | Ir.Special_index_kind.Workgroup_idx dim ->
        Printf.sprintf "gid.%c" [| 'x'; 'y'; 'z' |].(dim)

  let render_instruction (Ir.Lowered.Any_Instruction instr_node)
      (_vars_meta : (Ir.Var.t, Ir.var_metadata) Hashtbl.t)
      (var_names : (Ir.Var.t, string) Hashtbl.t) (smem_prefix : string)
      (indent_level : int ref) : string =
    let get_var_name v = Hashtbl.find var_names v in
    let current_indent = String.make (!indent_level * 2) ' ' in
    match instr_node with
    | Ir.Lowered.LI_Buffer { dtype; size_in_elements; out_var } ->
        Printf.sprintf "%s%s%s %s[%d];" current_indent smem_prefix
          (render_dtype (Dtype.Any_Dtype dtype))
          (get_var_name out_var) size_in_elements
    | Ir.Lowered.LI_Const_Scalar { value; dtype; out_var } ->
        Printf.sprintf "%s%s %s = %s;" current_indent
          (render_dtype (Dtype.Any_Dtype dtype))
          (get_var_name out_var)
          (render_const_value dtype value)
    | Ir.Lowered.LI_Range { upper_bound_exclusive; out_var; _ } ->
        let s =
          Printf.sprintf "%sfor (int %s = 0; %s < %s; ++%s) {" current_indent
            (get_var_name out_var) (get_var_name out_var)
            (get_var_name upper_bound_exclusive)
            (get_var_name out_var)
        in
        incr indent_level;
        s
    | Ir.Lowered.LI_End_Range ->
        decr indent_level;
        Printf.sprintf "%s}" (String.make (!indent_level * 2) ' ')
    | Ir.Lowered.LI_Special_Index { kind; out_var; _ } ->
        Printf.sprintf "%suint %s = %s;" current_indent (get_var_name out_var)
          (render_special_index_kind kind)
    | Ir.Lowered.LI_Load
        { buffer_source_var; indices_vars; valid_mask_var; out_var; dtype } -> (
        let idx_str =
          match indices_vars with
          | [ iv ] -> get_var_name iv
          | _ -> "/*complex_idx*/0"
        in
        let load_expr =
          Printf.sprintf "%s[%s]" (get_var_name buffer_source_var) idx_str
        in
        match valid_mask_var with
        | Some vm_var ->
            Printf.sprintf "%s%s %s = (%s != 0) ? (%s) : %s;" current_indent
              (render_dtype (Dtype.Any_Dtype dtype))
              (get_var_name out_var) (get_var_name vm_var) load_expr
              (default_value_str_for_dtype (Dtype.Any_Dtype dtype))
        | None ->
            Printf.sprintf "%s%s %s = %s;" current_indent
              (render_dtype (Dtype.Any_Dtype dtype))
              (get_var_name out_var) load_expr)
    | Ir.Lowered.LI_Store
        {
          buffer_target_var;
          indices_vars;
          scalar_value_to_store_var;
          valid_mask_var;
          _;
        } -> (
        let idx_str =
          match indices_vars with
          | [ iv ] -> get_var_name iv
          | _ -> "/*complex_idx*/0"
        in
        let base_store =
          Printf.sprintf "%s[%s] = %s;"
            (get_var_name buffer_target_var)
            idx_str
            (get_var_name scalar_value_to_store_var)
        in
        match valid_mask_var with
        | Some vm_var ->
            Printf.sprintf "%sif (%s != 0) { %s }" current_indent
              (get_var_name vm_var) base_store
        | None -> current_indent ^ base_store)
    | Ir.Lowered.LI_Scalar_ALU { op_type; inputs_vars; out_var; dtype } ->
        let op_str = render_scalar_alu_op op_type in
        let args_str =
          match inputs_vars with
          | [ a ] ->
              Printf.sprintf "%s%s" op_str (get_var_name a)
              (* Assuming unary ops are prefix *)
          | [ a; b ] ->
              Printf.sprintf "%s %s %s" (get_var_name a) op_str (get_var_name b)
          | _ -> "/*ALU_err*/"
        in
        Printf.sprintf "%s%s %s = %s;" current_indent
          (render_dtype (Dtype.Any_Dtype dtype))
          (get_var_name out_var) args_str

  let render ~device_info ~lowered_ir ~kernel_name =
    let buf = Buffer.create 1024 in
    Buffer.add_string buf "#include <metal_stdlib>\nusing namespace metal;\n\n";

    let var_names : (Ir.Var.t, string) Hashtbl.t =
      Hashtbl.create (Hashtbl.length lowered_ir.Ir.Lowered.vars_metadata)
    in
    let var_idx_counter = ref 0 in
    let ensure_var_name v hint_prefix =
      if not (Hashtbl.mem var_names v) then (
        Hashtbl.add var_names v
          (Printf.sprintf "%s_v%d" hint_prefix !var_idx_counter);
        incr var_idx_counter)
    in

    (* Kernel arguments (inputs then outputs) *)
    let ordered_kernel_args_ll_vars =
      lowered_ir.Ir.Lowered.kernel_input_vars
      @ lowered_ir.Ir.Lowered.kernel_output_vars
    in
    List.iter (fun v -> ensure_var_name v "arg") ordered_kernel_args_ll_vars;

    (* Other vars from metadata (e.g. for constants, intermediate scalars) *)
    Hashtbl.iter
      (fun v (meta : var_metadata) ->
        ensure_var_name v (Ir.Dtype.any_to_string meta.dtype))
      lowered_ir.Ir.Lowered.vars_metadata;

    let arg_decls =
      List.mapi
        (fun i var_id ->
          let meta = Hashtbl.find lowered_ir.Ir.Lowered.vars_metadata var_id in
          (* Assumes all arg vars have metadata *)
          Printf.sprintf "device %s* %s [[buffer(%d)]]"
            (render_dtype meta.dtype)
            (Hashtbl.find var_names var_id)
            i)
        ordered_kernel_args_ll_vars
    in
    let special_arg_decls =
      [
        "uint3 gtid [[thread_position_in_grid]]";
        "uint3 lid [[thread_position_in_threadgroup]]";
        "uint3 gid [[threadgroup_position_in_grid]]";
      ]
    in
    Buffer.add_string buf
      (Printf.sprintf "kernel void %s(\n  %s\n) {\n" kernel_name
         (String.concat ",\n  " (arg_decls @ special_arg_decls)));

    let indent_level = ref 1 in
    (* Start indenting inside kernel body *)
    List.iter
      (fun instr_any ->
        let line =
          render_instruction instr_any lowered_ir.Ir.Lowered.vars_metadata
            var_names
            (Device_info.renderer_smem_prefix device_info)
            indent_level
        in
        Buffer.add_string buf (line ^ "\n"))
      lowered_ir.Ir.Lowered.instructions;

    Buffer.add_string buf "}\n";
    Buffer.contents buf
end

module Compiler = struct
  type compile_options = Metal.CompileOptions.t

  let default_options (_dev_info : Device_info.t) = Metal.CompileOptions.init ()
  (* Simplification: fast_math can be default in Metal or set by user if
     needed *)

  let compile ~device_info ~source_code ~options =
    try
      let library =
        Metal.Library.on_device device_info.Device_info.device
          ~source:source_code options
      in
      let function_names =
        Metal.Library.get_function_names library |> Array.to_list
      in
      Ok { native_artifact = library; entry_points = function_names }
    with ex ->
      Error
        (Printf.sprintf "Metal compilation failed: %s" (Printexc.to_string ex))
end

module Runtime = struct
  let allocate_buffer ~device_info ~size_in_bytes ~dtype =
    let actual_size = if size_in_bytes = 0 then 1 else size_in_bytes in
    let res_opts =
      Metal.ResourceOptions.make
        ~storage_mode:Metal.ResourceOptions.storage_mode_shared ()
    in
    try
      Ok
        {
          native_buffer =
            Metal.Buffer.on_device device_info.Device_info.device
              ~length:actual_size res_opts;
          size_in_bytes;
          dtype;
          device_info;
        }
    with exn ->
      Error
        (Printf.sprintf "Metal alloc failed (size %d): %s" actual_size
           (Printexc.to_string exn))

  let deallocate_buffer (_buffer : ('a, 'b) device_buffer) =
    () (* ARC handles release *)

  let copy_to_device ~dest_buffer ~host_data ~host_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes = 0 then Ok ()
    else if dest_buffer.size_in_bytes < copy_size_bytes then
      Error "Metal: copy_to_device size exceeds buffer capacity."
    else
      let contents_ptr = Metal.Buffer.contents dest_buffer.native_buffer in
      if Ctypes.is_null contents_ptr then
        Error "Metal: copy_to_device: MTLBuffer contents pointer is null."
      else
        try
          let host_src_ptr =
            Ctypes.(
              to_voidp (ptr_of_raw_address host_data +@ host_data_offset_bytes))
          in
          memcpy_bytes ~dst:contents_ptr ~src:host_src_ptr ~len:copy_size_bytes;
          Ok ()
        with ex ->
          Error
            (Printf.sprintf "Metal: memcpy to device failed: %s"
               (Printexc.to_string ex))

  let copy_from_device ~src_buffer ~host_dest_ptr ~device_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes = 0 then Ok ()
    else if
      src_buffer.size_in_bytes < device_data_offset_bytes + copy_size_bytes
    then Error "Metal: copy_from_device read range exceeds buffer capacity."
    else
      let contents_ptr = Metal.Buffer.contents src_buffer.native_buffer in
      if Ctypes.is_null contents_ptr then
        Error "Metal: copy_from_device: MTLBuffer contents pointer is null."
      else
        try
          let device_src_ptr =
            Ctypes.(contents_ptr +@ device_data_offset_bytes)
          in
          memcpy_bytes
            ~dst:(Ctypes.to_voidp (Ctypes.ptr_of_raw_address host_dest_ptr))
            ~src:device_src_ptr ~len:copy_size_bytes;
          Ok ()
        with ex ->
          Error
            (Printf.sprintf "Metal: memcpy from device failed: %s"
               (Printexc.to_string ex))

  let get_kernel ~artifact ~kernel_name =
    try
      let func =
        Metal.Library.new_function_with_name artifact.native_artifact
          kernel_name
      in
      let device_obj = Metal.Library.get_device artifact.native_artifact in
      let pso, _ =
        Metal.ComputePipelineState.on_device_with_function device_obj func
      in
      let queue = Metal.CommandQueue.on_device device_obj in
      let attributes = Metal.Device.get_attributes device_obj in
      let di_for_kernel =
        { Device_info.device = device_obj; queue; attributes }
      in
      Ok
        { native_kernel = pso; name = kernel_name; device_info = di_for_kernel }
    with ex ->
      Error
        (Printf.sprintf "Metal get_kernel '%s' failed: %s" kernel_name
           (Printexc.to_string ex))

  let launch_kernel ~kernel ~global_dims ~local_dims ~args =
    let ( let* ) = Result.bind in
    let device_info = kernel.device_info in
    let cbuf = Metal.CommandBuffer.on_queue device_info.queue in
    let enc = Metal.ComputeCommandEncoder.on_buffer cbuf in
    Metal.ComputeCommandEncoder.set_compute_pipeline_state enc
      kernel.native_kernel;
    List.iteri
      (fun i (Any_Device_Buffer backend_buffer) ->
        Metal.ComputeCommandEncoder.set_buffer enc ~offset:0 ~index:i
          backend_buffer.native_buffer)
      args;

    let* lx, ly, lz =
      match local_dims with
      | Some ld when Array.length ld = 3 -> Ok (ld.(0), ld.(1), ld.(2))
      | Some _ ->
          Error
            "Metal launch_kernel: local_dims must have 3 elements or be None."
      | None ->
          let max_total =
            Metal.ComputePipelineState.get_max_total_threads_per_threadgroup
              kernel.native_kernel
          in
          let threads_x =
            if Array.length global_dims > 0 && global_dims.(0) > 0 then
              min max_total global_dims.(0)
            else min max_total 1
          in
          Ok (threads_x, 1, 1)
    in
    if lx <= 0 || ly <= 0 || lz <= 0 then
      Error "Metal launch_kernel: Deduced local_dims component is non-positive."
    else if Array.length global_dims <> 3 then
      Error "Metal launch_kernel: global_dims must have 3 elements."
    else
      let gx, gy, gz = (global_dims.(0), global_dims.(1), global_dims.(2)) in
      if gx <= 0 || gy <= 0 || gz <= 0 then
        Error "Metal launch_kernel: global_dims component is non-positive."
      else
        let grid_size = Metal.Size.{ width = gx; height = gy; depth = gz } in
        let group_size = Metal.Size.{ width = lx; height = ly; depth = lz } in
        Metal.ComputeCommandEncoder.dispatch_threadgroups enc
          ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;
        Metal.ComputeCommandEncoder.end_encoding enc;
        Metal.CommandBuffer.commit cbuf;
        Metal.CommandBuffer.wait_until_completed cbuf;
        match Metal.CommandBuffer.get_error cbuf with
        | Some err_msg ->
            Error
              (Printf.sprintf "Metal: Kernel '%s' failed: %s" kernel.name
                 err_msg)
        | None -> Ok ()

  let synchronize ~device_info =
    let cbuf = Metal.CommandBuffer.on_queue device_info.Device_info.queue in
    Metal.CommandBuffer.commit cbuf;
    Metal.CommandBuffer.wait_until_completed cbuf
end

let name = "METAL_GPU"
