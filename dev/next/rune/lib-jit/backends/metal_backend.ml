(* lib/jit/metal_backend.ml *)
open Nx_core
open Backend_intf

(* Minimal ObjC FFI bindings using ocaml-objc *)
module Objc_runtime = Objc.Runtime
module Objc_id = Objc.Id

let sel name = Objc_runtime.selector name
let ( !! ) = Objc_id.send_void
let ( !% ) = Objc_id.send
let ( !? ) = Objc_id.send_maybe

let nsstring_of_string s =
  let ns_string_cls = Objc_runtime.get_class "NSString" in
  (ns_string_cls !%sel "stringWithUTF8String:" : string -> Objc_id.t) s

let string_of_nsstring ns_s =
  let c_str : string = (ns_s !%sel "UTF8String" : unit -> string) () in
  c_str

(* NSError utility - rudimentary *)
let get_error_description_maybe (error_ptr : Objc_id.t Ctypes.ptr) :
    string option =
  if not (Ctypes.is_null error_ptr) then
    let error_obj = Ctypes.(!@error_ptr) in
    if Objc_id.is_null error_obj then None
    else
      Some
        (string_of_nsstring
           (error_obj !%sel "localizedDescription" : unit -> Objc_id.t)
           ())
  else None

module Metal_types = struct
  (* MTLResourceOptions *)
  let _MTLResourceStorageModeShared =
    0 (* Not used directly, part of a bitmask *)

  let _MTLResourceCPUCacheModeDefaultCache = 0
  let _MTLResourceHazardTrackingModeUntracked = 1 (* (1 << 2) *)

  let mtLResourceCPUCacheModeDefaultCache =
    Unsigned.ULong.of_int _MTLResourceCPUCacheModeDefaultCache

  (* Add others like MTLResourceStorageModeShared if needed directly via
     options *)
  let mtLResourceHazardTrackingModeUntracked =
    Unsigned.ULong.of_int _MTLResourceHazardTrackingModeUntracked
end

module Device_info = struct
  type t = {
    metal_device : Objc_id.t; (* MTLDevice *)
    metal_queue : Objc_id.t; (* MTLCommandQueue *)
    name : string;
  }

  let get_default () =
    let device =
      (Objc_runtime.get_class "MTLDevice" !%sel "createSystemDefaultDevice"
        : unit -> Objc_id.t option)
        ()
    in
    match device with
    | None -> failwith "Metal: No GPU found."
    | Some dev ->
        let queue = (dev !%sel "newCommandQueue" : unit -> Objc_id.t) () in
        let name_ns = (dev !%sel "name" : unit -> Objc_id.t) () in
        {
          metal_device = dev;
          metal_queue = queue;
          name = string_of_nsstring name_ns;
        }

  let max_shared_memory device_info =
    let length : Unsigned.ULong.t =
      (device_info.metal_device !%sel "maxThreadgroupMemoryLength"
        : unit -> Unsigned.ULong.t)
        ()
    in
    Unsigned.ULong.to_int
      length (* May truncate if > 2GB, but typical shared mem is KBs *)

  let max_workgroup_size device_info =
    (* This is a bit tricky. MTLDevice.maxThreadsPerThreadgroup gives total. We
       need to query it from a compiled pipeline state for optimal dimensions,
       but we don't have one here. Let's use a common safe default or query
       total. *)
    let max_total_threads_ul : Unsigned.ULong.t =
      (device_info.metal_device !%sel "maxThreadsPerThreadgroup"
        : unit -> Unsigned.ULong.t)
        ()
    in
    let max_total_threads = Unsigned.ULong.to_int max_total_threads_ul in
    (* Metal typically has threadExecutionWidth (warp size) of 32. A common 3D
       configuration might be [1024; 1; 1] or derived from max_total_threads.
       Tinygrad uses (1024, 1024, 64) as local_max, but that's for CUDA usually.
       Let's simplify to total threads and let launch figure it out or use a
       default. For Metal, maxThreadsPerThreadgroup is the product of
       dimensions. Individual dimension limits are higher, e.g., 1024,1024,1024
       for width,height,depth but product <= maxThreadsPerThreadgroup. *)
    [| max_total_threads; max_total_threads; max_total_threads |]
  (* Simplified; actual limits are per-dimension + total product *)

  let supports_dtype _ (Dtype.Any_dtype dt) =
    match dt with
    | Dtype.Float16 | Dtype.Float32 | Dtype.Int8 | Dtype.Int16 | Dtype.Int32
    | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint32 | Dtype.Bool ->
        true
    | Dtype.Float64 | Dtype.Int64
    | Dtype.Uint64
      (* Metal doesn't directly support 64-bit types in kernels widely, or
         bfloat16 *)
    | Dtype.Bfloat16 ->
        false
    | _ -> false (* Other complex types not typically supported in shaders *)

  let renderer_float4_str _ = Some "float4" (* Metal uses float4, etc. *)
  let renderer_smem_prefix _ = "threadgroup "

  let renderer_barrier_str _ =
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
end

type device_buffer_native = Objc_id.t (* MTLBuffer *)

type ('a, 'b) device_buffer = {
  native_buffer : device_buffer_native;
  size_in_bytes : int;
  dtype : ('a, 'b) Dtype.t;
  device_info : Device_info.t;
}

type compiled_artifact_native = Objc_id.t (* MTLComputePipelineState *)

type compiled_artifact = {
  native_artifact : compiled_artifact_native;
  entry_points : string list; (* Should be just one, the kernel_name *)
}

type callable_kernel_native =
  Objc_id.t (* Same as compiled_artifact_native for Metal *)

type callable_kernel = { native_kernel : callable_kernel_native; name : string }

(* --- Helper for Metal structure creation --- *)
(* Equivalent to to_struct in tinygrad's metal.py *)
module Mtl_structs = struct
  type mtlSize = {
    width : Unsigned.ULong.t;
    height : Unsigned.ULong.t;
    depth : Unsigned.ULong.t;
  }
  [@@deriving ctype]

  let mtlSize_t = Ctypes.structure "MTLSize"

  let () =
    Ctypes.field mtlSize_t "width" Ctypes.ulong (fun s -> s.width) |> ignore

  let () =
    Ctypes.field mtlSize_t "height" Ctypes.ulong (fun s -> s.height) |> ignore

  let () =
    Ctypes.field mtlSize_t "depth" Ctypes.ulong (fun s -> s.depth) |> ignore

  let () = Ctypes.seal mtlSize_t

  let make_mtl_size ~width ~height ~depth =
    let sz = Ctypes.make mtlSize_t in
    Ctypes.setf sz width (Unsigned.ULong.of_int width);
    Ctypes.setf sz height (Unsigned.ULong.of_int height);
    Ctypes.setf sz depth (Unsigned.ULong.of_int depth);
    sz
end

module Renderer = struct
  (* Based on tinygrad.renderer.cstyle.MetalRenderer *)

  let render_dtype (Dtype.Any_dtype dt) : string =
    match dt with
    | Dtype.Float32 -> "float"
    | Dtype.Float16 -> "half"
    | Dtype.Int32 -> "int"
    | Dtype.Uint32 -> "uint"
    | Dtype.Int16 -> "short"
    | Dtype.Uint16 -> "ushort"
    | Dtype.Int8 -> "char"
    | Dtype.Uint8 -> "uchar"
    | Dtype.Bool -> "bool"
    (* Vector types - tinygrad often generates these via UOp.VECTORIZE *)
    (* This renderer would need to handle UOp.VECTORIZE to generate float2, int4 etc. *)
    (* Or assume the Ir.Lowered already has vector types if Metal needs them explicitly *)
    | _ -> failwith ("MetalRenderer: Unsupported Dtype: " ^ Dtype.to_string dt)

  let render_scalar_alu_op (op_type : Ir.Lowered.scalar_alu_op_type) : string =
    match op_type with
    | Ir.Lowered.Scalar_Add -> "+"
    | Ir.Lowered.Scalar_Mul -> "*"
    | Ir.Lowered.Scalar_Max -> "fmax" (* Metal built-in *)
    | Ir.Lowered.Scalar_CmpLt -> "<"
  (* Add more ops: Div -> "/", Sin -> "sin", Exp2 -> "exp2", Log2 -> "log2",
     Sqrt -> "sqrt" *)

  let render_instruction (instr : Ir.Lowered.any_instruction)
      (vars_meta : (Ir.Var.t, Ir.var_metadata) Hashtbl.t)
      (var_names : (Ir.Var.t, string) Hashtbl.t) : string =
    let get_var_name v = Hashtbl.find var_names v in
    let get_var_meta v = Hashtbl.find vars_meta v in
    match instr with
    | Ir.Lowered.Any_Instruction
        (LI_Const_Scalar { value; dtype = dt_val; out_var }) ->
        let (Ir.Any_Dtype dt_any) = (get_var_meta out_var).dtype in
        let val_str = Dtype.value_to_string dt_any value in
        Printf.sprintf "%s %s = %s;"
          (render_dtype (Ir.Any_Dtype dt_val))
          (get_var_name out_var) val_str
    | Ir.Lowered.Any_Instruction
        (LI_Buffer { dtype = dt_buf; size_in_elements = _; out_var }) ->
        (* In MSL, buffers passed to kernels are pointers. If this is for a
           threadgroup buffer, it's different. Assume kernel arg for now. *)
        Printf.sprintf "device %s* %s;"
          (render_dtype (Ir.Any_Dtype dt_buf))
          (get_var_name out_var)
    | Ir.Lowered.Any_Instruction
        (LI_Range { name_hint = _; upper_bound_exclusive; out_var }) ->
        Printf.sprintf "for (int %s = 0; %s < %s; ++%s) {"
          (get_var_name out_var) (get_var_name out_var)
          (get_var_name upper_bound_exclusive)
          (* Assume upper_bound_exclusive is an int var *)
          (get_var_name out_var)
    | Ir.Lowered.Any_Instruction
        (LI_Special_Index { name_hint = _; kind; out_var }) ->
        let msl_idx_str =
          match kind with
          | Backend_intf.Global_task_idx 0 -> "gid.x"
          | Backend_intf.Global_task_idx 1 -> "gid.y"
          | Backend_intf.Global_task_idx 2 -> "gid.z"
          | Backend_intf.Local_thread_idx 0 -> "lid.x"
          | Backend_intf.Local_thread_idx 1 -> "lid.y"
          | Backend_intf.Local_thread_idx 2 -> "lid.z"
          | Backend_intf.Workgroup_idx 0 ->
              "tgid.x" (* threadgroup_position_in_grid *)
          | Backend_intf.Workgroup_idx 1 -> "tgid.y"
          | Backend_intf.Workgroup_idx 2 -> "tgid.z"
          (* group_id is an alias for threadgroup_position_in_grid *)
          | _ -> failwith "MetalRenderer: Unsupported LI_Special_Index kind"
        in
        Printf.sprintf "uint %s = %s;" (get_var_name out_var) msl_idx_str
    | Ir.Lowered.Any_Instruction
        (LI_Load
           {
             buffer_source_var;
             indices_vars;
             valid_mask_var = _;
             out_var;
             dtype = dt_load;
           }) ->
        (* Simplified: assumes 1D access or that indices_vars represents a
           flattened index *)
        let idx_str =
          match indices_vars with
          | [ iv ] -> get_var_name iv
          | _ ->
              "/* complex_idx_calc */0" (* Placeholder for multi-D to 1D calc *)
        in
        Printf.sprintf "%s %s = %s[%s];"
          (render_dtype (Ir.Any_Dtype dt_load))
          (get_var_name out_var)
          (get_var_name buffer_source_var)
          idx_str
    | Ir.Lowered.Any_Instruction
        (LI_Store
           {
             buffer_target_var;
             indices_vars;
             scalar_value_to_store_var;
             valid_mask_var = _;
           }) ->
        let idx_str =
          match indices_vars with
          | [ iv ] -> get_var_name iv
          | _ -> "/* complex_idx_calc */0"
        in
        Printf.sprintf "%s[%s] = %s;"
          (get_var_name buffer_target_var)
          idx_str
          (get_var_name scalar_value_to_store_var)
    | Ir.Lowered.Any_Instruction
        (LI_Scalar_ALU { op_type; inputs_vars; out_var; dtype = dt_alu }) ->
        let op_str = render_scalar_alu_op op_type in
        let args_str =
          String.concat
            (Printf.sprintf " %s " op_str)
            (List.map get_var_name inputs_vars)
        in
        Printf.sprintf "%s %s = %s;"
          (render_dtype (Ir.Any_Dtype dt_alu))
          (get_var_name out_var) args_str

  let render ~device_info:_ ~lowered_ir ~kernel_name =
    let buffer = Buffer.create 1024 in
    Buffer.add_string buffer "#include <metal_stdlib>\n";
    Buffer.add_string buffer "using namespace metal;\n\n";

    let var_names =
      Hashtbl.create (Hashtbl.length lowered_ir.Ir.Lowered.vars_metadata)
    in
    let next_var_idx = ref 0 in
    let get_fresh_var_name (v : Ir.Var.t) (hint : string) : string =
      let name = Printf.sprintf "%s_%d" hint !next_var_idx in
      incr next_var_idx;
      Hashtbl.add var_names v name;
      name
    in
    Hashtbl.iter
      (fun v meta -> ignore (get_fresh_var_name v "v"))
      lowered_ir.Ir.Lowered.vars_metadata;

    (* Kernel signature *)
    (* Metal needs explicit attribute for buffer bindings, e.g. [[buffer(0)]] *)
    let arg_idx_map = Hashtbl.create 10 in
    let current_arg_idx = ref 0 in

    let get_arg_binding_idx var_id =
      match Hashtbl.find_opt arg_idx_map var_id with
      | Some idx -> idx
      | None ->
          let idx = !current_arg_idx in
          Hashtbl.add arg_idx_map var_id idx;
          incr current_arg_idx;
          idx
    in

    let kernel_args_str =
      lowered_ir.Ir.Lowered.kernel_input_vars
      @ lowered_ir.Ir.Lowered.kernel_output_vars
      |> List.sort_uniq Ir.Var.compare (* Ensure consistent order *)
      |> List.map (fun var_id ->
             let meta =
               Hashtbl.find lowered_ir.Ir.Lowered.vars_metadata var_id
             in
             let (Ir.Any_Dtype dt_any) = meta.dtype in
             let binding_idx = get_arg_binding_idx var_id in
             Printf.sprintf "device %s* %s [[buffer(%d)]]"
               (render_dtype (Ir.Any_Dtype dt_any))
               (Hashtbl.find var_names var_id)
               binding_idx)
    in
    (* Add thread/grid position arguments *)
    let special_args =
      [
        "uint3 gid [[threadgroup_position_in_grid]]";
        "uint3 lid [[thread_position_in_threadgroup]]";
        (* Can add others like "uint tid [[thread_index_in_threadgroup]]" if
           needed by Special_Index *)
      ]
    in
    Buffer.add_string buffer
      (Printf.sprintf "kernel void %s(%s) {\n" kernel_name
         (String.concat ", " (kernel_args_str @ special_args)));

    List.iter
      (fun instr_any ->
        let code_line =
          render_instruction instr_any lowered_ir.Ir.Lowered.vars_metadata
            var_names
        in
        Buffer.add_string buffer (Printf.sprintf "  %s\n" code_line))
      lowered_ir.Ir.Lowered.instructions;

    Buffer.add_string buffer "}\n";
    Buffer.contents buffer
end

module Compiler = struct
  type compile_options = {
    fast_math_enabled : bool;
        (* Corresponds to MTLCCompileOptions.fastMathEnabled *)
        (* language_version can be added if specific MSL versions are needed *)
  }

  let default_options (_dev_info : Device_info.t) = { fast_math_enabled = true }

  let compile ~device_info ~source_code ~options =
    let compile_options_obj =
      (Objc_runtime.get_class "MTLCompileOptions" !%sel "new"
        : unit -> Objc_id.t)
        ()
    in
    (compile_options_obj !!sel "setFastMathEnabled:" : bool -> unit)
      options.fast_math_enabled;

    (* Set language version if needed: (compile_options_obj !! sel
       "setLanguageVersion:" : Unsigned.ULong.t -> unit)
       Metal_types.mtlLanguageVersion2_4; *)
    let error_ptr = Ctypes.allocate Objc_id.t Objc_id.null in
    let library : Objc_id.t option =
      (device_info.metal_device !%sel "newLibraryWithSource:options:error:"
        : Objc_id.t -> Objc_id.t -> Objc_id.t Ctypes.ptr -> Objc_id.t option)
        (nsstring_of_string source_code)
        compile_options_obj error_ptr
    in

    match (library, get_error_description_maybe error_ptr) with
    | Some lib, None -> (
        (* Assume kernel_name is the one passed to Renderer, which is usually the main one *)
        (* This is a simplification: JIT artifact might contain multiple kernels if source_code did *)
        let entry_point_name =
          (* Infer from source or require it to be passed. For single kernel compilation, it's fixed. *)
          (* Let's find "kernel void KERNEL_NAME(" pattern *)
          try
            let re =
              Re.Perl.compile_pat "kernel\\s+void\\s+([a-zA-Z0-9_]+)\\s*\\("
            in
            let groups = Re.exec re source_code in
            Re.Group.get groups 1
          with Not_found ->
            "kernel_main" (* Fallback name, this should be robust *)
        in
        let function_obj : Objc_id.t option =
          (lib !?sel "newFunctionWithName:" : Objc_id.t -> Objc_id.t option)
            (nsstring_of_string entry_point_name)
        in
        match function_obj with
        | Some func -> (
            let pso_error_ptr = Ctypes.allocate Objc_id.t Objc_id.null in
            let pipeline_state : Objc_id.t option =
              (device_info.metal_device !%sel
                 "newComputePipelineStateWithFunction:error:"
                : Objc_id.t -> Objc_id.t Ctypes.ptr -> Objc_id.t option)
                func pso_error_ptr
            in
            match
              (pipeline_state, get_error_description_maybe pso_error_ptr)
            with
            | Some pso, None ->
                Ok
                  { native_artifact = pso; entry_points = [ entry_point_name ] }
            | _, Some err_msg ->
                Error
                  (Printf.sprintf "Failed to create pipeline state for %s: %s"
                     entry_point_name err_msg)
            | None, None ->
                Error
                  (Printf.sprintf
                     "Failed to create pipeline state for %s (unknown error, \
                      no error object)."
                     entry_point_name))
        | None ->
            Error
              (Printf.sprintf "Metal function %s not found in library"
                 entry_point_name))
    | _, Some err_msg ->
        Error (Printf.sprintf "Metal library compilation failed: %s" err_msg)
    | None, None ->
        Error
          "Metal library compilation failed (unknown error, no error object)."
end

module Runtime = struct
  let allocate_buffer ~device_info ~size_in_bytes ~dtype =
    if size_in_bytes = 0 then
      (* Metal doesn't like zero-size buffers, but tinygrad sometimes allocates
         them. Return a dummy non-null Objc_id.t if possible or handle
         carefully. For now, let's create a minimal 1-byte buffer if size is 0
         to avoid null. This should align with how tinygrad handles zero-size
         buffers. Or, if the API supports it, return a specific "empty buffer"
         representation. Tinygrad's metal.py uses max(1, length). *)
      let actual_size = max 1 size_in_bytes in
      let buffer : Objc_id.t option =
        (device_info.metal_device !%sel "newBufferWithLength:options:"
          : Unsigned.ULong.t -> Unsigned.ULong.t -> Objc_id.t option)
          (Unsigned.ULong.of_int actual_size)
          Metal_types.mtLResourceCPUCacheModeDefaultCache
        (* Use MTLResourceStorageModeShared for CPU/GPU coherent memory
           typically, or MTLResourceStorageModeManaged/Private for GPU-only with
           explicit copies. Default cache for CPU is often
           MTLResourceStorageModeShared. Tinygrad uses
           MTLResourceStorageModeShared if not M1/M2 (which use default cache)
           Let's use default cache for simplicity. For hazard tracking:
           Metal_types.mtLResourceHazardTrackingModeUntracked *)
      in
      match buffer with
      | Some buf ->
          Ok { native_buffer = buf; size_in_bytes; dtype; device_info }
      | None ->
          Error
            "Metal failed to allocate buffer (newBufferWithLength returned nil)"
    else
      let buffer : Objc_id.t option =
        (device_info.metal_device !%sel "newBufferWithLength:options:"
          : Unsigned.ULong.t -> Unsigned.ULong.t -> Objc_id.t option)
          (Unsigned.ULong.of_int size_in_bytes)
          Metal_types.mtLResourceCPUCacheModeDefaultCache
      in
      match buffer with
      | Some buf ->
          Ok { native_buffer = buf; size_in_bytes; dtype; device_info }
      | None ->
          Error
            "Metal failed to allocate buffer (newBufferWithLength returned nil)"

  let deallocate_buffer (_buffer : ('a, 'b) device_buffer) =
    (* Metal buffers are reference counted by ARC (Automatic Reference Counting)
       within the Objective-C runtime. When the Objc_id.t representing the
       MTLBuffer goes out of scope and its OCaml finalizer runs (if ocaml-objc
       sets one up, or if it's manually released via Objc_id.release), ARC
       decrements the refcount. No explicit deallocate call is usually needed
       from this side unless we manually retained it. *)
    ()

  let copy_to_device ~dest_buffer ~host_data ~host_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes = 0 then Ok () (* No-op for zero bytes *)
    else if copy_size_bytes < 0 || host_data_offset_bytes < 0 then
      Error
        "copy_to_device: copy_size_bytes or host_data_offset_bytes cannot be \
         negative."
      (* Basic check against buffer size, assuming host_data has enough bytes *)
    else if copy_size_bytes > dest_buffer.size_in_bytes then
      Error
        (Printf.sprintf
           "copy_to_device: copy_size_bytes (%d) exceeds destination buffer \
            capacity (%d)."
           copy_size_bytes dest_buffer.size_in_bytes)
    else
      try
        let contents_ptr : Ctypes.voidp =
          (dest_buffer.native_buffer !%sel "contents" : unit -> Ctypes.voidp) ()
        in
        if Ctypes.is_null contents_ptr then
          Error
            "copy_to_device: MTLBuffer contents pointer is null. Buffer might \
             not be CPU accessible (e.g. private storage)."
        else
          let src_ptr_char = Ctypes.ptr_of_raw_address host_data in
          let src_ptr_offset_char =
            Ctypes.(src_ptr_char +@ host_data_offset_bytes)
          in
          Ctypes.memcpy ~dst:contents_ptr
            ~src:(Ctypes.to_voidp src_ptr_offset_char)
            ~len:copy_size_bytes;
          (* If buffer is MTLStorageModeManaged, might need didModifyRange here.
             (dest_buffer.native_buffer !! sel "didModifyRange:" : NSRange.t ->
             unit) (NSRange.make (Unsigned.ULong.of_int 0)
             (Unsigned.ULong.of_int copy_size_bytes)); For Shared storage, this
             is not needed. Assuming shared for simplicity. *)
          Ok ()
      with e ->
        Error
          (Printf.sprintf "copy_to_device memcpy failed: %s"
             (Printexc.to_string e))

  let copy_from_device ~src_buffer ~host_dest_ptr ~device_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes = 0 then Ok ()
    else if copy_size_bytes < 0 || device_data_offset_bytes < 0 then
      Error
        "copy_from_device: copy_size_bytes or device_data_offset_bytes cannot \
         be negative."
    else if
      device_data_offset_bytes + copy_size_bytes > src_buffer.size_in_bytes
    then
      Error
        (Printf.sprintf
           "copy_from_device: copy range [%d, %d) exceeds source buffer size \
            %d."
           device_data_offset_bytes
           (device_data_offset_bytes + copy_size_bytes)
           src_buffer.size_in_bytes)
    else
      try
        let contents_ptr : Ctypes.voidp =
          (src_buffer.native_buffer !%sel "contents" : unit -> Ctypes.voidp) ()
        in
        if Ctypes.is_null contents_ptr then
          Error
            "copy_from_device: MTLBuffer contents pointer is null. Buffer \
             might not be CPU accessible."
        else
          let src_device_ptr_offset =
            Ctypes.(contents_ptr +@ device_data_offset_bytes)
          in
          let dest_ptr_char = Ctypes.ptr_of_raw_address host_dest_ptr in
          Ctypes.memcpy
            ~dst:(Ctypes.to_voidp dest_ptr_char)
            ~src:src_device_ptr_offset ~len:copy_size_bytes;
          Ok ()
      with e ->
        Error
          (Printf.sprintf "copy_from_device memcpy failed: %s"
             (Printexc.to_string e))

  let get_kernel ~artifact ~kernel_name =
    (* In this setup, compiled_artifact's native_artifact *is* the
       MTLComputePipelineState *)
    Ok { native_kernel = artifact.native_artifact; name = kernel_name }

  (* Shared command buffer and encoder state for a sequence of launches, or create per launch *)
  (* For simplicity, let's create per launch for now. A real system might batch. *)
  let launch_kernel ~kernel ~global_dims ~local_dims ~args =
    let device_info =
      (* Need Device_info.t here, assume it's part of kernel or passed *)
      (* This is a bit awkward. Runtime functions take Device_info, but
         launch_kernel takes `kernel`. The kernel or artifact should perhaps
         hold a ref to its Device_info. For now, let's assume we can get it
         globally or via kernel. A better way: pass device_info to
         launch_kernel. *)
      Device_info.get_default
        () (* HACK: Get default. This should be from kernel context. *)
    in
    let command_buffer : Objc_id.t =
      (device_info.metal_queue !%sel "commandBuffer" : unit -> Objc_id.t) ()
    in
    let encoder : Objc_id.t =
      (command_buffer !%sel "computeCommandEncoder" : unit -> Objc_id.t) ()
    in

    (encoder !!sel "setComputePipelineState:" : Objc_id.t -> unit)
      kernel.native_kernel;

    List.iteri
      (fun i arg_nativeint ->
        (* Assume arg_nativeint is a pointer to a device_buffer.native_buffer
           (Objc_id.t for MTLBuffer) This requires careful handling of how args
           are passed. If args are raw pointers to MTLBuffers: This is tricky.
           `nativeint list` means args are raw addresses. But `MTLBuffer` are
           ObjC objects. We need `Objc_id.t list` for args. Let's assume `args`
           are `Objc_id.t` cast to `nativeint`. *)
        let mtl_buffer_obj = Objc_id.of_nativeint arg_nativeint in
        (encoder !!sel "setBuffer:offset:atIndex:"
          : Objc_id.t -> Unsigned.ULong.t -> Unsigned.ULong.t -> unit)
          mtl_buffer_obj (Unsigned.ULong.of_int 0) (Unsigned.ULong.of_int i))
      args;

    let gx, gy, gz = (global_dims.(0), global_dims.(1), global_dims.(2)) in
    let lx, ly, lz =
      match local_dims with
      | Some ld -> (ld.(0), ld.(1), ld.(2))
      | None ->
          (* Query preferred/max workgroup size from pipeline_state if not
             provided *)
          let pso = kernel.native_kernel in
          let max_total_threads_per_group : int =
            Unsigned.ULong.to_int
              (pso !%sel "maxTotalThreadsPerThreadgroup"
                : unit -> Unsigned.ULong.t)
          in
          let thread_exec_width : int =
            Unsigned.ULong.to_int
              (pso !%sel "threadExecutionWidth" : unit -> Unsigned.ULong.t)
          in
          (* Simple heuristic for 1D/2D/3D, can be much more complex *)
          if gz > 1 then
            let z_ =
              min thread_exec_width (min gz max_total_threads_per_group)
            in
            let y_ =
              min thread_exec_width (min gy (max_total_threads_per_group / z_))
            in
            let x_ =
              min thread_exec_width
                (min gx (max_total_threads_per_group / (z_ * y_)))
            in
            (max 1 x_, max 1 y_, max 1 z_ (* Ensure at least 1 *))
          else if gy > 1 then
            let y_ =
              min thread_exec_width (min gy max_total_threads_per_group)
            in
            let x_ =
              min thread_exec_width (min gx (max_total_threads_per_group / y_))
            in
            (max 1 x_, max 1 y_, 1)
          else (min gx max_total_threads_per_group, 1, 1)
    in

    if lx = 0 || ly = 0 || lz = 0 then
      Error "Metal launch_kernel: local_dims cannot be zero."
    else
      let grid_size_struct =
        Mtl_structs.make_mtl_size ~width:gx ~height:gy ~depth:gz
      in
      let group_size_struct =
        Mtl_structs.make_mtl_size ~width:lx ~height:ly ~depth:lz
      in

      (encoder !!sel "dispatchThreadgroups:threadsPerThreadgroup:"
        : Mtl_structs.mtlSize Ctypes.structure Ctypes.ptr ->
          Mtl_structs.mtlSize Ctypes.structure Ctypes.ptr ->
          unit)
        (Ctypes.addr grid_size_struct)
        (Ctypes.addr group_size_struct);

      (encoder !!sel "endEncoding" : unit -> unit) ();
      (command_buffer !!sel "commit" : unit -> unit) ();
      (* For synchronous execution as per this simple interface: *)
      (command_buffer !!sel "waitUntilCompleted" : unit -> unit) ();
      (* Check for errors on command_buffer.error if needed *)
      let error_obj : Objc_id.t option =
        (command_buffer !?sel "error" : unit -> Objc_id.t option) ()
      in
      match error_obj with
      | Some err when not (Objc_id.is_null err) ->
          let desc =
            string_of_nsstring
              ((err !%sel "localizedDescription" : unit -> Objc_id.t) ())
          in
          Error
            (Printf.sprintf "Metal kernel %s execution failed: %s" kernel.name
               desc)
      | _ -> Ok ()

  let synchronize ~device_info:_ =
    (* Metal is often command-buffered. True synchronization might mean waiting
       on the specific command queue or the last submitted command buffer. If
       launches are synchronous (waitUntilCompleted), this is a no-op. If
       launches are async, this would be (device_info.metal_queue !! sel
       "waitUntilAllCommandsAreScheduled" : unit -> unit) () or similar, or wait
       on a specific event/command buffer. Given launch_kernel does
       waitUntilCompleted, this is a no-op. *)
    ()
end

let name = "METAL"
