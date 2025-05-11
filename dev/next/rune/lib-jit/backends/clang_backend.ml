open Nx_core

let name = "CLANG_CPU"

module Device_info = struct
  type t = {
    arch_flags : string list; (* e.g., ["-march=native"] *)
    optimization_flags : string list; (* e.g., ["-O3"] *)
    float4_str : string option;
    smem_prefix : string;
    barrier_str : string;
  }

  let get_default () =
    {
      arch_flags =
        (match Sys.os_type with
        | "Win32" ->
            [
              "-target";
              "x86_64-pc-windows-msvc";
              "-fms-extensions";
              "-fms-compatibility";
            ]
            (* Basic Windows target *)
        | _ -> [ "-march=native" ]);
      (* More specific detection could be added *)
      optimization_flags = [ "-O3"; "-ffast-math" ];
      float4_str = Some "(float4)";
      smem_prefix = "";
      (* No special shared memory for basic C on CPU via this model *)
      barrier_str = "";
      (* No explicit barriers for single-threaded CPU execution model *)
    }

  let max_shared_memory _ = 0 (* N/A for this simple CPU model *)
  let max_workgroup_size _ = [| 1; 1; 1 |] (* Single "thread" effectively *)

  let supports_dtype _ _ =
    true (* Assume Clang supports most standard C types *)

  let renderer_float4_str t = t.float4_str
  let renderer_smem_prefix t = t.smem_prefix
  let renderer_barrier_str t = t.barrier_str
end

(* For C backend using dylibs, native_buffer can be a Ctypes.ptr to the
   allocated memory *)
type device_buffer_native = Ctypes.voidp

type ('a, 'b) device_buffer = {
  native_buffer : device_buffer_native;
  size_in_bytes : int;
  dtype : ('a, 'b) Dtype.t;
  device_info : Device_info.t;
}

(* Native artifact is a handle to the loaded dynamic library *)
type compiled_artifact_native = Ctypes.voidp (* result of Dl.dlopen *)

type compiled_artifact = {
  native_artifact : compiled_artifact_native;
  entry_points : string list;
  dylib_path : string; (* Keep path for dlclose *)
}

(* Native kernel is a Ctypes.fn (function pointer) *)
type callable_kernel_native =
  (Ctypes.voidp Ctypes.ptr -> unit) Ctypes.static_funptr

type callable_kernel = { native_kernel : callable_kernel_native; name : string }

module Renderer = struct
  (* This is where the logic from cstyle.py's ClangRenderer.render_kernel and
     its helpers would be implemented. It iterates through
     Ir.Lowered.graph_t.instructions. *)

  let render_dtype (type a b) (dt : (a, b) Dtype.t) : string =
    (* Simplified, expand based on cstyle.py type_map *)
    match dt with
    | Dtype.Float32 -> "float"
    | Dtype.Int32 -> "int"
    | Dtype.Bool -> "_Bool" (* C99 bool *)
    | Dtype.Uint8 -> "unsigned char"
    (* ... other types ... *)
    | _ -> failwith ("ClangRenderer: Unsupported Dtype: " ^ Dtype.to_string dt)

  let render_instruction (instr : Ir.Lowered.any_instruction)
      (vars_meta : (Ir.Var.t, Ir.var_metadata) Hashtbl.t)
      (var_names : (Ir.Var.t, string) Hashtbl.t) : string =
    let get_var_name v = Hashtbl.find var_names v in
    let get_var_meta v = Hashtbl.find vars_meta v in
    match instr with
    | Ir.Lowered.Any_Instruction (LI_Const_Scalar { value; dtype; out_var }) ->
        let (Ir.Any_Dtype dt_any) = (get_var_meta out_var).dtype in
        let val_str = Dtype.value_to_string dt_any value in
        (* Needs Dtype.value_to_string *)
        Printf.sprintf "%s %s = %s;" (render_dtype dtype) (get_var_name out_var)
          val_str
    | Ir.Lowered.Any_Instruction
        (LI_Buffer { dtype; size_in_elements = _; out_var }) ->
        (* For CPU C, actual buffer allocation is handled by Runtime. This is
           more of a declaration if we were generating standalone C. In JIT, the
           `out_var` will map to a pointer argument. Here we might just declare
           the pointer type if it's for a local variable (not kernel arg). This
           part needs careful consideration of how kernel args vs. internal vars
           are handled. For now, assume kernel arguments are pointers and this
           is for internal scalar representations. *)
        Printf.sprintf "%s* %s_ptr; // Placeholder for buffer var declaration"
          (render_dtype dtype) (get_var_name out_var)
    | Ir.Lowered.Any_Instruction
        (LI_Range { name_hint = _; upper_bound_exclusive; out_var }) ->
        Printf.sprintf "for (int %s = 0; %s < %s; ++%s) {"
          (get_var_name out_var) (get_var_name out_var)
          (get_var_name upper_bound_exclusive)
          (get_var_name out_var)
    | Ir.Lowered.Any_Instruction
        (LI_Load
           {
             buffer_source_var;
             indices_vars;
             valid_mask_var = _;
             out_var;
             dtype;
           }) ->
        (* Simplified: assumes 1D access for now *)
        let idx_str =
          match indices_vars with
          | [ iv ] -> get_var_name iv
          | _ -> "/* complex_idx */0" (* Placeholder for multi-D indexing *)
        in
        Printf.sprintf "%s %s = %s_ptr[%s];" (render_dtype dtype)
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
          | _ -> "/* complex_idx */0"
        in
        Printf.sprintf "%s_ptr[%s] = %s;"
          (get_var_name buffer_target_var)
          idx_str
          (get_var_name scalar_value_to_store_var)
    | Ir.Lowered.Any_Instruction
        (LI_Scalar_ALU { op_type; inputs_vars; out_var; dtype }) ->
        let op_str =
          match op_type with
          | Ir.Lowered.Scalar_Add -> "+"
          | Ir.Lowered.Scalar_Mul -> "*"
          | Ir.Lowered.Scalar_Max -> "fmax" (* Requires math.h or custom max *)
          | Ir.Lowered.Scalar_CmpLt -> "<"
        in
        let args_str =
          String.concat
            (Printf.sprintf " %s " op_str)
            (List.map get_var_name inputs_vars)
        in
        Printf.sprintf "%s %s = (%s);" (render_dtype dtype)
          (get_var_name out_var) args_str
    | Ir.Lowered.Any_Instruction
        (LI_Special_Index { name_hint = _; kind = _; out_var = _ }) ->
        "// LI_Special_Index not directly applicable for simple CPU C kernel \
         in this model"

  let render ~device_info:_ ~lowered_ir ~kernel_name =
    let buffer = Buffer.create 1024 in
    Buffer.add_string buffer "#include <math.h>\n";
    (* For fmax, etc. *)
    Buffer.add_string buffer "#include <stdbool.h>\n";
    (* For _Bool *)
    Buffer.add_string buffer "#include <stdint.h>\n\n";

    (* For int types like int32_t if used *)

    (* Kernel signature: void kernel_name(void* arg0, void* arg1, ...) tinygrad
       uses `float* data0`, `float* data1`. We'll generalize. This needs to map
       Ir.Lowered.kernel_input_vars and kernel_output_vars. For simplicity,
       assume all are just `void*` and cast inside. Or, better, get their types
       from vars_metadata. *)
    let var_names =
      Hashtbl.create (Hashtbl.length lowered_ir.Ir.Lowered.vars_metadata)
    in
    let next_var_idx = ref 0 in
    let get_fresh_var_name (v : Ir.Var.t) (hint : string) : string =
      let name = Printf.sprintf "%s%d" hint !next_var_idx in
      incr next_var_idx;
      Hashtbl.add var_names v name;
      name
    in
    (* Pre-assign names to all known variables to ensure consistency *)
    Hashtbl.iter
      (fun v meta ->
        let (Ir.Any_Dtype dt_any) = meta.Ir.dtype in
        let hint =
          Dtype.to_string dt_any (* Basic hint from dtype *)
          |> String.map (fun c -> if Char.is_alphanum c then c else '_')
        in
        ignore (get_fresh_var_name v hint))
      lowered_ir.Ir.Lowered.vars_metadata;

    let kernel_args =
      List.mapi
        (fun i var_id ->
          let meta = Hashtbl.find lowered_ir.Ir.Lowered.vars_metadata var_id in
          let (Ir.Any_Dtype dt_any) = meta.dtype in
          (* Kernel arguments are always pointers to the buffer type *)
          Printf.sprintf "%s* %s_ptr" (render_dtype dt_any)
            (Hashtbl.find var_names var_id))
        (lowered_ir.Ir.Lowered.kernel_input_vars
         @ lowered_ir.Ir.Lowered.kernel_output_vars
        |> List.sort_uniq Ir.Var.compare)
    in
    Buffer.add_string buffer
      (Printf.sprintf "void %s(%s) {\n" kernel_name
         (String.concat ", " kernel_args));

    (* Render instructions *)
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
    clang_path : string;
    arch_flags : string list;
    opt_flags : string list;
    shared_flags : string list;
    include_dirs : string list;
    library_dirs : string list;
    link_libs : string list;
  }

  let default_options (dev_info : Device_info.t) =
    {
      clang_path = "clang";
      (* Or getenv "CLANG" *)
      arch_flags = dev_info.arch_flags;
      opt_flags = dev_info.optimization_flags;
      shared_flags =
        (match Sys.os_type with
        | "Unix" | "Cygwin" -> [ "-fPIC"; "-shared" ]
        | "Win32" ->
            [ "-DLL" ] (* Adjust based on MinGW or MSVC Clang behavior *)
        | _ -> [ "-fPIC"; "-shared" ]);
      include_dirs = [];
      library_dirs = [];
      link_libs = [];
    }

  let run_command cmd args =
    let full_cmd = Filename.quote_command cmd args in
    if getenv_int "DEBUG" >= 2 then Printf.eprintf "CMD: %s\n" full_cmd;
    let ic = Unix.open_process_in full_cmd in
    let output = Buffer.create 128 in
    try
      while true do
        Buffer.add_channel output ic 1024
      done
    with End_of_file -> (
      ();
      let status = Unix.close_process_in ic in
      match status with
      | Unix.WEXITED 0 -> Ok (Buffer.contents output)
      | Unix.WEXITED n ->
          Error
            (Printf.sprintf "Command '%s' failed with exit code %d. Output:\n%s"
               full_cmd n (Buffer.contents output))
      | Unix.WSIGNALED n ->
          Error
            (Printf.sprintf "Command '%s' killed by signal %d. Output:\n%s"
               full_cmd n (Buffer.contents output))
      | Unix.WSTOPPED n ->
          Error
            (Printf.sprintf "Command '%s' stopped by signal %d. Output:\n%s"
               full_cmd n (Buffer.contents output)))

  let compile ~device_info:_ ~source_code ~options =
    (* 1. Create temporary files for .c and .so/.dylib *)
    let temp_c_file = Filename.temp_file "kernel_" ".c" in
    let temp_so_file =
      Filename.temp_file "kernel_"
        (match Sys.os_type with "Win32" -> ".dll" | _ -> ".so")
    in
    try
      (* 2. Write source_code to .c file *)
      let oc = open_out temp_c_file in
      output_string oc source_code;
      close_out oc;

      (* 3. Construct clang command *)
      let cmd_args =
        options.arch_flags @ options.opt_flags @ options.shared_flags
        @ List.map (fun d -> "-I" ^ d) options.include_dirs
        @ List.map (fun d -> "-L" ^ d) options.library_dirs
        @ [ temp_c_file; "-o"; temp_so_file ]
        @ List.map (fun l -> "-l" ^ l) options.link_libs
      in
      (* 4. Execute clang *)
      match run_command options.clang_path cmd_args with
      | Ok _ -> (
          (* 5. Load the .so/.dylib *)
          try
            let native_artifact =
              Dl.dlopen ~filename:temp_so_file
                ~flags:[ Dl.RTLD_NOW; Dl.RTLD_LOCAL ]
            in
            (* We don't know entry points here, Runtime.get_kernel will find
               them *)
            Ok { native_artifact; entry_points = []; dylib_path = temp_so_file }
          with Dl.DLError msg ->
            Error
              (Printf.sprintf "Dl.dlopen failed for %s: %s" temp_so_file msg))
      | Error msg -> Error msg
    with exn ->
      (* Clean up temp files on error if they were created *)
      if Sys.file_exists temp_c_file then Sys.remove temp_c_file;
      if Sys.file_exists temp_so_file then Sys.remove temp_so_file;
      Error
        (Printf.sprintf "Compilation failed with exception: %s"
           (Printexc.to_string exn))
  (* Not removing temp_so_file on success because dlopen needs it. It should be
     removed when the artifact is no longer needed/garbage collected, or a
     proper caching mechanism should manage these files. For simplicity, let's
     leak it for now or have a finalizer on compiled_artifact. *)
end

module Runtime = struct
  let allocate_buffer ~device_info:_ ~size_in_bytes ~dtype =
    try
      (* Ctypes.allocate_n is for statically known types. For raw bytes,
         Ctypes.allocate Ctypes.char size_in_bytes works, then cast to voidp. Or
         use C.stdlib.malloc if linking against C stdlib. Unix.map_file could be
         an option for mmap-based allocation too. Simplest for now: Bigarray for
         host-side managed memory that C can see. Actually, tinygrad just uses
         libc.malloc. *)
      let ptr = Ctypes.allocate_n Ctypes.char ~count:size_in_bytes in
      Ok
        {
          native_buffer = Ctypes.to_voidp ptr;
          size_in_bytes;
          dtype;
          device_info = Device_info.get_default ();
        }
    with e ->
      Error (Printf.sprintf "Allocation failed: %s" (Printexc.to_string e))

  let deallocate_buffer buffer =
    (* If using Ctypes.allocate_n, it's garbage collected.
         If using libc.malloc, would need Ctypes.Posix.Stdlib.free.
         For this sketch, assume GC or manual C free if we used that.
         Tinygrad's C backend uses libc.free. Let's assume we need to free.
         To do this properly, native_buffer should store the original Ctypes.ptr not voidp.
         Or, we make `allocate_buffer` return a raw C malloc'd pointer.
      *)
    (* Ctypes.Posix.Stdlib.free buffer.native_buffer (* if native_buffer was from malloc *) *)
    ignore buffer;
    ()

  let copy_to_device ~dest_buffer ~host_data ~host_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes + host_data_offset_bytes > dest_buffer.size_in_bytes then
      (* This check is wrong, should be host_data size*)
      Error "Copy size exceeds host data bounds or dest_buffer size"
    else if copy_size_bytes > dest_buffer.size_in_bytes then
      Error "Copy size exceeds destination buffer capacity"
    else
      try
        let src_ptr = Ctypes.(host_data +@ host_data_offset_bytes) in
        Ctypes.memcpy ~dst:dest_buffer.native_buffer ~src:src_ptr
          ~len:copy_size_bytes;
        Ok ()
      with e ->
        Error
          (Printf.sprintf "copy_to_device failed: %s" (Printexc.to_string e))

  let copy_from_device ~src_buffer ~host_dest_ptr ~device_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes + device_data_offset_bytes > src_buffer.size_in_bytes
    then Error "Copy range exceeds source device buffer bounds"
    else
      try
        let src_device_ptr =
          Ctypes.(src_buffer.native_buffer +@ device_data_offset_bytes)
        in
        Ctypes.memcpy ~dst:host_dest_ptr ~src:src_device_ptr
          ~len:copy_size_bytes;
        Ok ()
      with e ->
        Error
          (Printf.sprintf "copy_from_device failed: %s" (Printexc.to_string e))

  let get_kernel ~artifact ~kernel_name =
    try
      (* FFI type: (Ctypes.ptr Ctypes.voidp -> returning Ctypes.void) This
         means: a function taking one argument, which is a pointer to (array of)
         void pointers. The return type is void. *)
      let kernel_fn_type = Ctypes.(ptr voidp @-> returning void) in
      let native_kernel =
        Ctypes.foreign_symbol ~from:artifact.native_artifact kernel_name
          kernel_fn_type
      in
      Ok { native_kernel; name = kernel_name }
    with
    | Dl.DLError msg ->
        Error (Printf.sprintf "Dl.dlsym failed for %s: %s" kernel_name msg)
    | Not_found ->
        Error
          (Printf.sprintf "Symbol %s not found in compiled artifact" kernel_name)

  let launch_kernel ~kernel ~global_dims:_ (* unused for simple CPU *)
      ~local_dims:_ (* unused *) ~args =
    try
      kernel.native_kernel args;
      (* Directly call the function pointer *)
      Ok ()
    with e ->
      Error
        (Printf.sprintf "Kernel launch for %s failed: %s" kernel.name
           (Printexc.to_string e))

  let synchronize ~device_info:_ = () (* No-op for synchronous CPU execution *)
end
