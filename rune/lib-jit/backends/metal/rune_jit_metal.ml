(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* metal_backend.ml – final trimmed version *)

open Rune_jit

(* ───── Misc Helpers ───── *)

let foreign_memcpy =
  Foreign.foreign "memcpy"
    Ctypes.(ptr void @-> ptr void @-> size_t @-> returning (ptr void))

let memcpy_bytes ~dst ~src ~len =
  if len > 0 then ignore (foreign_memcpy dst src (Unsigned.Size_t.of_int len))

(* ───── Opaque Native Handles ───── *)

type device_info = {
  device : Metal.Device.t;
  queue : Metal.CommandQueue.t;
  attributes : Metal.Device.attributes;
}

type device_buffer_native = Metal.Buffer.t
type compiled_artifact_native = Metal.Library.t
type callable_kernel_native = Metal.ComputePipelineState.t

(* ───── Public Record Wrappers (Match backend_intf) ───── *)

type nonrec 'a device_buffer =
  ('a, device_buffer_native) Backend_intf.device_buffer

type nonrec any_device_buffer =
  device_buffer_native Backend_intf.any_device_buffer

type nonrec compiled_artifact =
  compiled_artifact_native Backend_intf.compiled_artifact

type nonrec callable_kernel =
  callable_kernel_native Backend_intf.callable_kernel

(* ───── Device_info ───── *)

module Device_info = struct
  let get_default () =
    let device = Metal.Device.create_system_default () in
    let queue = Metal.CommandQueue.on_device device in
    { device; queue; attributes = Metal.Device.get_attributes device }

  let max_shared_memory t =
    Unsigned.ULong.to_int t.attributes.max_threadgroup_memory_length

  let max_workgroup_size t =
    let m = t.attributes.max_threads_per_threadgroup in
    [| m.width; m.height; m.depth |]

  let supports_dtype _ (Dtype.Any_Dtype _) = true
  let renderer_float4_str _ = Some "float4"
  let renderer_smem_prefix _ = "threadgroup "

  let renderer_barrier_str _ =
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
end

(* ───── Renderer ───── *)

module Renderer = struct
  (* Helpers *)
  let metal_of_dtype (Dtype.Any_Dtype dt) =
    match dt with
    | Dtype.Float32 -> "float"
    | Dtype.Int32 -> "int"
    | Dtype.Uint8 -> "uchar"
    | Dtype.Bool -> "bool"
    | Dtype.Unit -> "void"

  (* lit_zero not currently used let lit_zero (Dtype.Any_Dtype dt) = match dt
     with | Dtype.Float32 -> "0.0f" | Dtype.Int32 | Dtype.Uint8 -> "0" |
     Dtype.Bool -> "false" | Dtype.Unit -> "" *)

  let binop_str = function
    | Ir.Add -> "+"
    | Sub -> "-"
    | Mul -> "*"
    | Div -> "/"
    | Idiv -> "/" (* Integer division - Metal uses same / operator *)
    | Fdiv -> "/" (* Float division *)
    | Mod -> "%"
    | Pow -> "pow"
    | Max -> "fmax"
    | Min -> "fmin"
    | Cmplt -> "<"
    | Cmpne -> "!="
    | Xor -> "^"
    | Or -> "|"
    | And -> "&"
    | Shl -> "<<"
    | Shr -> ">>"

  let unary_op_str = function
    | Ir.Neg -> "-"
    | Log2 -> "log2"
    | Exp2 -> "exp2"
    | Sin -> "sin"
    | Sqrt -> "sqrt"
    | Recip -> "1.0f/" (* Prefix for reciprocal *)

  let special_idx_str = function
    | Ir.Special_index_kind.Global_task_idx d ->
        Printf.sprintf "gtid.%c" [| 'x'; 'y'; 'z' |].(d)
    | Local_thread_idx d -> Printf.sprintf "lid.%c" [| 'x'; 'y'; 'z' |].(d)
    | Workgroup_idx d -> Printf.sprintf "gid.%c" [| 'x'; 'y'; 'z' |].(d)

  (* -------- stable names in one pass -------- *)

  let build_name_map (ir : Ir.Lowered.graph_t) arg_ll_vars =
    let open Hashtbl in
    let keys =
      Seq.append (to_seq_keys ir.vars_metadata) (List.to_seq arg_ll_vars)
      |> List.of_seq |> List.sort_uniq Var.compare
    in
    let pairs = List.mapi (fun i v -> (v, Printf.sprintf "v%d" i)) keys in
    of_seq (List.to_seq pairs)

  let render_instruction ~name_of ~smem_prefix ~local_buffer_vars ~names indent
      (ins : Ir.Lowered.instruction) : string =
    let ind = String.make (indent * 2) ' ' in
    let v = name_of in
    match ins with
    | L_Buffer { dtype; size; out } ->
        (* Note: L_Buffer is handled specially in render function to avoid
           collisions *)
        Printf.sprintf "%s%s%s %s[%d];" ind smem_prefix (metal_of_dtype dtype)
          (v out) size
    | L_Const { dtype; value; out } ->
        Printf.sprintf "%s%s %s = %s;" ind (metal_of_dtype dtype) (v out) value
    | L_Range { idx; bound } ->
        Printf.sprintf "%sfor (int %s = 0; %s < %s; ++%s) {" ind (v idx) (v idx)
          (v bound) (v idx)
    | L_EndRange -> ind ^ "}"
    | L_Special { dst; kind } ->
        Printf.sprintf "%suint %s = %s;" ind (v dst) (special_idx_str kind)
    | L_Load { dst; buf; idx; dtype; valid = _ } ->
        Printf.sprintf "%s%s %s = %s[%s];" ind (metal_of_dtype dtype) (v dst)
          (v buf) (v idx)
    | L_Store { buf; idx; src; valid = _ } ->
        (* For stores to kernel arguments, use the original name without _local
           suffix *)
        let buf_name =
          if Hashtbl.mem local_buffer_vars buf then
            Hashtbl.find names buf (* Use original name for kernel args *)
          else v buf
        in
        Printf.sprintf "%s%s[%s] = %s;" ind buf_name (v idx) (v src)
    | L_ALU { dst; op; args; dtype } -> (
        match (op, args) with
        | Ir.Lowered.Binary bop, [ a; b ] -> (
            match bop with
            | Add | Sub | Mul | Div | Idiv | Fdiv | Mod | Xor | Or | And | Shl
            | Shr ->
                Printf.sprintf "%s%s %s = %s %s %s;" ind (metal_of_dtype dtype)
                  (v dst) (v a) (binop_str bop) (v b)
            | Max | Min ->
                Printf.sprintf "%s%s %s = %s(%s, %s);" ind
                  (metal_of_dtype dtype) (v dst) (binop_str bop) (v a) (v b)
            | Cmplt | Cmpne ->
                Printf.sprintf "%sbool %s = %s %s %s;" ind (v dst) (v a)
                  (binop_str bop) (v b)
            | Pow ->
                Printf.sprintf "%s%s %s = pow(%s, %s);" ind
                  (metal_of_dtype dtype) (v dst) (v a) (v b))
        | Ir.Lowered.Unary uop, [ a ] -> (
            match uop with
            | Neg | Recip ->
                Printf.sprintf "%s%s %s = %s%s;" ind (metal_of_dtype dtype)
                  (v dst) (unary_op_str uop) (v a)
            | Log2 | Exp2 | Sin | Sqrt ->
                Printf.sprintf "%s%s %s = %s(%s);" ind (metal_of_dtype dtype)
                  (v dst) (unary_op_str uop) (v a))
        | Ir.Lowered.Ternary Where, [ cond; x; y ] ->
            Printf.sprintf "%s%s %s = %s ? %s : %s;" ind (metal_of_dtype dtype)
              (v dst) (v cond) (v x) (v y)
        | Ir.Lowered.Ternary Mulacc, [ a; b; c ] ->
            Printf.sprintf "%s%s %s = fma(%s, %s, %s);" ind
              (metal_of_dtype dtype) (v dst) (v a) (v b) (v c)
        | _ -> ind ^ "// unsupported ALU op/args combination")
    | L_Local { dtype; size; out } ->
        Printf.sprintf "%s%s%s %s[%d];" ind smem_prefix (metal_of_dtype dtype)
          (v out) size
    | L_Acc { dtype; out } ->
        Printf.sprintf "%s%s %s;" ind (metal_of_dtype dtype) (v out)
    | L_If { cond } -> Printf.sprintf "%sif (%s) {" ind (v cond)
    | L_EndIf -> ind ^ "}"
    | L_Barrier -> ind ^ "threadgroup_barrier(mem_flags::mem_threadgroup);"
    | L_Cast { dst; src; dtype } ->
        Printf.sprintf "%s%s %s = (%s)(%s);" ind (metal_of_dtype dtype) (v dst)
          (metal_of_dtype dtype) (v src)
    | L_Assign { dst; src } -> Printf.sprintf "%s%s = %s;" ind (v dst) (v src)
    | L_Define_Global { ptr; dtype; size } ->
        Printf.sprintf "%sdevice %s* %s[%d];" ind (metal_of_dtype dtype) (v ptr)
          size
    | L_Vconst { dst; values; dtype } ->
        let vals = String.concat ", " (Array.to_list values) in
        Printf.sprintf "%s%s %s = {%s};" ind (metal_of_dtype dtype) (v dst) vals
    | L_Define_Var { sym_var; out } ->
        Printf.sprintf "%suint %s = %d; // sym var %s [%d,%d]" ind (v out)
          sym_var.min_val sym_var.name sym_var.min_val sym_var.max_val
    | L_Block { block_id; start } ->
        if start then Printf.sprintf "%s// BLOCKSTART %d" ind block_id
        else Printf.sprintf "%s// BLOCKEND %d" ind block_id
    | L_Unroll { idx; iterations } ->
        Printf.sprintf
          "%s#pragma unroll(%d)\n%sfor(int %s = 0; %s < %d; %s++) {" ind
          iterations ind (v idx) (v idx) iterations (v idx)
    | L_Gep { dst; src; indices; dtype } ->
        let idx_str =
          String.concat ""
            (Array.to_list
               (Array.map (fun i -> Printf.sprintf "[%d]" i) indices))
        in
        Printf.sprintf "%s%s %s = %s%s;" ind (metal_of_dtype dtype) (v dst)
          (v src) idx_str
    | L_Vectorize { dst; srcs; dtype } ->
        let vals = String.concat ", " (Array.to_list (Array.map v srcs)) in
        Printf.sprintf "%s%s %s = {%s};" ind (metal_of_dtype dtype) (v dst) vals
    | L_Ptrcat { dst; ptrs; dtype } ->
        (* Pointer concatenation - simplified *)
        Printf.sprintf "%s%s* %s = %s; // ptrcat" ind (metal_of_dtype dtype)
          (v dst)
          (if Array.length ptrs > 0 then v ptrs.(0) else "nullptr")
    | L_Wmma { dst; a; b; c; m; n; k; dtype } ->
        (* Tensor core operations - Metal doesn't have native WMMA, use regular
           matmul *)
        Printf.sprintf "%s// WMMA %dx%dx%d\n%s%s %s = %s + %s * %s;" ind m n k
          ind (metal_of_dtype dtype) (v dst) (v c) (v a) (v b)
    | L_Bitcast { dst; src; dtype } ->
        Printf.sprintf "%s%s %s = as_type<%s>(%s);" ind (metal_of_dtype dtype)
          (v dst) (metal_of_dtype dtype) (v src)
    | L_Custom { dst; op_name; args; attributes = _; inline } ->
        let args_str = String.concat ", " (Array.to_list (Array.map v args)) in
        let dst_str = match dst with Some d -> v d ^ " = " | None -> "" in
        if inline then
          Printf.sprintf "%s%s%s(%s); // inline custom op" ind dst_str op_name
            args_str
        else Printf.sprintf "%s%s%s(%s);" ind dst_str op_name args_str
    | L_Noop -> ind ^ "// noop"

  (* -------- entry point -------- *)

  let dedup_preserve_order lst =
    let seen = Hashtbl.create 7 in
    List.filter
      (fun v ->
        if Hashtbl.mem seen v then false
        else (
          Hashtbl.add seen v ();
          true))
      lst

  let render ~device_info ~(lowered_ir : Ir.Lowered.graph_t) ~kernel_name =
    let arg_ll =
      dedup_preserve_order
        (lowered_ir.kernel_input_vars @ lowered_ir.kernel_output_vars)
    in

    (* First pass: identify variables that need renaming *)
    let local_buffer_vars = Hashtbl.create 10 in
    let arg_set =
      List.fold_left (fun acc v -> Var.Set.add v acc) Var.Set.empty arg_ll
    in
    List.iter
      (fun instr ->
        match instr with
        | Ir.Lowered.L_Buffer { out; _ } | Ir.Lowered.L_Local { out; _ } ->
            if Var.Set.mem out arg_set then
              Hashtbl.add local_buffer_vars out true
        | _ -> ())
      lowered_ir.instructions;

    (* Build name map with special handling for collision variables *)
    let names = build_name_map lowered_ir arg_ll in

    (* Create a name function that handles collisions *)
    let name_of v =
      let base = Hashtbl.find names v in
      (* If this variable is a local buffer that collides with kernel arg,
         always use the _local suffix *)
      if Hashtbl.mem local_buffer_vars v then base ^ "_local" else base
    in

    (* Special function for kernel arguments - never add suffix *)
    let name_of_arg v = Hashtbl.find names v in

    let buf = Buffer.create 1024 in
    Buffer.add_string buf "#include <metal_stdlib>\nusing namespace metal;\n\n";

    (* arguments *)
    let arg_decl i v =
      let md = Hashtbl.find lowered_ir.vars_metadata v in
      Printf.sprintf "device %s* %s [[buffer(%d)]]" (metal_of_dtype md.dtype)
        (name_of_arg v) i
    in
    let arg_line =
      List.mapi arg_decl arg_ll
      @ [
          "uint3 gtid [[thread_position_in_grid]]";
          "uint3 lid  [[thread_position_in_threadgroup]]";
          "uint3 gid  [[threadgroup_position_in_grid]]";
        ]
      |> String.concat ",\n  "
    in
    Buffer.add_string buf
      (Printf.sprintf "kernel void %s(\n  %s\n) {\n" kernel_name arg_line);

    (* body *)
    let indent = ref 1 in
    List.iter
      (fun ins ->
        (match ins with
        | Ir.Lowered.L_Range _ -> ()
        | L_EndRange -> decr indent
        | _ -> ());
        Buffer.add_string buf
          (render_instruction ~name_of
             ~smem_prefix:(Device_info.renderer_smem_prefix device_info)
             ~local_buffer_vars ~names !indent ins);
        Buffer.add_char buf '\n';
        match ins with L_Range _ -> incr indent | _ -> ())
      lowered_ir.instructions;

    Buffer.add_string buf "}\n";
    Buffer.contents buf
end

(* ───── Compiler ───── *)

module Compiler = struct
  type compile_options = Metal.CompileOptions.t

  let default_options _ = Metal.CompileOptions.init ()

  let compile ~device_info ~source_code ~options =
    try
      let lib =
        Metal.Library.on_device device_info.device ~source:source_code options
      in
      Ok
        {
          Backend_intf.native_artifact = lib;
          entry_points = Array.to_list (Metal.Library.get_function_names lib);
        }
    with ex -> Error (Printexc.to_string ex)
end

(* ───── Runtime ───── *)

module Runtime = struct
  let allocate_buffer ~device_info ~size_in_bytes ~dtype =
    let len = if size_in_bytes = 0 then 1 else size_in_bytes in
    let opts =
      Metal.ResourceOptions.make
        ~storage_mode:Metal.ResourceOptions.storage_mode_shared ()
    in
    try
      Ok
        {
          Backend_intf.native_buffer =
            Metal.Buffer.on_device device_info.device ~length:len opts;
          size_in_bytes = len;
          dtype;
        }
    with ex -> Error (Printexc.to_string ex)

  (* helper: byte-wise pointer arithmetic on a void* *)
  let[@inline] voidp_with_byte_offset vp ~bytes =
    (* cast void* → char* → add → cast back *)
    Ctypes.(to_voidp (from_voidp char vp +@ bytes))

  let copy_to_device ~dest_buffer ~host_data ~host_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes = 0 then Ok ()
    else if dest_buffer.Backend_intf.size_in_bytes < copy_size_bytes then
      Error "size overflow"
    else
      let dst = Metal.Buffer.contents dest_buffer.native_buffer in
      let src =
        Ctypes.ptr_of_raw_address host_data
        |> voidp_with_byte_offset ~bytes:host_data_offset_bytes
      in
      memcpy_bytes ~dst ~src ~len:copy_size_bytes;
      Ok ()

  let copy_from_device ~src_buffer ~host_dest_ptr ~device_data_offset_bytes
      ~copy_size_bytes =
    if copy_size_bytes = 0 then Ok ()
    else if
      src_buffer.Backend_intf.size_in_bytes
      < device_data_offset_bytes + copy_size_bytes
    then Error "range overflow"
    else
      let src =
        Metal.Buffer.contents src_buffer.native_buffer
        |> voidp_with_byte_offset ~bytes:device_data_offset_bytes
      in
      let dst = Ctypes.ptr_of_raw_address host_dest_ptr |> Ctypes.to_voidp in
      memcpy_bytes ~dst ~src ~len:copy_size_bytes;
      Ok ()

  let get_kernel ~artifact ~kernel_name =
    if not (List.mem kernel_name artifact.Backend_intf.entry_points) then
      Error
        (Printf.sprintf "Kernel %s not found in compiled artifact" kernel_name)
    else
      try
        let lib = artifact.native_artifact in
        let fn = Metal.Library.new_function_with_name lib kernel_name in
        let dev = Metal.Library.get_device lib in
        let pso, _reflection =
          Metal.ComputePipelineState.on_device_with_function dev fn
        in
        Ok { Backend_intf.native_kernel = pso; name = kernel_name }
      with exn -> Error (Printexc.to_string exn)

  let launch_kernel ?local_dims ~device_info ~global_dims ~args kernel =
    let di = device_info in
    let cb = Metal.CommandBuffer.on_queue di.queue in
    let enc = Metal.ComputeCommandEncoder.on_buffer cb in
    Metal.ComputeCommandEncoder.set_compute_pipeline_state enc
      kernel.Backend_intf.native_kernel;
    List.iteri
      (fun i (Backend_intf.Any_Device_Buffer b) ->
        Metal.ComputeCommandEncoder.set_buffer enc ~offset:0 ~index:i
          b.native_buffer)
      args;
    let ldims = Option.value ~default:[| 1; 1; 1 |] local_dims in
    let grid =
      Metal.Size.
        {
          width = global_dims.(0);
          height = global_dims.(1);
          depth = global_dims.(2);
        }
    in
    let group =
      Metal.Size.{ width = ldims.(0); height = ldims.(1); depth = ldims.(2) }
    in
    Metal.ComputeCommandEncoder.dispatch_threadgroups enc
      ~threadgroups_per_grid:grid ~threads_per_threadgroup:group;
    Metal.ComputeCommandEncoder.end_encoding enc;
    Metal.CommandBuffer.commit cb;
    Metal.CommandBuffer.wait_until_completed cb;
    match Metal.CommandBuffer.get_error cb with
    | None -> Ok ()
    | Some e -> Error e

  let synchronize ~device_info =
    let cb = Metal.CommandBuffer.on_queue device_info.queue in
    Metal.CommandBuffer.commit cb;
    Metal.CommandBuffer.wait_until_completed cb
end

let name = "METAL_GPU"
