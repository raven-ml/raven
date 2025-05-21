(* metal_backend.ml – final trimmed version *)

open Ir
open Backend_intf

module B : Backend_intf.S = struct
  (* ───────────── Misc helpers ───────────── *)

  let foreign_memcpy =
    Foreign.foreign "memcpy"
      Ctypes.(ptr void @-> ptr void @-> size_t @-> returning (ptr void))

  let memcpy_bytes ~dst ~src ~len =
    if len > 0 then ignore (foreign_memcpy dst src (Unsigned.Size_t.of_int len))

  (* ───────────── Opaque native handles ───────────── *)

  type device_info = {
    device : Metal.Device.t;
    queue : Metal.CommandQueue.t;
    attributes : Metal.Device.attributes;
  }

  type device_buffer_native = Metal.Buffer.t
  type compiled_artifact_native = Metal.Library.t
  type callable_kernel_native = Metal.ComputePipelineState.t

  (* ───────────── Public record wrappers (match backend_intf) ───────────── *)

  type nonrec 'a device_buffer = ('a, device_buffer_native) device_buffer
  type nonrec any_device_buffer = device_buffer_native any_device_buffer
  type nonrec compiled_artifact = compiled_artifact_native compiled_artifact
  type nonrec callable_kernel = callable_kernel_native callable_kernel

  (* ───────────── Device_info ───────────── *)

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

  (* ───────────── Renderer ───────────── *)

  module Renderer = struct
    (* Helpers *)
    let metal_of_dtype (Dtype.Any_Dtype dt) =
      match dt with
      | Dtype.Float32 -> "float"
      | Dtype.Int32 -> "int"
      | Dtype.Uint8 -> "uchar"
      | Dtype.Bool -> "bool"
      | Dtype.Unit -> "void"

    let lit_zero (Dtype.Any_Dtype dt) =
      match dt with
      | Dtype.Float32 -> "0.0f"
      | Dtype.Int32 | Dtype.Uint8 -> "0"
      | Dtype.Bool -> "false"
      | Dtype.Unit -> ""

    let binop_str = function
      | Add -> "+"
      | Sub -> "-"
      | Mul -> "*"
      | Div -> "/"
      | Max -> "fmax"
      | Min -> "fmin"

    let scalar_op_str = function
      | Ir.Lowered.Bin k -> binop_str k
      | CmpLt -> "<"

    let special_idx_str = function
      | Special_index_kind.Global_task_idx d ->
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

    let render_instruction ~name_of ~smem_prefix indent
        (ins : Ir.Lowered.instruction) : string =
      let ind = String.make (indent * 2) ' ' in
      let v = name_of in
      match ins with
      | L_Buffer { dtype; size; out } ->
          Printf.sprintf "%s%s%s %s[%d];" ind smem_prefix (metal_of_dtype dtype)
            (v out) size
      | L_Const { dtype; value; out } ->
          Printf.sprintf "%s%s %s = %s;" ind (metal_of_dtype dtype) (v out)
            value
      | L_Range { idx; upper } ->
          Printf.sprintf "%sfor (int %s = 0; %s < %s; ++%s) {" ind (v idx)
            (v idx) (v upper) (v idx)
      | L_EndRange -> ind ^ "}"
      | L_SpecialIndex { dst; kind } ->
          Printf.sprintf "%suint %s = %s;" ind (v dst) (special_idx_str kind)
      | L_Load { dst; buf; idxs = [ i ]; mask; dtype } -> (
          let base = Printf.sprintf "%s[%s]" (v buf) (v i) in
          match mask with
          | None ->
              Printf.sprintf "%s%s %s = %s;" ind (metal_of_dtype dtype) (v dst)
                base
          | Some m ->
              Printf.sprintf "%s%s %s = (%s!=0) ? (%s) : (%s);" ind
                (metal_of_dtype dtype) (v dst) (v m) base (lit_zero dtype))
      | L_Store { buf; idxs = [ i ]; src; mask } -> (
          let rhs = Printf.sprintf "%s[%s] = %s;" (v buf) (v i) (v src) in
          match mask with
          | None -> ind ^ rhs
          | Some m -> Printf.sprintf "%sif (%s!=0) { %s }" ind (v m) rhs)
      | L_ALU { dst; op; args = [ a; b ]; dtype } ->
          Printf.sprintf "%s%s %s = %s %s %s;" ind (metal_of_dtype dtype)
            (v dst) (v a) (scalar_op_str op) (v b)
      | _ -> ind ^ "// unsupported/ill-formed"

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

    let render ~device_info ~(lowered_ir : Lowered.graph_t) ~kernel_name =
      let arg_ll =
        dedup_preserve_order
          (lowered_ir.kernel_input_vars @ lowered_ir.kernel_output_vars)
      in
      let names = build_name_map lowered_ir arg_ll in
      let name_of v = Hashtbl.find names v in

      let buf = Buffer.create 1024 in
      Buffer.add_string buf
        "#include <metal_stdlib>\nusing namespace metal;\n\n";

      (* arguments *)
      let arg_decl i v =
        let md = Hashtbl.find lowered_ir.vars_metadata v in
        Printf.sprintf "device %s* %s [[buffer(%d)]]" (metal_of_dtype md.dtype)
          (name_of v) i
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
          | Lowered.L_Range _ -> ()
          | L_EndRange -> decr indent
          | _ -> ());
          Buffer.add_string buf
            (render_instruction ~name_of
               ~smem_prefix:(Device_info.renderer_smem_prefix device_info)
               !indent ins);
          Buffer.add_char buf '\n';
          match ins with L_Range _ -> incr indent | _ -> ())
        lowered_ir.instructions;

      Buffer.add_string buf "}\n";
      Buffer.contents buf
  end

  (* ───────────── Compiler ───────────── *)

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
            native_artifact = lib;
            entry_points = Array.to_list (Metal.Library.get_function_names lib);
          }
      with ex -> Error (Printexc.to_string ex)
  end

  (* ───────────── Runtime ───────────── *)

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
            native_buffer =
              Metal.Buffer.on_device device_info.device ~length:len opts;
            size_in_bytes = len;
            dtype;
          }
      with ex -> Error (Printexc.to_string ex)

    let copy_to_device ~dest_buffer ~host_data ~host_data_offset_bytes
        ~copy_size_bytes =
      if copy_size_bytes = 0 then Ok ()
      else if dest_buffer.size_in_bytes < copy_size_bytes then
        Error "size overflow"
      else
        let dst = Metal.Buffer.contents dest_buffer.native_buffer in
        let src =
          Ctypes.(
            to_voidp (ptr_of_raw_address host_data +@ host_data_offset_bytes))
        in
        memcpy_bytes ~dst ~src ~len:copy_size_bytes;
        Ok ()

    let copy_from_device ~src_buffer ~host_dest_ptr ~device_data_offset_bytes
        ~copy_size_bytes =
      if copy_size_bytes = 0 then Ok ()
      else if
        src_buffer.size_in_bytes < device_data_offset_bytes + copy_size_bytes
      then Error "range overflow"
      else
        let src =
          Ctypes.(
            Metal.Buffer.contents src_buffer.native_buffer
            +@ device_data_offset_bytes)
        in
        let dst = Ctypes.to_voidp (Ctypes.ptr_of_raw_address host_dest_ptr) in
        memcpy_bytes ~dst ~src ~len:copy_size_bytes;
        Ok ()

    let get_kernel ~artifact ~kernel_name =
      if not (List.mem kernel_name artifact.entry_points) then
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
          Ok { native_kernel = pso; name = kernel_name }
        with exn -> Error (Printexc.to_string exn)

    let launch_kernel ?local_dims ~device_info ~global_dims ~args kernel =
      let di = device_info in
      let cb = Metal.CommandBuffer.on_queue di.queue in
      let enc = Metal.ComputeCommandEncoder.on_buffer cb in
      Metal.ComputeCommandEncoder.set_compute_pipeline_state enc
        kernel.native_kernel;
      List.iteri
        (fun i (Any_Device_Buffer b) ->
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
end

include B
