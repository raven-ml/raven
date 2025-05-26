open Metal

(* External declaration for embedded kernel data *)
external metal_kernels_data : unit -> string = "metal_kernels_data"

let kernel_source = lazy (metal_kernels_data ())
let get_function library name = Library.new_function_with_name library name

let compile_library device =
  let source = Lazy.force kernel_source in
  let options = CompileOptions.init () in
  CompileOptions.set_fast_math_enabled options true;
  try Library.on_device device ~source options
  with e ->
    failwith
      (Printf.sprintf "kernels: failed to compile Metal library: %s"
         (Printexc.to_string e))

let get_binary_kernel (type a b) ctx (dtype : (a, b) Nx_core.Dtype.t) op_name =
  let cache_table =
    match dtype with
    | Nx_core.Dtype.Float32 -> ctx.Internal.kernels.binary_f32
    | Nx_core.Dtype.Float64 -> ctx.Internal.kernels.binary_f64
    | Nx_core.Dtype.Int32 -> ctx.Internal.kernels.binary_i32
    | Nx_core.Dtype.Int64 -> ctx.Internal.kernels.binary_i64
    | _ -> failwith "get_binary_kernel: unsupported dtype"
  in

  match Hashtbl.find_opt cache_table op_name with
  | Some func -> func
  | None ->
      let metal_type = Internal.dtype_to_metal_type dtype in
      let kernel_name = Printf.sprintf "%s_%s" op_name metal_type in
      let func = get_function ctx.Internal.library kernel_name in
      Hashtbl.add cache_table op_name func;
      func

let get_unary_kernel (type a b) ctx (dtype : (a, b) Nx_core.Dtype.t) op_name =
  let cache_table =
    match dtype with
    | Nx_core.Dtype.Float32 -> ctx.Internal.kernels.unary_f32
    | Nx_core.Dtype.Float64 -> ctx.Internal.kernels.unary_f64
    | Nx_core.Dtype.Int32 -> ctx.Internal.kernels.unary_i32
    | Nx_core.Dtype.Int64 -> ctx.Internal.kernels.unary_i64
    | _ -> failwith "get_unary_kernel: unsupported dtype"
  in

  match Hashtbl.find_opt cache_table op_name with
  | Some func -> func
  | None ->
      let metal_type = Internal.dtype_to_metal_type dtype in
      let kernel_name = Printf.sprintf "%s_%s" op_name metal_type in
      let func = get_function ctx.Internal.library kernel_name in
      Hashtbl.add cache_table op_name func;
      func

let get_reduce_kernel ctx op_name dtype_suffix =
  let cache_key = Printf.sprintf "%s_%s" op_name dtype_suffix in
  match Hashtbl.find_opt ctx.Internal.kernels.reduce cache_key with
  | Some func -> func
  | None ->
      let func = get_function ctx.Internal.library cache_key in
      Hashtbl.add ctx.Internal.kernels.reduce cache_key func;
      func

let get_special_kernel ctx kernel_name =
  match Hashtbl.find_opt ctx.Internal.kernels.special kernel_name with
  | Some func -> func
  | None ->
      let func = get_function ctx.Internal.library kernel_name in
      Hashtbl.add ctx.Internal.kernels.special kernel_name func;
      func

let create_compute_pipeline device func =
  try
    let pipeline, _reflection =
      ComputePipelineState.on_device_with_function device func
    in
    pipeline
  with e ->
    failwith
      (Printf.sprintf "kernels: failed to create compute pipeline: %s"
         (Printexc.to_string e))
