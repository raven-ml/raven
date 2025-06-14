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

(* Generic kernel getter using the nested cache structure *)
let get_kernel ctx category kernel_name =
  (* Get the category cache *)
  let category_cache =
    match Hashtbl.find_opt ctx.Internal.kernels category with
    | Some cache -> cache
    | None -> failwith (Printf.sprintf "Unknown kernel category: %s" category)
  in
  (* Check if kernel exists in cache *)
  match Hashtbl.find_opt category_cache kernel_name with
  | Some func -> func
  | None ->
      (* Compile and cache the kernel *)
      let func = get_function ctx.Internal.library kernel_name in
      Hashtbl.add category_cache kernel_name func;
      func

let get_binary_kernel (type a b) ctx (dtype : (a, b) Nx_core.Dtype.t) op_name =
  let dtype_suffix = Internal.dtype_to_metal_type dtype in
  get_kernel ctx "binary" (Printf.sprintf "%s_%s" op_name dtype_suffix)

let get_unary_kernel (type a b) ctx (dtype : (a, b) Nx_core.Dtype.t) op_name =
  let dtype_suffix = Internal.dtype_to_metal_type dtype in
  get_kernel ctx "unary" (Printf.sprintf "%s_%s" op_name dtype_suffix)

let get_reduce_kernel ctx op_name dtype_suffix =
  get_kernel ctx "reduce" (Printf.sprintf "%s_%s" op_name dtype_suffix)

let get_special_kernel ctx kernel_name =
  (* For special kernels, we use the kernel name as-is *)
  get_kernel ctx "special" kernel_name

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
