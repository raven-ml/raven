(* rune_jit.ml *)

module Ir = Ir
module Dtype = Ir.Dtype
module Var = Ir.Var
module Backend_intf = Backend_intf

type 'a kernel_artifact = {
  spec_name : string;
  compiled : 'a;
  arg_order : Var.t list; (* inputs first, then outputs *)
}

type 'a exe_internal = {
  kernels : 'a kernel_artifact list;
  graph_meta : (Var.t, Ir.var_metadata) Hashtbl.t;
  graph_outputs : Var.t list;
}

type 'a executable = Executable of 'a exe_internal

(* helper: monadic left-fold *)
let rec result_fold_left f init = function
  | [] -> Ok init
  | x :: xs ->
      let ( let* ) = Result.bind in
      let* acc = f init x in
      result_fold_left f acc xs

(* ───── top-level API ───── *)

let schedule = Scheduler.schedule

let lower ~kernel_spec ~original_graph_vars_metadata =
  Lowerer.lower_kernel ~kernel_spec ~original_graph_vars_metadata

let compile (type callable_kernel_native)
    ~(backend :
       (module Backend_intf.S
          with type callable_kernel_native = callable_kernel_native))
    (graph : Ir.graph_t) =
  let ( let* ) = Result.bind in
  let module B =
    (val backend
        : Backend_intf.S
        with type callable_kernel_native = callable_kernel_native)
  in
  let specs = Scheduler.schedule graph in
  let dev = B.Device_info.get_default () in
  let opts = B.Compiler.default_options dev in

  let compile_kernel (spec : Scheduler.kernel_spec_t) =
    let* lowered =
      Lowerer.lower_kernel ~kernel_spec:spec
        ~original_graph_vars_metadata:graph.vars_metadata
    in
    let src =
      B.Renderer.render ~device_info:dev ~lowered_ir:lowered
        ~kernel_name:spec.name
    in
    let* art =
      B.Compiler.compile ~device_info:dev ~source_code:src ~options:opts
    in
    let* kern = B.Runtime.get_kernel ~artifact:art ~kernel_name:spec.name in
    Ok
      {
        spec_name = spec.name;
        compiled = kern;
        arg_order = spec.inputs @ spec.outputs;
      }
  in
  let* kernels =
    result_fold_left
      (fun acc spec ->
        let* k = compile_kernel spec in
        Ok (k :: acc))
      [] specs
  in
  Ok
    (Executable
       {
         kernels = List.rev kernels;
         graph_meta = graph.vars_metadata;
         graph_outputs = graph.output_vars;
       })

let execute (type device_buffer_native callable_kernel_native)
    ~(backend :
       (module Backend_intf.S
          with type device_buffer_native = device_buffer_native
           and type callable_kernel_native = callable_kernel_native))
    (Executable exe) ~inputs ~(outputs : Var.t list) =
  let ( let* ) = Result.bind in
  let module B = (val backend) in
  let dev = B.Device_info.get_default () in
  let live = Hashtbl.copy inputs in

  let ensure_output v =
    if Hashtbl.mem live v then Ok ()
    else
      match Hashtbl.find_opt exe.graph_meta v with
      | None -> Error ("Missing metadata for " ^ Var.to_string v)
      | Some { dtype = Dtype.Any_Dtype dt; shape; _ } ->
          let bytes = Array.fold_left ( * ) 1 shape * Dtype.sizeof_elt dt in
          let* buf =
            B.Runtime.allocate_buffer ~device_info:dev ~size_in_bytes:bytes
              ~dtype:dt
          in
          Hashtbl.add live v (Backend_intf.Any_Device_Buffer buf);
          Ok ()
  in
  let* () = result_fold_left (fun () v -> ensure_output v) () outputs in

  let buffer_for v =
    match Hashtbl.find_opt live v with
    | Some b -> Ok b
    | None -> Error ("No buffer for " ^ Var.to_string v)
  in

  let launch k =
    let* args =
      result_fold_left
        (fun acc hl ->
          let* b = buffer_for hl in
          Ok (b :: acc))
        [] k.arg_order
    in
    (* Get the size from the first output variable's shape *)
    let size = 
      match k.arg_order with
      | [] -> 128  (* fallback *)
      | first_var :: _ ->
          match Hashtbl.find_opt exe.graph_meta first_var with
          | None -> 128  (* fallback *)
          | Some { shape; _ } -> Array.fold_left ( * ) 1 shape
    in
    B.Runtime.launch_kernel ~device_info:dev ~global_dims:[| size; 1; 1 |]
      ?local_dims:None ~args:(List.rev args) k.compiled
  in
  let* () = result_fold_left (fun () k -> launch k) () exe.kernels in

  let result_tbl = Hashtbl.create (List.length outputs) in
  List.iter
    (fun v ->
      Option.iter
        (fun b -> Hashtbl.add result_tbl v b)
        (Hashtbl.find_opt live v))
    outputs;
  Ok result_tbl

(* ───── convenience wrappers ───── *)

let allocate_buffer (type device_buffer_native)
    ~(backend :
       (module Backend_intf.S
          with type device_buffer_native = device_buffer_native)) ~size_in_bytes
    ~(dtype : 'a Dtype.t) =
  let module B = (val backend) in
  let device_info = B.Device_info.get_default () in
  B.Runtime.allocate_buffer ~device_info ~size_in_bytes ~dtype

let copy_to_device (type device_buffer_native)
    ~(backend :
       (module Backend_intf.S
          with type device_buffer_native = device_buffer_native)) ~dest_buffer
    ~host =
  let module B = (val backend) in
  let bytes = Bigarray.Array1.size_in_bytes host in
  if bytes = 0 then Ok ()
  else
    let ptr =
      Ctypes.(raw_address_of_ptr (to_voidp (bigarray_start array1 host)))
    in
    B.Runtime.copy_to_device ~dest_buffer ~host_data:ptr
      ~host_data_offset_bytes:0 ~copy_size_bytes:bytes

let copy_from_device (type device_buffer_native)
    ~(backend :
       (module Backend_intf.S
          with type device_buffer_native = device_buffer_native)) ~src_buffer
    ~dest =
  let module B = (val backend) in
  let bytes = Bigarray.Array1.size_in_bytes dest in
  if bytes = 0 then Ok ()
  else
    let ptr =
      Ctypes.(raw_address_of_ptr (to_voidp (bigarray_start array1 dest)))
    in
    B.Runtime.copy_from_device ~src_buffer ~host_dest_ptr:ptr
      ~device_data_offset_bytes:0 ~copy_size_bytes:bytes

module Debug = struct
  let schedule = schedule

  let lower_kernel ~kernel_spec ~original_graph_vars_metadata =
    lower ~kernel_spec ~original_graph_vars_metadata
end
