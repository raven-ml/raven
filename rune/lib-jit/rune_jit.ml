(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune_jit.ml *)

module Ir = Ir
module Dtype = Ir.Dtype
module Var = Ir.Var
module Backend_intf = Backend_intf
module Shape_expr = Shape_expr

type 'a kernel_artifact = {
  kernel_id : int;
  kernel_name : string;
  compiled : 'a; (* backend callable kernel *)
  arg_order : Var.t list; (* inputs first, then outputs *)
  global_dims : int array; (* [|gx; gy; gz|] from Scheduled.context *)
  local_dims : int array option;
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

(* ───── LEGACY (optional) tinygrad-style path ───── *)

let compile_legacy (type callable_kernel_native)
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
  let specs = Grouper.group graph in
  let dev = B.Device_info.get_default () in
  let opts = B.Compiler.default_options dev in

  let compile_kernel (spec : Grouper.cluster_t) =
    let lowered =
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
        kernel_id = -1;
        kernel_name = spec.name;
        compiled = kern;
        arg_order = spec.inputs @ spec.outputs;
        global_dims = [| 128; 1; 1 |];
        local_dims = None;
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

(* ───── NEW: Scheduled IR pipeline ───── *)

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
  (* 1) Build Scheduled IR *)
  let scheduled : Ir.Scheduled.graph_t = Schedule.build graph in
  let dev = B.Device_info.get_default () in
  let opts = B.Compiler.default_options dev in

  (* 2) Compile scheduled items (only kernels for now) *)
  let compile_item (it : Ir.Scheduled.schedule_item) =
    match it.operation with
    | Ir.Scheduled.S_Kernel
        { kernel_id; kernel_name; ops; inputs; outputs; context; _ } ->
        (* Bridge to existing Lowerer: synthesize a cluster_t *)
        let input_vars =
          List.map (fun (b : Ir.Scheduled.buffer_info) -> b.buf_var) inputs
        in
        let output_vars =
          List.map (fun (b : Ir.Scheduled.buffer_info) -> b.buf_var) outputs
        in
        let spec : Grouper.cluster_t =
          {
            name = kernel_name;
            nodes = ops;
            inputs = input_vars;
            outputs = output_vars;
            vars_metadata = scheduled.vars_metadata;
          }
        in
        let lowered =
          Lowerer.lower_kernel ~kernel_spec:spec
            ~original_graph_vars_metadata:scheduled.vars_metadata
        in
        let src =
          B.Renderer.render ~device_info:dev ~lowered_ir:lowered ~kernel_name
        in
        let* art =
          B.Compiler.compile ~device_info:dev ~source_code:src ~options:opts
        in
        let* kern = B.Runtime.get_kernel ~artifact:art ~kernel_name in
        Ok
          (Some
             {
               kernel_id;
               kernel_name;
               compiled = kern;
               arg_order = input_vars @ output_vars;
               global_dims = context.global_dims;
               local_dims = Some context.local_dims;
             })
    | _ ->
        (* Skip non-kernel items (transfers/sync) until Multi is wired to
           runtime *)
        Ok None
  in

  let* built =
    result_fold_left
      (fun acc it ->
        let* kopt = compile_item it in
        Ok (kopt :: acc))
      []
      (Array.to_list scheduled.schedule_items)
  in
  let kernels = List.filter_map (fun x -> x) (List.rev built) in

  Ok
    (Executable
       {
         kernels;
         graph_meta = scheduled.vars_metadata;
         graph_outputs = graph.output_vars;
       })

(* ───── Execute ───── *)

let execute (type device_buffer_native callable_kernel_native)
    ~(backend :
       (module Backend_intf.S
          with type device_buffer_native = device_buffer_native
           and type callable_kernel_native = callable_kernel_native))
    (Executable exe) ~inputs ~(outputs : Var.t list) =
  let ( let* ) = Result.bind in
  let module B = (val backend) in
  let dev = B.Device_info.get_default () in
  let live : (Var.t, B.any_device_buffer) Hashtbl.t = Hashtbl.copy inputs in

  (* Allocate a buffer for var v if missing, using graph_meta *)
  let ensure_buffer (v : Var.t) : (B.any_device_buffer, string) result =
    match Hashtbl.find_opt live v with
    | Some b -> Ok b
    | None -> (
        match Hashtbl.find_opt exe.graph_meta v with
        | None -> Error ("Missing metadata for " ^ Var.to_string v)
        | Some { dtype = Dtype.Any_Dtype dt; shape; _ } ->
            let elem_count =
              Array.fold_left ( * ) 1
                (if Array.length shape = 0 then [| 1 |] else shape)
            in
            let bytes = elem_count * Dtype.sizeof_elt dt in
            let* buf =
              B.Runtime.allocate_buffer ~device_info:dev ~size_in_bytes:bytes
                ~dtype:dt
            in
            let any = Backend_intf.Any_Device_Buffer buf in
            Hashtbl.add live v any;
            Ok any)
  in

  let launch (k : _ kernel_artifact) =
    let* args =
      result_fold_left
        (fun acc v ->
          let* b = ensure_buffer v in
          Ok (b :: acc))
        [] k.arg_order
    in
    B.Runtime.launch_kernel ~device_info:dev ~global_dims:k.global_dims
      ?local_dims:k.local_dims ~args:(List.rev args) k.compiled
  in

  let* () = result_fold_left (fun () k -> launch k) () exe.kernels in

  (* Collect requested outputs *)
  let result_tbl = Hashtbl.create (List.length outputs) in
  List.iter
    (fun v ->
      match Hashtbl.find_opt live v with
      | Some b -> Hashtbl.add result_tbl v b
      | None -> (
          (* If a requested output didn’t exist yet, allocate an empty buffer so
             caller can fill *)
          match Hashtbl.find_opt exe.graph_meta v with
          | Some { dtype = Dtype.Any_Dtype dt; shape; _ } -> (
              let elem_count =
                Array.fold_left ( * ) 1
                  (if Array.length shape = 0 then [| 1 |] else shape)
              in
              let bytes = elem_count * Dtype.sizeof_elt dt in
              match
                B.Runtime.allocate_buffer ~device_info:dev ~size_in_bytes:bytes
                  ~dtype:dt
              with
              | Ok buf ->
                  let any = Backend_intf.Any_Device_Buffer buf in
                  Hashtbl.add result_tbl v any;
                  Hashtbl.add live v any
              | Error _ -> ())
          | None -> ()))
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

(* Internal modules exposed for testing *)
module Internal = struct
  module Grouper = Grouper
  module Lowerer = Lowerer
end
