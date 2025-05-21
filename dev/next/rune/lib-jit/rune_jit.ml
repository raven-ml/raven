(* rune_jit.ml *)
module Ir = Ir
module Dtype = Ir.Dtype
module Var = Ir.Var

(* ───────────────────── backend interface ──────────────────── *)

module Metal_backend = Metal_backend

(* ───────────────────────── helpers ────────────────────────── *)
let string_contains_substring ~substr str =
  let len_sub = String.length substr and len = String.length str in
  let rec loop i =
    i + len_sub <= len && (String.sub str i len_sub = substr || loop (i + 1))
  in
  len_sub = 0 || (len_sub <= len && loop 0)

let fold_left_result f init xs =
  List.fold_left (fun acc x -> Result.bind acc (fun a -> f a x)) init xs

(* ──────────────────── scheduling & lowering ───────────────── *)
let schedule = Scheduler.schedule

let lower ~kernel_spec ~original_graph_vars_metadata =
  Lowerer.lower_kernel ~kernel_spec ~original_graph_vars_metadata

module Make (Backend : Backend_intf.S) = struct
  (* Short local aliases so downstream code is readable *)
  type ('a, 'b) device_buffer = ('a, 'b) Backend.device_buffer
  type callable_kernel = Backend.callable_kernel

  (* This is the key change: Re-declare any_device_buffer to be definitionally
     equal to Backend.any_device_buffer and expose its constructor. *)
  type any_device_buffer = Backend.any_device_buffer =
    | Any_Device_Buffer :
        ('a_elt, 'b_layout_phantom) device_buffer
        -> any_device_buffer
  [@@unboxed]
  (* Ensure the unboxed attribute is carried over *)

  (* ───────────────────────── rendering ───────────────────────── *)
  let render ~lowered_graph ~kernel_name =
    let device_info = Backend.Device_info.get_default () in
    try
      Ok
        (Backend.Renderer.render ~device_info ~lowered_ir:lowered_graph
           ~kernel_name)
    with ex ->
      Error
        (Printf.sprintf "Rendering '%s' failed: %s" kernel_name
           (Printexc.to_string ex))

  (* ────────────────────── executable types ───────────────────── *)
  type kernel_artefact = {
    spec_name : string;
    compiled_kernel : callable_kernel;
    hl_var_to_kernel_arg_idx : (Var.t, int) Hashtbl.t;
    ordered_hl_buffer_vars_for_args : Var.t list;
  }

  type executable_internal = {
    ordered_kernels_to_run : kernel_artefact list;
    original_graph_vars_metadata : (Var.t, Ir.var_metadata) Hashtbl.t;
    original_graph_output_vars : Var.t list;
  }

  type executable = Executable of executable_internal

  (* ───────────────────────── compilation ──────────────────────── *)

  let compile (graph : Ir.graph_t) : (executable, string) result =
    let ( let* ) = Result.bind in
    let device_info = Backend.Device_info.get_default () in
    let compile_options = Backend.Compiler.default_options device_info in
    let* kernel_specs = schedule graph in

    let compile_one_spec spec =
      let* lowered =
        lower ~kernel_spec:spec
          ~original_graph_vars_metadata:graph.vars_metadata
        |> Result.map_error (fun e -> "lowering-" ^ e)
      in
      let* src =
        render ~lowered_graph:lowered ~kernel_name:spec.name
        |> Result.map_error (fun e -> "render-" ^ e)
      in
      let* artifact =
        Backend.Compiler.compile ~device_info ~source_code:src
          ~options:compile_options
        |> Result.map_error (fun e -> "compile-" ^ e)
      in
      let* kernel =
        Backend.Runtime.get_kernel ~artifact ~kernel_name:spec.name
        |> Result.map_error (fun e -> "get_kernel-" ^ e)
      in
      (* argument ordering: inputs first, then outputs *)
      let ordered_hl_bufs = spec.inputs @ spec.outputs in
      let idx_tbl = Hashtbl.create (List.length ordered_hl_bufs) in
      List.iteri (fun i v -> Hashtbl.add idx_tbl v i) ordered_hl_bufs;
      Ok
        {
          spec_name = spec.name;
          compiled_kernel = kernel;
          hl_var_to_kernel_arg_idx = idx_tbl;
          ordered_hl_buffer_vars_for_args = ordered_hl_bufs;
        }
    in
    let* artefacts =
      fold_left_result
        (fun acc spec ->
          let* a = compile_one_spec spec in
          Ok (a :: acc))
        (Ok []) kernel_specs
    in
    Ok
      (Executable
         {
           ordered_kernels_to_run = List.rev artefacts;
           original_graph_vars_metadata = graph.vars_metadata;
           original_graph_output_vars = graph.output_vars;
         })

  (* ──────────────────────── execution ────────────────────────── *)
  let execute (Executable exe) ~(inputs : (Var.t, any_device_buffer) Hashtbl.t)
      ~(outputs_vars : Var.t list) :
      ((Var.t, any_device_buffer) Hashtbl.t, string) result =
    let ( let* ) = Result.bind in
    let device_info = Backend.Device_info.get_default () in
    let live = Hashtbl.copy inputs in
    (* --- ensure every declared output has a backing buffer --- *)
    let alloc_output hl_var =
      match Hashtbl.find_opt live hl_var with
      | Some _ -> Ok ()
      | None -> (
          match Hashtbl.find_opt exe.original_graph_vars_metadata hl_var with
          | None ->
              Error
                (Printf.sprintf "No metadata for output %s"
                   (Var.to_string hl_var))
          | Some { Ir.dtype = Ir.Dtype.Any_Dtype dt; shape } ->
              let sz = Array.fold_left ( * ) 1 shape * Dtype.sizeof_elt dt in
              let* buf =
                Backend.Runtime.allocate_buffer ~device_info ~size_in_bytes:sz
                  ~dtype:dt
              in
              Hashtbl.add live hl_var (Backend.Any_Device_Buffer buf);
              Ok ())
    in
    let* () =
      fold_left_result (fun () v -> alloc_output v) (Ok ()) outputs_vars
    in
    (* --- helper to fetch/auto-allocate an argument buffer --- *)
    let ensure_buffer hl_var =
      match Hashtbl.find_opt live hl_var with
      | Some b -> Ok b
      | None ->
          Error
            (Printf.sprintf "Missing buffer for kernel argument %s"
               (Var.to_string hl_var))
    in
    (* --- launch kernels in order --- *)
    let launch_kernel art =
      let* arg_bufs =
        fold_left_result
          (fun acc hl ->
            let* b = ensure_buffer hl in
            Ok (b :: acc))
          (Ok []) art.ordered_hl_buffer_vars_for_args
      in
      let arg_bufs = List.rev arg_bufs in
      Backend.Runtime.launch_kernel ~kernel:art.compiled_kernel
        ~global_dims:[| 128; 1; 1 |] ~local_dims:None ~args:arg_bufs
    in
    let* () =
      fold_left_result
        (fun () art -> launch_kernel art)
        (Ok ()) exe.ordered_kernels_to_run
    in
    (* collect only the requested outputs *)
    let result_tbl = Hashtbl.create (List.length outputs_vars) in
    List.iter
      (fun v ->
        match Hashtbl.find_opt live v with
        | Some b -> Hashtbl.add result_tbl v b
        | None ->
            Printf.eprintf
              "Warning: buffer for output %s vanished during execution\n"
              (Var.to_string v))
      outputs_vars;
    Ok result_tbl

  (* ──────────────── utility buffer helpers (public) ───────────── *)
  let allocate_buffer ~size_in_bytes ~(dtype : ('a, 'b) Dtype.t) =
    let device_info = Backend.Device_info.get_default () in
    Backend.Runtime.allocate_buffer ~device_info ~size_in_bytes ~dtype

  let copy_to_device ~(dest_buffer : ('a, 'b) device_buffer)
      ~(host_data : ('a, 'c, 'd) Bigarray.Array1.t) =
    let bytes = Bigarray.Array1.size_in_bytes host_data in
    if bytes = 0 then Ok ()
    else
      let host_ptr =
        Ctypes.(raw_address_of_ptr (to_voidp (bigarray_start array1 host_data)))
      in
      Backend.Runtime.copy_to_device ~dest_buffer ~host_data:host_ptr
        ~host_data_offset_bytes:0 ~copy_size_bytes:bytes

  let copy_from_device ~(src_buffer : ('a, 'b) device_buffer)
      ~(host_dest_bigarray : ('a, 'c, 'd) Bigarray.Array1.t) =
    let bytes = Bigarray.Array1.size_in_bytes host_dest_bigarray in
    if bytes = 0 then Ok ()
    else
      let host_ptr =
        Ctypes.(
          raw_address_of_ptr
            (to_voidp (bigarray_start array1 host_dest_bigarray)))
      in
      Backend.Runtime.copy_from_device ~src_buffer ~host_dest_ptr:host_ptr
        ~device_data_offset_bytes:0 ~copy_size_bytes:bytes
end
