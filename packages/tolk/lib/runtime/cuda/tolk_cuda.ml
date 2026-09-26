(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

type storage = { address : nativeint; host : bool; registered : bool }
let buffer_kind : storage Type.Id.t = Type.Id.make ()

module Ffi = struct
  external init : unit -> unit = "caml_tolk_cuda_init"
  external device_get : int -> int = "caml_tolk_cuda_device_get"

  external compute_capability : int -> int * int
    = "caml_tolk_cuda_compute_capability"

  external ctx_create : int -> nativeint = "caml_tolk_cuda_ctx_create"
  external ctx_destroy : nativeint -> unit = "caml_tolk_cuda_ctx_destroy"

  external ctx_set_current : nativeint -> unit
    = "caml_tolk_cuda_ctx_set_current"

  external mem_alloc : int -> nativeint = "caml_tolk_cuda_mem_alloc"
  external mem_free : nativeint -> unit = "caml_tolk_cuda_mem_free"
  external mem_host_alloc : int -> nativeint = "caml_tolk_cuda_mem_host_alloc"
  external mem_free_host : nativeint -> unit = "caml_tolk_cuda_mem_free_host"

  external mem_host_register : nativeint -> int -> int = "caml_tolk_cuda_mem_host_register"
  external mem_host_unregister : nativeint -> unit = "caml_tolk_cuda_mem_host_unregister"
  external host_read : bytes -> nativeint -> unit = "caml_tolk_cuda_host_read"
  external module_load : string -> nativeint = "caml_tolk_cuda_module_load"

  external module_function : nativeint -> string -> nativeint
    = "caml_tolk_cuda_module_function"

  external module_unload : nativeint -> unit = "caml_tolk_cuda_module_unload"

  external hcq_create : nativeint -> nativeint = "caml_tolk_cuda_hcq_create"
  external hcq_destroy : nativeint -> unit = "caml_tolk_cuda_hcq_destroy"
  external hcq_await : nativeint -> nativeint -> int64 -> int -> unit = "caml_tolk_cuda_hcq_await"
  external hcq_synchronize : nativeint -> unit = "caml_tolk_cuda_hcq_synchronize"
  external hcq_symbol : string -> nativeint = "caml_tolk_cuda_hcq_symbol"
  external profile_clock : unit -> float = "caml_tolk_cuda_profile_clock"

end

module State = struct
  type t = {
    operation_owner : Tolk_uop.Storage.Owner.t;
    context : nativeint;
    queue : nativeint;
    mutable closed : bool;
    functions : (string * string, Device.Buffer.t) Hashtbl.t;
    function_lock : Mutex.t;
    mutable timeline : Device.Buffer.t option;
    mutable handles : Device.Buffer.t option;
    arch : string;
  }

  let create device_id =
    Ffi.init ();
    let cu_device = Ffi.device_get device_id in
    let context = Ffi.ctx_create cu_device in
    try
      let major, minor = Ffi.compute_capability cu_device in
      let arch = Printf.sprintf "sm_%d%d" major minor in
      let queue = Ffi.hcq_create context in
      (try
         let state = {
           operation_owner = Tolk_uop.Storage.Owner.create ();
           queue; closed = false; timeline = None; handles = None;
           functions = Hashtbl.create 16; function_lock = Mutex.create ();
           context; arch } in
         state
       with exn ->
         let bt = Printexc.get_raw_backtrace () in
         Ffi.hcq_destroy queue;
         Printexc.raise_with_backtrace exn bt)
    with exn ->
      let bt = Printexc.get_raw_backtrace () in
      Ffi.ctx_destroy context;
      Printexc.raise_with_backtrace exn bt

  let with_function_lock t f =
    Tolk_uop.Storage.with_operation (fun () -> Mutex.protect t.function_lock f)

  let timeline t = with_function_lock t (fun () -> t.timeline)

  let synchronize t = Tolk_uop.Storage.Owner.run [t.operation_owner] (fun () ->
    if not t.closed then begin
      Ffi.ctx_set_current t.context;
      Ffi.hcq_synchronize t.queue
    end)

  let shutdown t = Tolk_uop.Storage.Owner.run [t.operation_owner] (fun () ->
    let retire = Mutex.protect t.function_lock (fun () ->
        if t.closed then false else begin t.closed <- true; true end) in
    if retire then begin
      Fun.protect ~finally:(fun () ->
          Ffi.ctx_destroy t.context;
          Mutex.protect t.function_lock (fun () -> Hashtbl.clear t.functions))
        (fun () -> Ffi.hcq_destroy t.queue)
    end)

end

module Allocator = struct
  let raw state =
    let alloc size spec =
      Ffi.ctx_set_current state.State.context;
      let host = spec.Device.Buffer_spec.host || spec.cpu_access in
      let address = match spec.external_ptr with
        | Some ptr -> ptr
        | None -> if host then Ffi.mem_host_alloc size else Ffi.mem_alloc size in
      {address; host; registered = false}
    in
    let free buf size spec =
      ignore size;
      if not state.State.closed then begin
        State.synchronize state;
        match spec.Device.Buffer_spec.external_ptr with
        | Some _ -> ()
        | None -> if buf.host then Ffi.mem_free_host buf.address else Ffi.mem_free buf.address
      end
    in
    let offset buf size byte_offset =
      ignore size;
      if byte_offset < 0 then invalid_arg "CUDA buffer offset must be non-negative";
      {buf with address = Nativeint.add buf.address (Nativeint.of_int byte_offset);
        registered = false}
    in
    let map source =
      Ffi.ctx_set_current state.State.context;
      let Device.Allocator.Pack source_allocator = Device.Buffer.allocator source in
      match Type.Id.provably_equal source_allocator.kind buffer_kind with
      | Some Type.Equal ->
          let raw = Option.get (Device.Buffer.get buffer_kind source) in
          if not raw.host then
            raise (Tolk_uop.Storage.Mapping_unavailable "CUDA device storage requires host staging");
          {raw with registered = false}
      | None ->
          (match Device.Buffer.host_addr source with
           | None -> raise (Tolk_uop.Storage.Mapping_unavailable "CUDA mapping requires host-accessible storage")
           | Some address ->
               Ffi.ctx_set_current state.State.context;
               let registration = Ffi.mem_host_register address (Device.Buffer.nbytes source) in
               if registration < 0 then
                 raise (Tolk_uop.Storage.Mapping_unavailable "CUDA cannot register this host range");
               {address; host = true; registered = registration = 1})
    in
    let unmap raw =
      if not state.State.closed then begin
        State.synchronize state;
        if raw.registered then Ffi.mem_host_unregister raw.address
      end
    in
    Device.Allocator.{owner = state.State.operation_owner; kind = buffer_kind;
      host = (fun (buf : storage) -> if buf.host then Some buf.address else None);
      mapping = Some {map; unmap};
      synchronize = (fun () -> State.synchronize state);
      alloc; free;
      addr = Some (fun buf -> buf.address); offset = Some offset }

  let create state =
    Device.Allocator.Pack (Device.Lru_allocator.wrap (raw state))
end

module Queue = struct
  open Tolk_uop
  module U = Uop
  module B = Device.Buffer

  let word state value =
    let allocator = { (Storage.Host_allocator.make ~synchronize:(fun () -> ()))
        with owner = state.State.operation_owner } in
    let b = B.create ~device:"CPU" ~size:1 ~dtype:Dtype.uint64
        (Device.Allocator.Pack allocator) in
    let bytes = Bytes.create 8 in
    Bytes.set_int64_le bytes 0 (Int64.of_nativeint value);
    B.ensure_allocated b;
    B.copyin b bytes;
    b

  let bufferize state name u = match U.as_param u with
    | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
        let libs, symbol = (Marshal.from_string data 0 : string list * string) in
        if libs <> [] then invalid_arg "CUDA host helpers do not load libraries";
        Some (word state (Ffi.hcq_symbol symbol))
    | Some {param = {allocation = Some ("cuda_context", _); _}; _} ->
        Some (State.with_function_lock state (fun () ->
            if state.State.closed then invalid_arg "CUDA device is closed";
            match state.State.handles with
            | Some buffer -> buffer
            | None ->
                let buffer = word state state.queue in
                state.handles <- Some buffer;
                buffer))
    | Some {param = {allocation = Some ("cuda_function", data); _}; _} ->
        let object_ = (Marshal.from_string data 0 : Tiny_elf.t) in
        let key = Bytes.to_string object_.lib, object_.name in
        Some (Mutex.protect state.State.function_lock (fun () ->
            if state.State.closed then invalid_arg "CUDA device is closed";
            match Hashtbl.find_opt state.State.functions key with
            | Some buffer -> buffer
            | None ->
                Ffi.ctx_set_current state.State.context;
                let module_ = Ffi.module_load (fst key) in
                try
                  let buffer = word state (Ffi.module_function module_ object_.name) in
                  Hashtbl.add state.State.functions key buffer;
                  buffer
                with exn ->
                  let backtrace = Printexc.get_raw_backtrace () in
                  Ffi.module_unload module_;
                  Printexc.raise_with_backtrace exn backtrace))
    | Some _ when U.node_tag u = Some "timeline" ->
        Some (State.with_function_lock state (fun () ->
            if state.State.closed then invalid_arg "CUDA device is closed";
            match state.State.timeline with
            | Some buffer -> buffer
            | None ->
                let spec = {Device.Buffer_spec.default with host = true; nolru = true} in
                let buffer = B.create ~device:name ~size:2 ~dtype:Dtype.uint64 ~spec
                    (Device.Allocator.Pack (Allocator.raw state)) in
                B.ensure_allocated buffer;
                B.copyin buffer (Bytes.make 16 '\000');
                state.timeline <- Some buffer;
                buffer))
    | _ -> None

  let create state device_name =
    let host = try Device.get "CPU" with Failure _ -> Tolk_cpu.create "CPU" in
    let copy call =
      let supported = match U.as_call call with
      | Some {args; _} -> List.for_all (fun arg ->
          match U.device_of arg with
          | Some (U.Single name) ->
              List.mem (List.hd (String.split_on_char ':' name)) ["CUDA"; "CPU"]
          | _ -> false) args
      | None -> false in
      if supported then Some "COPY:0" else None in
    let completion () =
      match State.timeline state with
      | None -> Fun.const ()
      | Some timeline ->
          let address = B.addr timeline in
          let bytes = Bytes.create 8 in
          Ffi.host_read bytes (Nativeint.add address 8n);
          let value = Bytes.get_int64_le bytes 0 in
          fun timeout ->
            ignore timeout;
            Ffi.hcq_await state.State.queue address value (Helpers.getenv "HCQDEV_WAIT_TIMEOUT_MS" 30000);
            ignore (Sys.opaque_identity timeline) in
    Device.{timestamp_divider = 1000.; profile_offset = (fun () -> Profile.calibrate (fun () -> Ffi.profile_clock));
      completion; prepare = (fun () -> ()); host = Device.name host; max_kernel_bindings = None; config = (fun () -> ""); copy; encode = Cuda_queue.encode device_name; lower = Cuda_queue.lower device_name;
      compile = Codegen.to_program ~optimize:false (Device.renderer host)}
end


let create name =
  let device_id =
    match String.index_opt name ':' with
    | Some i -> (
        let suffix = String.sub name (i + 1) (String.length name - i - 1) in
        match int_of_string_opt suffix with
        | Some id -> id
        | None -> invalid_arg (Printf.sprintf "invalid CUDA device %S" name))
    | None -> 0
  in
  let state = State.create device_id in
  try
    let allocator = Allocator.create state in
    let renderer_set = Device.Renderer_set.make ~device:name ~arch:state.State.arch
        [ "CUDA", (fun target ->
            let arch = match Gpu_target.parse_cuda_arch target.Tolk_uop.Target.arch with
              | Some arch -> arch
              | None -> invalid_arg ("unsupported CUDA architecture: " ^ target.arch) in
            let compiler = Tolk_nvrtc.Compiler_nvrtc.create ~cache_key:"cuda" target.arch in
            Renderer.with_compiler compiler (Cstyle.cuda arch)) ] in
    let synchronize () = State.synchronize state in
    let device = Device.make ~name ~allocator ~renderer_set ~synchronize:(fun timeout -> ignore timeout; synchronize ())
      ~queue:(Queue.create state name) ~bufferize:(Queue.bufferize state name) () in
    at_exit (fun () -> State.shutdown state);
    device
  with exn ->
    let bt = Printexc.get_raw_backtrace () in
    State.shutdown state;
    Printexc.raise_with_backtrace exn bt
