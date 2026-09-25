(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

type storage = { address : nativeint; host : bool; registered : bool }
let buffer_kind : storage Type.Id.t = Type.Id.make ()
let buffer_address buf = match Device.Buffer.get buffer_kind buf with
  | Some storage -> storage.address | None -> 0n

module Ffi = struct
  external init : unit -> unit = "caml_tolk_cuda_init"
  external device_get : int -> int = "caml_tolk_cuda_device_get"

  external compute_capability : int -> int * int
    = "caml_tolk_cuda_compute_capability"

  external ctx_create : int -> nativeint = "caml_tolk_cuda_ctx_create"

  external ctx_set_current : nativeint -> unit
    = "caml_tolk_cuda_ctx_set_current"

  external ctx_synchronize : unit -> unit = "caml_tolk_cuda_ctx_synchronize"
  external mem_alloc : int -> nativeint = "caml_tolk_cuda_mem_alloc"
  external mem_free : nativeint -> unit = "caml_tolk_cuda_mem_free"
  external mem_host_alloc : int -> nativeint = "caml_tolk_cuda_mem_host_alloc"
  external mem_free_host : nativeint -> unit = "caml_tolk_cuda_mem_free_host"

  external mem_host_register : nativeint -> int -> bool = "caml_tolk_cuda_mem_host_register"
  external mem_host_unregister : nativeint -> unit = "caml_tolk_cuda_mem_host_unregister"
  external enable_peer : int -> int -> nativeint -> bool = "caml_tolk_cuda_enable_peer"
  external memcpy_peer : nativeint -> nativeint -> nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_peer"

  external host_write : nativeint -> bytes -> unit
    = "caml_tolk_cuda_host_write"

  external memcpy_htod_async : nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_htod_async"

  external memcpy_dtoh_ptr : nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_dtoh_ptr"

  external host_read : bytes -> nativeint -> unit
    = "caml_tolk_cuda_host_read"

  external memcpy_async : nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_async"

  external module_load : bytes -> nativeint = "caml_tolk_cuda_module_load"

  external program_create : nativeint -> string -> int -> int array -> nativeint
    = "caml_tolk_cuda_program_create"

  external program_free : nativeint -> unit = "caml_tolk_cuda_program_free"

  external module_unload : nativeint -> unit = "caml_tolk_cuda_module_unload"

  external launch_kernel :
    nativeint ->
    nativeint array ->
    int64 array ->
    int array ->
    int array ->
    bool ->
    float option = "caml_tolk_cuda_launch_kernel_bc" "caml_tolk_cuda_launch_kernel"

  external graph_create : int -> nativeint = "caml_tolk_cuda_graph_create"

  external graph_add_kernel :
    nativeint ->
    nativeint ->
    int array ->
    int array ->
    nativeint array ->
    int64 array ->
    int array ->
    int = "caml_tolk_cuda_graph_add_kernel_bc" "caml_tolk_cuda_graph_add_kernel"

  external graph_add_copy :
    nativeint ->
    nativeint ->
    nativeint ->
    nativeint ->
    int ->
    int array ->
    int = "caml_tolk_cuda_graph_add_copy_bc" "caml_tolk_cuda_graph_add_copy"

  external graph_instantiate : nativeint -> unit
    = "caml_tolk_cuda_graph_instantiate"

  external graph_set_buf : nativeint -> int -> int -> nativeint -> unit
    = "caml_tolk_cuda_graph_set_buf"

  external graph_set_val : nativeint -> int -> int -> int64 -> unit
    = "caml_tolk_cuda_graph_set_val"

  external graph_set_launch : nativeint -> int -> int array -> int array -> unit
    = "caml_tolk_cuda_graph_set_launch"

  external graph_set_params : nativeint -> int -> unit
    = "caml_tolk_cuda_graph_set_params"

  external graph_launch : nativeint -> bool -> float option
    = "caml_tolk_cuda_graph_launch"

  external graph_destroy : nativeint -> unit = "caml_tolk_cuda_graph_destroy"
end

module State = struct
  type t = {
    name : string;
    device : int;
    context : nativeint;
    arch : string;
    peers : (nativeint, bool) Hashtbl.t;
    mutable pending_copyin : (storage * int * Device.Buffer_spec.t) list;
    (* The device's LRU-wrapped allocator; set right after creation and used
       by copyin staging and pending-buffer release. *)
    mutable allocator : storage Device.Allocator.t option;
  }

  let devices : t list ref = ref []

  let create name device_id =
    Ffi.init ();
    let cu_device = Ffi.device_get device_id in
    let context = Ffi.ctx_create cu_device in
    let major, minor = Ffi.compute_capability cu_device in
    let arch = Printf.sprintf "sm_%d%d" major minor in
    let state = { name = Device.canonicalize name; device = cu_device; context; arch;
      peers = Hashtbl.create 4; pending_copyin = []; allocator = None } in
    devices := !devices @ [ state ];
    state

  let synchronize t =
    Ffi.ctx_set_current t.context;
    Ffi.ctx_synchronize ();
    let pending = t.pending_copyin in
    t.pending_copyin <- [];
    List.iter
      (fun (buf, size, spec) ->
        (Option.get t.allocator).Device.Allocator.free buf size spec)
      pending

  let synchronize_system () = List.iter synchronize !devices

  let find name = List.find_opt (fun state -> state.name = Device.canonicalize name) !devices

  let enable_peer dst src =
    if dst.context = src.context then true else
    match Hashtbl.find_opt dst.peers src.context with
    | Some supported -> supported
    | None ->
        Ffi.ctx_set_current dst.context;
        let supported = Ffi.enable_peer dst.device src.device src.context in
        Hashtbl.add dst.peers src.context supported;
        supported
end

module Allocator = struct
  let host_spec = { Device.Buffer_spec.default with host = true }

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
      State.synchronize state;
      match spec.Device.Buffer_spec.external_ptr with
      | Some _ -> ()
      | None -> if buf.host then Ffi.mem_free_host buf.address else Ffi.mem_free buf.address
    in
    let copyin buf bytes =
      if buf.host then begin
        State.synchronize state;
        Ffi.host_write buf.address bytes
      end else begin
        Ffi.ctx_set_current state.State.context;
        let size = Bytes.length bytes in
        let host = (Option.get state.State.allocator).Device.Allocator.alloc size host_spec in
        state.State.pending_copyin <- (host, size, host_spec) :: state.State.pending_copyin;
        Ffi.host_write host.address bytes;
        Ffi.memcpy_htod_async buf.address host.address size
      end
    in
    let copyout bytes buf =
      State.synchronize_system ();
      if buf.host then Ffi.host_read bytes buf.address else begin
        Ffi.ctx_set_current state.State.context;
        let size = Bytes.length bytes in
        let allocator = Option.get state.State.allocator in
        let host = allocator.Device.Allocator.alloc size host_spec in
        Fun.protect ~finally:(fun () -> allocator.free host size host_spec)
          (fun () -> Ffi.memcpy_dtoh_ptr host.address buf.address size;
                     Ffi.host_read bytes host.address)
      end
    in
    let transfer ~dest ~src ~dest_device ~src_device nbytes =
      match State.find dest_device, State.find src_device with
      | Some dst, Some source when State.enable_peer dst source ->
          if dst.context = source.context then begin
            Ffi.ctx_set_current dst.context;
            Ffi.memcpy_async dest.address src.address nbytes
          end else begin
            State.synchronize source;
            State.synchronize dst;
            Ffi.memcpy_peer dest.address dst.context src.address source.context nbytes;
            State.synchronize dst
          end;
          true
      | _ -> false
    in
    let offset buf size byte_offset =
      ignore size;
      if byte_offset < 0 then invalid_arg "CUDA buffer offset must be non-negative";
      {buf with address = Nativeint.add buf.address (Nativeint.of_int byte_offset);
        registered = false}
    in
    let map source =
      Ffi.ctx_set_current state.State.context;
      match State.find (Device.Buffer.device source) with
      | Some owner ->
          let raw = Option.get (Device.Buffer.get buffer_kind source) in
          if not (raw.host || State.enable_peer state owner) then
            invalid_arg "CUDA peer storage is not accessible";
          {raw with registered = false}
      | None ->
          (match Device.Buffer.host_addr source with
           | None -> invalid_arg "CUDA mapping requires host-accessible storage"
           | Some address ->
               Ffi.ctx_set_current state.State.context;
               let registered = Ffi.mem_host_register address (Device.Buffer.nbytes source) in
               {address; host = true; registered})
    in
    let unmap raw =
      State.synchronize state;
      if raw.registered then Ffi.mem_host_unregister raw.address
    in
    Device.Allocator.{kind = buffer_kind;
      host = (fun (buf : storage) -> if buf.host then Some buf.address else None);
      mapping = Some {map; unmap};
      synchronize = (fun () -> State.synchronize_system ());
      alloc; free; copyin; copyout;
      addr = Some (fun buf -> buf.address); offset = Some offset;
      transfer = Some transfer; supports_transfer = true;
      copy_from_disk = None; supports_copy_from_disk = false}

  let create state =
    let allocator = Device.Lru_allocator.wrap (raw state) in
    state.State.allocator <- Some allocator;
    Device.Allocator.Pack allocator
end

module Program = struct
  let runtime state (obj : Tolk_uop.Tiny_elf.t) =
    let entry_name = obj.name and lib = obj.lib in
    let fields = Tolk_uop.Tiny_elf.layout obj.signature in
    let nbufs = List.fold_left (fun n (arg : Tolk_uop.Tiny_elf.argument) ->
        n + if arg.addrspace = Tolk_uop.Dtype.Alu then 0 else 1) 0 obj.signature in
    let layout = Array.of_list (List.concat_map (fun (field : Tolk_uop.Tiny_elf.field) ->
        [ field.argument.slot; field.offset; field.size ]) fields) in
    Ffi.ctx_set_current state.State.context;
    let module_ = Ffi.module_load lib in
    let func = try Ffi.program_create module_ entry_name nbufs layout
      with exn -> Ffi.module_unload module_; raise exn in
    let default_local = [| 1; 1; 1 |] in
    let unloaded = ref false in
    let call bufs ~global ~local ~vals ~wait ~timeout:_ =
      let bufs = Array.map buffer_address bufs in
      if !unloaded then invalid_arg "CUDA program has been unloaded";
      let local = Option.value local ~default:default_local in
      Ffi.ctx_set_current state.State.context;
      Ffi.launch_kernel func bufs vals global local
        wait
    in
    let free () =
      if not !unloaded then begin
        Ffi.ctx_set_current state.State.context;
        unloaded := true;
        Fun.protect ~finally:(fun () -> Ffi.program_free func)
          (fun () -> Ffi.module_unload module_)
      end in
    Device.{ call; free; handle = func }
end

module Graph = struct
  (* Batched replay through CUDA execution graphs: kernel launches become
     kernel nodes and buffer copies become device-to-device memcpy nodes,
     instantiated once and relaunched with a single driver call. *)
  let build state (nodes : Device.Graph.node array) =
    Ffi.ctx_set_current state.State.context;
    let g = Ffi.graph_create (Array.length nodes) in
    (try
       Array.iter
         (function
           | Device.Graph.Kernel { handle; global; local; bufs; vals; deps } ->
               ignore (Ffi.graph_add_kernel g handle global local
                         (Array.map buffer_address bufs)
                         (Array.map Int64.of_int vals) deps : int)
           | Device.Graph.Copy { dest; src; nbytes; deps } ->
               ignore (Ffi.graph_add_copy g state.State.context
                         (buffer_address dest)
                         (buffer_address src) nbytes
                         deps : int))
         nodes;
       Ffi.graph_instantiate g
     with exn -> Ffi.graph_destroy g; raise exn);
    let exec =
      {
        Device.Graph.set_buf = (fun node pos buf ->
          Ffi.graph_set_buf g node pos
            (buffer_address buf));
        set_val = (fun node idx v -> Ffi.graph_set_val g node idx (Int64.of_int v));
        set_launch_dims = (fun node ~global ~local ->
          Ffi.graph_set_launch g node global local);
        set_params = (fun node -> Ffi.graph_set_params g node);
        launch = (fun ~wait ->
          Ffi.ctx_set_current state.State.context;
          Ffi.graph_launch g wait);
      }
    in
    Gc.finalise (fun (_ : Device.Graph.exec) -> Ffi.graph_destroy g) exec;
    exec

  let create state =
    {
      Device.Graph.supports_copy = true;
      max_buffer_offset = None;
      build = build state;
    }
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
  let state = State.create name device_id in
  let allocator = Allocator.create state in
  let renderer_set = Device.Renderer_set.make ~device:name ~arch:state.State.arch
      [ "CUDA", (fun target ->
          let arch = match Gpu_target.parse_cuda_arch target.Tolk_uop.Target.arch with
            | Some arch -> arch
            | None -> invalid_arg ("unsupported CUDA architecture: " ^ target.arch) in
          let compiler = Tolk_nvrtc.Compiler_nvrtc.create ~cache_key:"cuda" target.arch in
          Renderer.with_compiler compiler (Cstyle.cuda arch)) ] in
  let runtime = Program.runtime state in
  let synchronize () = State.synchronize state in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ~graph:(Graph.create state) ()
