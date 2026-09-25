(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

type storage = { address : nativeint; host : bool; registered : bool }
let buffer_kind : storage Type.Id.t = Type.Id.make ()
let buffer_address ~device buf = match Device.Buffer.get ~device buffer_kind buf with
  | Some storage -> storage.address | None -> 0n

module Ffi = struct
  external init : unit -> unit = "caml_tolk_cuda_init"
  external device_get : int -> int = "caml_tolk_cuda_device_get"

  external compute_capability : int -> int * int
    = "caml_tolk_cuda_compute_capability"

  external ctx_create : int -> nativeint = "caml_tolk_cuda_ctx_create"

  external ctx_set_current : nativeint -> unit
    = "caml_tolk_cuda_ctx_set_current"

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

  external memcpy_htod_async : nativeint -> nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_htod_async"

  external memcpy_dtoh_ptr : nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_dtoh_ptr"

  external host_read : bytes -> nativeint -> unit
    = "caml_tolk_cuda_host_read"

  external memcpy_async : nativeint -> nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_async"

  external module_load : bytes -> nativeint = "caml_tolk_cuda_module_load"

  external program_create : nativeint -> string -> int -> int array -> nativeint
    = "caml_tolk_cuda_program_create"

  external program_free : nativeint -> unit = "caml_tolk_cuda_program_free"

  external module_unload : nativeint -> unit = "caml_tolk_cuda_module_unload"

  external launch_kernel :
    nativeint ->
    nativeint ->
    nativeint array ->
    int64 array ->
    int array ->
    int array ->
    bool ->
    float option = "caml_tolk_cuda_launch_kernel_bc" "caml_tolk_cuda_launch_kernel"

  external hcq_create : nativeint -> nativeint = "caml_tolk_cuda_hcq_create"
  external hcq_synchronize : nativeint -> unit = "caml_tolk_cuda_hcq_synchronize"
  external hcq_symbol : string -> nativeint = "caml_tolk_cuda_hcq_symbol"
  external program_function : nativeint -> nativeint = "caml_tolk_cuda_program_function"

end

module State = struct
  type t = {
    name : string;
    device : int;
    context : nativeint;
    queue : nativeint;
    mutable timeline : Device.Buffer.t option;
    mutable handles : Device.Buffer.t option;
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
    let queue = Ffi.hcq_create context in
    let state = { queue; timeline = None; handles = None; name = Device.canonicalize name; device = cu_device; context; arch;
      peers = Hashtbl.create 4; pending_copyin = []; allocator = None } in
    devices := !devices @ [ state ];
    state

  let synchronize t =
    Ffi.ctx_set_current t.context;
    Ffi.hcq_synchronize t.queue;
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
        Ffi.memcpy_htod_async state.queue buf.address host.address size
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
            Ffi.memcpy_async dst.queue dest.address src.address nbytes
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
      let bufs = Array.map (buffer_address ~device:state.State.name) bufs in
      if !unloaded then invalid_arg "CUDA program has been unloaded";
      let local = Option.value local ~default:default_local in
      Ffi.ctx_set_current state.State.context;
      Ffi.launch_kernel state.queue func bufs vals global local
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

module Queue = struct
  open Tolk_uop
  module U = Uop
  module B = Device.Buffer

  let word ?(release = fun () -> ()) value =
    let base = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
    let allocator = {base with free = (fun address size spec ->
        release (); base.free address size spec)} in
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
        Some (word (Ffi.hcq_symbol symbol))
    | Some {param = {allocation = Some ("cuda_context", _); _}; _} ->
        let b = match state.State.handles with
          | Some b -> b
          | None -> let b = word state.queue in state.handles <- Some b; b in
        Some b
    | Some {param = {allocation = Some ("cuda_function", data); _}; _} ->
        let object_ = (Marshal.from_string data 0 : Tiny_elf.t) in
        let program = Program.runtime state object_ in
        (try Some (word (Ffi.program_function program.handle)
            ~release:(fun () -> State.synchronize state; program.free ()))
         with exn -> program.free (); raise exn)
    | Some _ when U.node_tag u = Some "timeline" ->
        let b = match state.State.timeline with
          | Some b -> b
          | None ->
              let spec = {Device.Buffer_spec.default with host = true; nolru = true} in
              let b = B.create ~device:name ~size:2 ~dtype:Dtype.uint64 ~spec
                  (Device.Allocator.Pack (Allocator.raw state)) in
              B.ensure_allocated b;
              B.copyin b (Bytes.make 16 '\000'); state.timeline <- Some b; b in
        Some b
    | _ -> None

  let create state device_name =
    let host = try Device.get "CPU" with Failure _ -> Tolk_cpu.create "CPU" in
    let copy call = match U.as_call call with
      | Some {args; _} -> List.for_all (fun arg ->
          U.device_of arg = Some (U.Single state.State.name)) args
      | None -> false in
    Device.{host = Device.name host; copy; encode = Cuda_queue.encode device_name; lower = Cuda_queue.lower device_name;
      compile = Codegen.to_program ~optimize:false host (Device.renderer host)}
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
    ~queue:(Queue.create state name) ~bufferize:(Queue.bufferize state name) ()
