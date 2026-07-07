(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

module Ffi = struct
  external create_device : unit -> nativeint = "caml_tolk_metal_create_device"
  external release_device : nativeint -> unit = "caml_tolk_metal_release_device"

  external create_command_queue : nativeint -> nativeint
    = "caml_tolk_metal_create_command_queue"

  external release_command_queue : nativeint -> unit
    = "caml_tolk_metal_release_command_queue"

  external buffer_alloc : nativeint -> int -> nativeint
    = "caml_tolk_metal_buffer_alloc"

  external buffer_free : nativeint -> unit = "caml_tolk_metal_buffer_free"

  external buffer_copyin : nativeint -> int -> bytes -> unit
    = "caml_tolk_metal_buffer_copyin"

  external buffer_copyout : bytes -> nativeint -> int -> unit
    = "caml_tolk_metal_buffer_copyout"

  external program_create : nativeint -> string -> bytes -> nativeint
    = "caml_tolk_metal_program_create"

  external program_free : nativeint -> unit = "caml_tolk_metal_program_free"

  external program_dispatch :
    nativeint ->
    nativeint ->
    nativeint array ->
    int array ->
    int array ->
    int array ->
    int array ->
    nativeint
    = "caml_tolk_metal_program_dispatch_bc" "caml_tolk_metal_program_dispatch"

  external command_buffer_wait : nativeint -> unit
    = "caml_tolk_metal_command_buffer_wait"

  external compile : string -> bytes option = "caml_tolk_metal_compile"

  external icb_create : nativeint -> int -> nativeint
    = "caml_tolk_metal_icb_create"

  external icb_encode :
    nativeint ->
    int ->
    nativeint ->
    nativeint array ->
    int array ->
    nativeint ->
    int array ->
    int array ->
    int array ->
    unit = "caml_tolk_metal_icb_encode_bc" "caml_tolk_metal_icb_encode"

  external icb_update_buffer :
    nativeint -> int -> int -> nativeint -> int -> unit
    = "caml_tolk_metal_icb_update_buffer"

  external icb_update_dispatch :
    nativeint -> int -> int array -> int array -> unit
    = "caml_tolk_metal_icb_update_dispatch_bc"
      "caml_tolk_metal_icb_update_dispatch"

  external icb_execute :
    nativeint ->
    nativeint ->
    int ->
    nativeint array ->
    nativeint array ->
    nativeint = "caml_tolk_metal_icb_execute"

  external icb_release : nativeint -> unit = "caml_tolk_metal_icb_release"
  external needs_icb_fix : nativeint -> bool = "caml_tolk_metal_needs_icb_fix"

  external blit_copy :
    nativeint -> nativeint -> int -> nativeint -> int -> int -> nativeint
    = "caml_tolk_metal_blit_copy_bc" "caml_tolk_metal_blit_copy"

  external create_shared_event : nativeint -> nativeint
    = "caml_tolk_metal_create_shared_event"

  external release_shared_event : nativeint -> unit
    = "caml_tolk_metal_release_shared_event"

  external encode_signal_event : nativeint -> nativeint -> int -> unit
    = "caml_tolk_metal_encode_signal_event"

  external encode_wait_event : nativeint -> nativeint -> int -> unit
    = "caml_tolk_metal_encode_wait_event"

  external command_buffer_gpu_time : nativeint -> float * float
    = "caml_tolk_metal_command_buffer_gpu_time"

  external command_buffer_wait_time : nativeint -> float
    = "caml_tolk_metal_command_buffer_wait_time"

  external device_name : nativeint -> string = "caml_tolk_metal_device_name"
  external device_arch : nativeint -> string = "caml_tolk_metal_device_arch"
end

module Buffer_token = struct
  type buffer = {
    token : nativeint;
    handle : nativeint;
    size : int;
    offset : int;
  }

  let next = Atomic.make (-1)
  let mutex = Mutex.create ()
  let strong_table : (nativeint, buffer) Hashtbl.t = Hashtbl.create 1024
  let weak_table : (nativeint, buffer Weak.t) Hashtbl.t = Hashtbl.create 1024

  let unregister_token token =
    Mutex.lock mutex;
    Hashtbl.remove strong_table token;
    Hashtbl.remove weak_table token;
    Mutex.unlock mutex

  let register ?(strong = false) handle ~size ~offset =
    let token = Nativeint.of_int (Atomic.fetch_and_add next (-1)) in
    let buffer = { token; handle; size; offset } in
    Mutex.lock mutex;
    if strong then Hashtbl.add strong_table token buffer
    else begin
      let weak = Weak.create 1 in
      Weak.set weak 0 (Some buffer);
      Hashtbl.add weak_table token weak
    end;
    Mutex.unlock mutex;
    if not strong then Gc.finalise (fun b -> unregister_token b.token) buffer;
    buffer

  let unregister buffer = unregister_token buffer.token

  let resolve token =
    Mutex.lock mutex;
    let buffer =
      match Hashtbl.find_opt strong_table token with
      | Some buffer -> Some buffer
      | None -> (
          match Hashtbl.find_opt weak_table token with
          | None -> None
          | Some weak -> (
              match Weak.get weak 0 with
              | Some buffer -> Some buffer
              | None ->
                  Hashtbl.remove weak_table token;
                  None))
    in
    Mutex.unlock mutex;
    match buffer with
    | Some buffer -> buffer
    | None when Nativeint.compare token Nativeint.zero >= 0 ->
        { token; handle = token; size = 0; offset = 0 }
    | None ->
        invalid_arg
          (Printf.sprintf "unknown Metal buffer token %nd" token)

  let resolve_array tokens =
    let len = Array.length tokens in
    let handles = Array.make len Nativeint.zero in
    let offsets = Array.make len 0 in
    for i = 0 to len - 1 do
      let buffer = resolve tokens.(i) in
      handles.(i) <- buffer.handle;
      offsets.(i) <- buffer.offset
    done;
    (handles, offsets)
end

module State = struct
  type t = {
    device : nativeint;
    queue : nativeint;
    shared_event : nativeint;
    mutable timeline_value : int;
    mutable in_flight : nativeint list;
    mutable closed : bool;
    needs_icb_fix : bool;
    device_name : string;
    arch : Gpu_target.metal;
  }

  let create () =
    let device = Ffi.create_device () in
    try
      let queue = Ffi.create_command_queue device in
      try
        let shared_event = Ffi.create_shared_event device in
        try
          let needs_icb_fix = Ffi.needs_icb_fix device in
          let device_name = Ffi.device_name device in
          let arch =
            match Gpu_target.parse_metal_arch (Ffi.device_arch device) with
            | Some arch -> arch
            | None -> failwith "invalid Metal device architecture"
          in
          {
            device;
            queue;
            shared_event;
            timeline_value = 0;
            in_flight = [];
            closed = false;
            needs_icb_fix;
            device_name;
            arch;
          }
        with exn ->
          Ffi.release_shared_event shared_event;
          raise exn
      with exn ->
        Ffi.release_command_queue queue;
        raise exn
    with exn ->
      Ffi.release_device device;
      raise exn

  let synchronize t =
    let rec drain = function
      | [] -> ()
      | cmd :: rest ->
          t.in_flight <- rest;
          Ffi.command_buffer_wait cmd;
          drain rest
    in
    drain t.in_flight

  let shutdown t =
    if not t.closed then (
      synchronize t;
      Ffi.release_shared_event t.shared_event;
      Ffi.release_command_queue t.queue;
      Ffi.release_device t.device;
      t.closed <- true)

  let is_virtual t =
    let name = String.lowercase_ascii t.device_name in
    let rec has_substring s sub i =
      if i + String.length sub > String.length s then false
      else if String.sub s i (String.length sub) = sub then true
      else has_substring s sub (i + 1)
    in
    has_substring name "virtual" 0
end

module Allocator = struct
  let raw state =
    let alloc size spec =
      let handle =
        match spec.Device.Buffer_spec.external_ptr with
        | Some ptr -> ptr
        | None -> Ffi.buffer_alloc state.State.device size
      in
      Buffer_token.register ~strong:true handle ~size ~offset:0
    in
    let free buf _size spec =
      Buffer_token.unregister buf;
      match spec.Device.Buffer_spec.external_ptr with
      | Some _ -> ()
      | None -> Ffi.buffer_free buf.Buffer_token.handle
    in
    let copyin buf bytes =
      State.synchronize state;
      Ffi.buffer_copyin buf.Buffer_token.handle buf.offset bytes
    in
    let copyout bytes buf =
      State.synchronize state;
      Ffi.buffer_copyout bytes buf.Buffer_token.handle buf.offset
    in
    let transfer ~dest ~src nbytes =
      State.synchronize state;
      let cmd =
        Ffi.blit_copy state.State.queue src.Buffer_token.handle src.offset
          dest.Buffer_token.handle dest.offset nbytes
      in
      state.State.in_flight <- cmd :: state.State.in_flight;
      State.synchronize state
    in
    let addr buf = buf.Buffer_token.token in
    let offset buf size byte_offset =
      if byte_offset < 0 then
        invalid_arg "Metal buffer offset must be non-negative";
      if byte_offset + size > buf.Buffer_token.size then
        invalid_arg "Metal buffer view exceeds base buffer";
      Buffer_token.register buf.handle ~size
        ~offset:(buf.offset + byte_offset)
    in
    {
      Device.Allocator.alloc;
      free;
      copyin;
      copyout;
      addr;
      offset = Some offset;
      transfer = Some transfer;
      supports_transfer = true;
      copy_from_disk = None;
      supports_copy_from_disk = false;
    }

  let create state =
    Device.Allocator.Pack (Device.Lru_allocator.wrap (raw state))
end

module Compiler = struct
  let compile src =
    match Ffi.compile src with
    | Some binary -> binary
    | None -> Bytes.of_string src
    | exception Failure _ -> Bytes.of_string src

  let create () =
    Compiler.make ~name:"METAL" ~cachekey:"compile_metal_direct" ~compile ()
end

module Program = struct
  let runtime state entry_name lib ~runtimevars:_ =
    let handle = Ffi.program_create state.State.device entry_name lib in
    let local_dims = [| 1; 1; 1 |] in
    let call bufs ~global ~local ~vals ~wait ~timeout:_ =
      let local = Option.value local ~default:local_dims in
      let bufs, buf_offsets = Buffer_token.resolve_array bufs in
      let args = Array.map Int64.to_int vals in
      let cmd =
        Ffi.program_dispatch state.State.queue handle bufs buf_offsets
          args global local
      in
      if wait then Some (Ffi.command_buffer_wait_time cmd)
      else begin
        state.State.in_flight <- cmd :: state.State.in_flight;
        None
      end
    in
    let free () = Ffi.program_free handle in
    Device.{ call; free; handle }
end

module Icb = struct
  type t = { handle : nativeint; count : int }

  let create state ~count =
    let handle = Ffi.icb_create state.State.device count in
    { handle; count }

  let encode t ~index ~program ~buffers ~arg_buf ~arg_offsets ~global ~local =
    let buffers, buffer_offsets = Buffer_token.resolve_array buffers in
    Ffi.icb_encode t.handle index program buffers buffer_offsets arg_buf
      arg_offsets global local

  let update_buffer t ~index ~buf_index ~buf =
    let buffer = Buffer_token.resolve buf in
    Ffi.icb_update_buffer t.handle index buf_index buffer.handle buffer.offset

  let update_dispatch t ~index ~global ~local =
    Ffi.icb_update_dispatch t.handle index global local

  let execute state t ~resources ~pipelines =
    let resources, _offsets = Buffer_token.resolve_array resources in
    let fix_pipelines = if state.State.needs_icb_fix then pipelines else [||] in
    let cmd =
      Ffi.icb_execute state.State.queue t.handle t.count resources fix_pipelines
    in
    state.State.in_flight <- cmd :: state.State.in_flight

  let release t = Ffi.icb_release t.handle
end

let create name =
  let state = State.create () in
  at_exit (fun () -> State.shutdown state);
  let allocator = Allocator.create state in
  let renderer =
    Renderer.with_compiler (Compiler.create ()) (Cstyle.metal state.State.arch)
  in
  let renderer_set = Device.Renderer_set.make [renderer, None] in
  let runtime = Program.runtime state in
  let synchronize () = State.synchronize state in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ()
