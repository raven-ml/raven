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

  external buffer_copyin : nativeint -> bytes -> unit
    = "caml_tolk_metal_buffer_copyin"

  external buffer_copyout : bytes -> nativeint -> unit
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
    nativeint ->
    int array ->
    int array ->
    int array ->
    unit = "caml_tolk_metal_icb_encode_bc" "caml_tolk_metal_icb_encode"

  external icb_update_buffer : nativeint -> int -> int -> nativeint -> unit
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

  external device_name : nativeint -> string = "caml_tolk_metal_device_name"
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
  }

  let create () =
    let device = Ffi.create_device () in
    let queue = Ffi.create_command_queue device in
    let shared_event = Ffi.create_shared_event device in
    let needs_icb_fix = Ffi.needs_icb_fix device in
    let device_name = Ffi.device_name device in
    {
      device;
      queue;
      shared_event;
      timeline_value = 0;
      in_flight = [];
      closed = false;
      needs_icb_fix;
      device_name;
    }

  let synchronize t =
    List.iter Ffi.command_buffer_wait t.in_flight;
    t.in_flight <- []

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
      match spec.Device.Buffer_spec.external_ptr with
      | Some ptr -> ptr
      | None -> Ffi.buffer_alloc state.State.device size
    in
    let free buf _size spec =
      match spec.Device.Buffer_spec.external_ptr with
      | Some _ -> ()
      | None -> Ffi.buffer_free buf
    in
    let copyin buf bytes =
      State.synchronize state;
      Ffi.buffer_copyin buf bytes
    in
    let copyout bytes buf =
      State.synchronize state;
      Ffi.buffer_copyout bytes buf
    in
    let transfer ~dest ~src nbytes =
      State.synchronize state;
      let cmd = Ffi.blit_copy state.State.queue src 0 dest 0 nbytes in
      state.State.in_flight <- cmd :: state.State.in_flight;
      State.synchronize state
    in
    let addr buf = buf in
    {
      Device.Allocator.alloc;
      free;
      copyin;
      copyout;
      addr;
      offset = Some (fun buf _size _offset -> buf);
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

  let create () = Compiler.make ~name:"METAL" ~cachekey:"compile_metal" ~compile ()
end

module Program = struct
  let runtime state entry_name lib ~runtimevars:_ =
    let handle = Ffi.program_create state.State.device entry_name lib in
    let local_dims = [| 1; 1; 1 |] in
    let call bufs ~global ~local ~vals:_ ~wait ~timeout:_ =
      let local = Option.value local ~default:local_dims in
      let buf_offsets = Array.make (Array.length bufs) 0 in
      let cmd =
        Ffi.program_dispatch state.State.queue handle bufs buf_offsets
          [||] global local
      in
      state.State.in_flight <- cmd :: state.State.in_flight;
      if wait then begin
        State.synchronize state;
        None
      end else
        None
    in
    let free () = Ffi.program_free handle in
    Device.{ call; free }
end

module Icb = struct
  type t = { handle : nativeint; count : int }

  let create state ~count =
    let handle = Ffi.icb_create state.State.device count in
    { handle; count }

  let encode t ~index ~program ~buffers ~arg_buf ~arg_offsets ~global ~local =
    Ffi.icb_encode t.handle index program buffers arg_buf arg_offsets global
      local

  let update_buffer t ~index ~buf_index ~buf =
    Ffi.icb_update_buffer t.handle index buf_index buf

  let update_dispatch t ~index ~global ~local =
    Ffi.icb_update_dispatch t.handle index global local

  let execute state t ~resources ~pipelines =
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
    Renderer.with_compiler (Compiler.create ()) Cstyle.metal
  in
  let renderer_set = Device.Renderer_set.make [renderer, None] in
  let runtime = Program.runtime state in
  let synchronize () = State.synchronize state in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ()
