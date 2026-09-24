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

  external buffer_contents :
    nativeint -> int -> int -> Device.Allocator.host_view
    = "caml_tolk_metal_buffer_contents"

  external program_create : nativeint -> string -> bytes -> int -> int array -> nativeint
    = "caml_tolk_metal_program_create"

  external program_free : nativeint -> unit = "caml_tolk_metal_program_free"

  external program_dispatch :
    nativeint ->
    nativeint ->
    nativeint array ->
    int array ->
    int64 array ->
    int array ->
    int array ->
    nativeint
    = "caml_tolk_metal_program_dispatch_bc" "caml_tolk_metal_program_dispatch"

  external program_args_size : nativeint -> int = "caml_tolk_metal_program_args_size"

  external program_write_args :
    nativeint -> nativeint -> int -> nativeint array -> int array -> int64 array -> unit
    = "caml_tolk_metal_program_write_args_bc" "caml_tolk_metal_program_write_args"

  external program_set_buffer :
    nativeint -> nativeint -> int -> int -> nativeint -> int -> unit
    = "caml_tolk_metal_program_set_buffer_bc" "caml_tolk_metal_program_set_buffer"

  external program_set_value : nativeint -> nativeint -> int -> int -> int64 -> unit
    = "caml_tolk_metal_program_set_value"

  external command_buffer_wait : nativeint -> unit
    = "caml_tolk_metal_command_buffer_wait"

  external compile : string -> bytes option = "caml_tolk_metal_compile"

  external icb_create : nativeint -> int -> nativeint
    = "caml_tolk_metal_icb_create"

  external icb_encode :
    nativeint -> int -> nativeint -> nativeint -> int -> int array -> int array -> unit
    = "caml_tolk_metal_icb_encode_bc" "caml_tolk_metal_icb_encode"

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

module Metal_buffer = struct
  type t = { handle : nativeint; size : int; offset : int }
  let kind : t Type.Id.t = Type.Id.make ()
  let get buf =
    Option.value (Device.Buffer.get kind buf)
      ~default:{ handle = 0n; size = 0; offset = 0 }
  let resolve_array buffers =
    let len = Array.length buffers in
    let handles = Array.make len Nativeint.zero in
    let offsets = Array.make len 0 in
    for i = 0 to len - 1 do
      let buffer = get buffers.(i) in
      handles.(i) <- buffer.handle;
      offsets.(i) <- buffer.offset
    done;
    handles, offsets
end

module State = struct
  type t = {
    device : nativeint;
    queue : nativeint;
    shared_event : nativeint;
    mutable timeline_value : int;
    mutable in_flight : nativeint list;
    mutable synchronizations : int;
    mutable closed : bool;
    needs_icb_fix : bool;
    device_name : string;
    arch : string;
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
          let arch = Ffi.device_arch device in
          {
            device;
            queue;
            shared_event;
            timeline_value = 0;
            in_flight = [];
            synchronizations = 0;
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
    t.synchronizations <- t.synchronizations + 1;
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
      Metal_buffer.{handle; size; offset = 0}
    in
    let free buf _size spec =
      match spec.Device.Buffer_spec.external_ptr with
      | Some _ -> ()
      | None -> Ffi.buffer_free buf.Metal_buffer.handle
    in
    let copyin buf bytes =
      State.synchronize state;
      Ffi.buffer_copyin buf.Metal_buffer.handle buf.offset bytes
    in
    let copyout bytes buf =
      State.synchronize state;
      Ffi.buffer_copyout bytes buf.Metal_buffer.handle buf.offset
    in
    (* tinygrad's [_as_buffer]: the shared buffer's contents, in place. *)
    let as_buffer buf nbytes =
      Ffi.buffer_contents buf.Metal_buffer.handle buf.offset nbytes
    in
    let transfer ~dest ~src ~dest_device ~src_device nbytes =
      if Device.canonicalize dest_device <> Device.canonicalize src_device then false
      else begin
        State.synchronize state;
        let cmd =
          Ffi.blit_copy state.State.queue src.Metal_buffer.handle src.offset
            dest.Metal_buffer.handle dest.offset nbytes
        in
        state.State.in_flight <- cmd :: state.State.in_flight;
        State.synchronize state;
        true
      end
    in
    let offset buf size byte_offset =
      if byte_offset < 0 then
        invalid_arg "Metal buffer offset must be non-negative";
      if byte_offset + size > buf.Metal_buffer.size then
        invalid_arg "Metal buffer view exceeds base buffer";
      Metal_buffer.{handle = buf.handle; size; offset = buf.offset + byte_offset}
    in
    {
      Device.Allocator.kind = Metal_buffer.kind;
      alloc;
      free;
      copyin;
      copyout;
      as_buffer = Some as_buffer;
      addr = None;
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
  let runtime state (obj : Tolk_uop.Tiny_elf.t) =
    let entry_name = obj.name and lib = obj.lib in
    let fields = Tolk_uop.Tiny_elf.layout obj.signature in
    let layout = Array.of_list (List.concat_map (fun (field : Tolk_uop.Tiny_elf.field) ->
        [ field.argument.slot; field.offset; field.size ]) fields) in
    let nbufs = List.fold_left (fun n (arg : Tolk_uop.Tiny_elf.argument) ->
        if arg.addrspace = Tolk_uop.Dtype.Alu then n else n + 1) 0 obj.signature in
    let handle = Ffi.program_create state.State.device entry_name lib nbufs layout in
    let local_dims = [| 1; 1; 1 |] in
    let call bufs ~global ~local ~vals ~wait ~timeout:_ =
      let local = Option.value local ~default:local_dims in
      let bufs, buf_offsets = Metal_buffer.resolve_array bufs in
      let cmd =
        Ffi.program_dispatch state.State.queue handle bufs buf_offsets
          vals global local
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

  let encode t ~index ~program ~arg_buf ~arg_offset ~global ~local =
    Ffi.icb_encode t.handle index program arg_buf arg_offset global local

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

module Graph = struct
  (* Batched replay through an indirect command buffer: every kernel launch is
     encoded once as an indirect compute command, and a replay submits the
     whole sequence in a single command buffer. Commands are separated by
     barriers, so they run in recording order and node dependencies need no
     encoding. Each command binds one argument structure in a shared arena;
     signature slots locate its full GPU addresses and typed scalar values. *)
  let build state (nodes : Device.Graph.node array) =
    let count = Array.length nodes in
    let kernels =
      Array.map
        (function
          | Device.Graph.Kernel { handle; global; local; bufs; vals; _ } ->
              (handle, global, local, bufs, vals)
          | Device.Graph.Copy _ ->
              invalid_arg "Metal graph: unsupported COPY node")
        nodes
    in
    let arg_offsets = Array.make (count + 1) 0 in
    Array.iteri
      (fun j (program, _, _, _, _) ->
        let size = Ffi.program_args_size program in
        arg_offsets.(j + 1) <- arg_offsets.(j) + ((size + 255) / 256 * 256))
      kernels;
    let arg_buf = Ffi.buffer_alloc state.State.device (max 8 arg_offsets.(count)) in
    let program j = let handle, _, _, _, _ = kernels.(j) in handle in
    let write_val j i v =
      Ffi.program_set_value (program j) arg_buf arg_offsets.(j) i (Int64.of_int v)
    in
    let icb =
      try Icb.create state ~count
      with exn -> Ffi.buffer_free arg_buf; raise exn in
    (* Referenced buffer resources are retained across replay and refreshed
       when an argument's GPU address is patched. *)
    let bound =
      try Array.mapi
        (fun j (program, global, local, buffers, vals) ->
          let buffers, offsets = Metal_buffer.resolve_array buffers in
          Ffi.program_write_args program arg_buf arg_offsets.(j) buffers offsets
            (Array.map Int64.of_int vals);
          Icb.encode icb ~index:j ~program ~arg_buf ~arg_offset:arg_offsets.(j)
            ~global ~local;
          buffers)
        kernels
      with exn ->
        Icb.release icb;
        Ffi.buffer_free arg_buf;
        raise exn
    in
    let dedup handles =
      let seen = Hashtbl.create 64 in
      List.filter
        (fun h ->
          h <> Nativeint.zero && not (Hashtbl.mem seen h)
          && (Hashtbl.replace seen h ();
              true))
        handles
      |> Array.of_list
    in
    let fix_icb =
      Helpers.getenv "FIX_METAL_ICB" (Bool.to_int state.State.needs_icb_fix)
      <> 0
    in
    let pipelines =
      if fix_icb then
        dedup
          (Array.to_list
             (Array.map (fun (program, _, _, _, _) -> program) kernels))
      else [||]
    in
    let resources () =
      dedup
        (arg_buf :: List.concat_map Array.to_list (Array.to_list bound))
    in
    let all_resources = ref (resources ()) in
    let rebound = ref false in
    let last = ref None in
    (* The recorded commands and argument arena are read by the GPU until
       the previous replay completes, so it is awaited before either is
       patched or resubmitted. A synchronize since that replay has already
       awaited and released its command buffer, whose address a later command
       buffer may reuse. *)
    let settle () =
      match !last with
      | Some (cmd, at)
        when at = state.State.synchronizations
             && List.mem cmd state.State.in_flight ->
          state.State.in_flight <-
            List.filter (fun c -> c <> cmd) state.State.in_flight;
          last := None;
          Ffi.command_buffer_wait cmd
      | _ -> last := None
    in
    let launch ~wait =
      settle ();
      if !rebound then begin
        all_resources := resources ();
        rebound := false
      end;
      let cmd =
        Ffi.icb_execute state.State.queue icb.Icb.handle count !all_resources
          pipelines
      in
      if wait then Some (Ffi.command_buffer_wait_time cmd)
      else begin
        state.State.in_flight <- cmd :: state.State.in_flight;
        last := Some (cmd, state.State.synchronizations);
        None
      end
    in
    let exec =
      {
        Device.Graph.set_buf =
          (fun node pos buf ->
            settle ();
            let buffer = Metal_buffer.get buf in
            Ffi.program_set_buffer (program node) arg_buf arg_offsets.(node) pos
              buffer.handle buffer.offset;
            bound.(node).(pos) <- buffer.handle;
            rebound := true);
        set_val =
          (fun node idx v ->
            settle ();
            write_val node idx v);
        set_launch_dims =
          (fun node ~global ~local ->
            settle ();
            Icb.update_dispatch icb ~index:node ~global ~local);
        set_params = (fun _ -> ());
        launch;
      }
    in
    (* A finaliser can run inside any allocation, including one that [settle]
       or [launch] makes while updating the in-flight list, so it leaves that
       list alone. It need not await the last replay: command buffers retain
       what they reference, so a replay in flight holds the ICB and the
       resources it declared. *)
    Gc.finalise
      (fun (_ : Device.Graph.exec) ->
        if not state.State.closed then begin
          Icb.release icb;
          Ffi.buffer_free arg_buf
        end)
      exec;
    exec

  let create state =
    {
      Device.Graph.supports_copy = false;
      max_buffer_offset = None;
      build = build state;
    }
end

let create name =
  let state = State.create () in
  at_exit (fun () -> State.shutdown state);
  let allocator = Allocator.create state in
  let renderer_set = Device.Renderer_set.make ~device:name
      ~arch:state.State.arch
      [ "METAL", (fun target ->
          let arch = match Gpu_target.parse_metal_arch target.Tolk_uop.Target.arch with
            | Some arch -> arch
            | None -> invalid_arg ("unsupported Metal architecture: " ^ target.arch) in
          Renderer.with_compiler (Compiler.create ()) (Cstyle.metal arch)) ] in
  let runtime = Program.runtime state in
  let synchronize () = State.synchronize state in
  let graph =
    if State.is_virtual state then None else Some (Graph.create state)
  in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ?graph ()
