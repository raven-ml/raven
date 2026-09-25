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

  external buffer_contents : nativeint -> nativeint = "caml_tolk_metal_buffer_contents"

  external buffer_free : nativeint -> unit = "caml_tolk_metal_buffer_free"

  external buffer_copyin : nativeint -> int -> bytes -> unit
    = "caml_tolk_metal_buffer_copyin"

  external buffer_copyout : bytes -> nativeint -> int -> unit
    = "caml_tolk_metal_buffer_copyout"

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
  external hcq_create : nativeint -> nativeint -> nativeint = "caml_tolk_metal_hcq_create"
  external hcq_release : nativeint -> unit = "caml_tolk_metal_hcq_release"
  external hcq_resource : nativeint -> nativeint -> bool -> unit = "caml_tolk_metal_hcq_resource"
  external hcq_wait : nativeint -> int64 -> unit = "caml_tolk_metal_hcq_wait"
  external hcq_symbol : string -> nativeint = "caml_tolk_metal_hcq_symbol"
  external buffer_address : nativeint -> nativeint = "caml_tolk_metal_buffer_address"

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
    context : nativeint;
    mutable timeline : Device.Buffer.t option;
    mutable context_buffer : Device.Buffer.t option;
    mutable in_flight : nativeint list;
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
            context = Ffi.hcq_create queue shared_event;
            timeline = None;
            context_buffer = None;
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
    if not t.closed then begin
      drain t.in_flight;
      Option.iter (fun timeline ->
          let value = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 8 in
          Ffi.hcq_wait t.context value) t.timeline
    end

  let shutdown t =
    if not t.closed then (
      synchronize t;
      Ffi.hcq_release t.context;
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
      (try Ffi.hcq_resource state.State.context handle true
       with exn ->
         if spec.Device.Buffer_spec.external_ptr = None then Ffi.buffer_free handle;
         raise exn);
      Metal_buffer.{handle; size; offset = 0}
    in
    let free buf _size spec =
      State.synchronize state;
      if not state.State.closed then
        Ffi.hcq_resource state.State.context buf.Metal_buffer.handle false;
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
      host = (fun buf -> Some (Nativeint.add (Ffi.buffer_contents buf.Metal_buffer.handle)
          (Nativeint.of_int buf.offset)));
      mapping = None;
      synchronize = (fun () -> State.synchronize state);
      alloc;
      free;
      copyin;
      copyout;
      addr = Some (fun buf -> Nativeint.add (Ffi.buffer_address buf.Metal_buffer.handle)
          (Nativeint.of_int buf.offset));
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

module Queue = struct
  open Tolk_uop
  module U = Uop
  module B = Device.Buffer

  type command = { object_ : Tiny_elf.t; global : int array; local : int array; offset : int }
  type descriptor = { commands : command list; header : int }

  let host_buffer ?(bytes = Bytes.empty) size =
    let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
    let b = B.create ~device:"CPU" ~size ~dtype:Dtype.uint64 allocator in
    B.ensure_allocated b;
    B.copyin b (if Bytes.length bytes = 0 then Bytes.make (size * 8) '\000' else bytes);
    b

  let word value =
    let bytes = Bytes.create 8 in
    Bytes.set_int64_le bytes 0 (Int64.of_nativeint value);
    host_buffer ~bytes 1

  let context name = U.placeholder ~shape:[1] ~dtype:Dtype.uint64 ~slot:0
      ~device:(U.Single name) ~allocation:("metal_context", "") ()
  let index p i = U.index ~ptr:p ~idxs:[U.const_int i] ()
  let load p i = U.load ~src:(index p i) ()
  let call name ?after fn dtype args = Hcq2.ccall ~host:name ?after ~name:fn ~dtype args

  let bufferize state name u = match U.as_param u with
    | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
        let libs, symbol = (Marshal.from_string data 0 : string list * string) in
        if libs <> [] then invalid_arg "Metal host helpers do not load libraries";
        Some (word (Ffi.hcq_symbol symbol))
    | Some {param = {allocation = Some ("metal_context", _); _}; _} ->
        let buffer = match state.State.context_buffer with
          | Some b -> b | None -> let b = word state.context in state.context_buffer <- Some b; b in
        Some buffer
    | Some {param = {allocation = Some ("metal_icb", data); _}; _} ->
        let desc = (Marshal.from_string data 0 : descriptor) in
        let allocator = Allocator.raw state in
        let live = ref None in
        let alloc size spec =
          let raw = allocator.alloc size spec in
          let programs = ref [] and icb = ref None in
          (try
             Ffi.buffer_copyin raw.Metal_buffer.handle 0 (Bytes.make size '\000');
             let commands = Array.of_list desc.commands in
             let indirect = Icb.create state ~count:(Array.length commands) in
             icb := Some indirect;
             Array.iteri (fun i c ->
                 let program = Program.runtime state c.object_ in
                 programs := program :: !programs;
                 Icb.encode indirect ~index:i ~program:program.Device.handle
                   ~arg_buf:raw.handle ~arg_offset:c.offset ~global:c.global ~local:c.local) commands;
             let programs = List.rev !programs in
             let values = [indirect.handle; Nativeint.of_int indirect.count; 0n;
               Nativeint.of_int (Helpers.getenv "FIX_METAL_ICB" (Bool.to_int state.State.needs_icb_fix));
               Nativeint.of_int (List.length programs)] @ List.map (fun p -> p.Device.handle) programs in
             let bytes = Bytes.create (8 * List.length values) in
             List.iteri (fun i v -> Bytes.set_int64_le bytes (8 * i) (Int64.of_nativeint v)) values;
             Ffi.buffer_copyin raw.handle desc.header bytes;
             live := Some (indirect, programs);
             raw
           with exn ->
             Option.iter Icb.release !icb;
             List.iter (fun p -> p.Device.free ()) !programs;
             allocator.free raw size spec;
             raise exn) in
        let free raw size spec =
          State.synchronize state;
          let bytes = Bytes.create 8 in
          Ffi.buffer_copyout bytes raw.Metal_buffer.handle (desc.header + 16);
          let command = Int64.to_nativeint (Bytes.get_int64_le bytes 0) in
          if command <> 0n then Ffi.command_buffer_wait command;
          Option.iter (fun (icb, programs) -> Icb.release icb;
              List.iter (fun p -> p.Device.free ()) programs) !live;
          live := None;
          allocator.free raw size spec in
        let spec = {Device.Buffer_spec.default with nolru = true; cpu_access = true} in
        Some (B.create ~device:name ~size:(U.max_numel u) ~dtype:(U.dtype u) ~spec
          (Device.Allocator.Pack {allocator with alloc; free}))
    | Some _ when U.node_tag u = Some "timeline" ->
        let buffer = match state.State.timeline with
          | Some b -> b | None -> let b = host_buffer 2 in state.timeline <- Some b; b in
        Some buffer
    | _ -> None

  let lower name u = match U.as_load u with
    | Some {src; _} ->
        (match U.as_index src with
         | Some {ptr; idxs = [i]} when U.const_int_value i = Some 0
             && U.node_tag (U.buf_uop ptr) = Some "timeline" ->
             let deps = if U.op ptr = Ops.After then List.tl (U.children ptr) else [] in
             Some (call name ~after:deps "tolk_metal_hcq_poll" Dtype.uint64 [load (context name) 0])
         | _ -> None)
    | _ -> None

  let encode name u = match U.op u, U.arg u, U.children u with
    | Ops.Custom_function, U.Arg.String "submit_metal_compute", [linear; dependency] ->
        let commands = ref [] and rows = ref [] and sizes = ref [] and used = ref 0 in
        let signal = ref None in
        let align n a = (n + a - 1) / a * a in
        List.iter (fun node -> match U.as_call node, U.arg node with
            | Some {body; args}, _ when U.op body = Ops.Program ->
                let info = Option.get (U.as_program_info body) in
                let buffers = List.filter (fun a -> not (U.is_bound_var a)) args in
                let bound = List.filter_map (fun a -> match U.as_bind a with
                    | Some {var; value} -> Option.map (fun n -> n, value) (U.program_var_name var)
                    | None -> None) args in
                let variables = List.map (fun v -> match U.program_var_name v with
                    | Some n -> Option.value (List.assoc_opt n bound) ~default:v | None -> v) info.vars in
                let actuals = List.map (fun i -> U.getaddr ~device:name ~src:(List.nth buffers i) ()) info.globals @ variables in
                let object_ = U.to_elf body in
                let fields = Tiny_elf.layout object_.signature in
                let offset = align !used 256 in
                let end_ = ref (offset + 8) in
                List.iter (fun (field : Tiny_elf.field) ->
                    let value = List.nth actuals field.argument.slot in
                    let dtype = if field.argument.addrspace = Dtype.Alu then field.argument.dtype else Dtype.uint64 in
                    rows := (offset + field.offset, U.cast ~src:value ~dtype) :: !rows;
                    end_ := max !end_ (offset + field.offset + field.size)) fields;
                used := !end_;
                let dims = info.global_size @ info.local_size in
                let initial = List.map (function U.Launch_int n -> n | U.Launch_float f -> int_of_float f | U.Launch_sym _ -> 1) dims in
                let pad xs = Array.init 3 (fun i -> if i < List.length xs then List.nth xs i else 1) in
                let global = pad (List.filteri (fun i _ -> i < List.length info.global_size) initial) in
                let local = pad (List.filteri (fun i _ -> i >= List.length info.global_size) initial) in
                if List.exists (function U.Launch_sym _ -> true | _ -> false) dims then begin
                  let at = align !used 8 in
                  let values ds = List.init 3 (fun i ->
                      if i >= List.length ds then U.const (Const.int Dtype.uint64 1) else
                      match List.nth ds i with
                      | U.Launch_int n -> U.const (Const.int Dtype.uint64 n)
                      | U.Launch_float f -> U.const (Const.int Dtype.uint64 (int_of_float f))
                      | U.Launch_sym v -> U.cast ~src:v ~dtype:Dtype.uint64) in
                  List.iteri (fun i v -> rows := (at + 8 * i, v) :: !rows)
                    (values info.global_size @ values info.local_size);
                  sizes := (List.length !commands, at) :: !sizes;
                  used := at + 48
                end;
                commands := !commands @ [{object_; global; local; offset}]
            | _, U.Arg.Typed ("store", _) -> signal := Some (U.src node).(1)
            | _, U.Arg.Typed (("barrier" | "wait"), _) -> ()
            | _ -> invalid_arg "Metal queue: unsupported instruction") (U.children linear);
        let header = align !used 8 in
        let size = header + 8 * (5 + List.length !commands) in
        let desc = {commands = !commands; header} in
        let buffer = U.placeholder ~shape:[size] ~dtype:Dtype.uint8 ~slot:0
            ~device:(U.Single name) ~volatile:true
            ~allocation:("metal_icb", Marshal.to_string desc []) () in
        let patched = Hcq2.patch ~after:[dependency] buffer (List.rev !rows) in
        let header_ptr = index patched header in
        let header_words = U.bitcast ~src:(U.shrink ~src:patched ~offset:(U.const_int header)
            ~size:(U.const_int (size - header))) ~dtype:Dtype.uint64 in
        let previous = ref [dependency; patched] in
        List.iter (fun (command, at) ->
            previous := [call name ~after:!previous "tolk_metal_hcq_update" Dtype.void
              [load header_words 0; U.const (Const.int Dtype.uint64 command); index patched at]]) (List.rev !sizes);
        Some (call name ~after:!previous "tolk_metal_hcq_submit" Dtype.void
          [load (context name) 0; header_ptr; Option.get !signal])
    | _ -> None

  let create state device_name =
    let host = try Device.get "CPU" with Failure _ -> Tolk_cpu.create "CPU" in
    Device.{prepare = (fun () -> ()); host = Device.name host; copy = (fun _ -> false); encode = encode device_name; lower = lower device_name;
      compile = Codegen.to_program ~optimize:false host (Device.renderer host)}, bufferize state device_name
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
  let queue, bufferize = Queue.create state name in
  let queue = if State.is_virtual state then None else Some queue in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ?queue ~bufferize ()
