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

  external buffer_wrap : nativeint -> nativeint -> int -> nativeint
    = "caml_tolk_metal_buffer_wrap"

  external has_unified_memory : nativeint -> bool
    = "caml_tolk_metal_has_unified_memory"

  external buffer_contents : nativeint -> nativeint = "caml_tolk_metal_buffer_contents"

  external buffer_free : nativeint -> unit = "caml_tolk_metal_buffer_free"

  external buffer_copyin : nativeint -> int -> bytes -> unit
    = "caml_tolk_metal_buffer_copyin"

  external program_create : nativeint -> string -> string -> nativeint
    = "caml_tolk_metal_program_create"

  external program_free : nativeint -> unit = "caml_tolk_metal_program_free"

  external compile : string -> bytes option = "caml_tolk_metal_compile"

  external icb_create : nativeint -> int -> nativeint
    = "caml_tolk_metal_icb_create"

  external icb_encode :
    nativeint -> int -> nativeint -> nativeint -> int -> int array -> int array -> int -> unit
    = "caml_tolk_metal_icb_encode_bc" "caml_tolk_metal_icb_encode"

  external icb_release : nativeint -> unit = "caml_tolk_metal_icb_release"
  external needs_icb_fix : nativeint -> bool = "caml_tolk_metal_needs_icb_fix"

  external device_arch : nativeint -> string = "caml_tolk_metal_device_arch"
  external hcq_create : nativeint -> nativeint = "caml_tolk_metal_hcq_create"
  external hcq_release : nativeint -> unit = "caml_tolk_metal_hcq_release"
  external hcq_resource : nativeint -> nativeint -> bool -> unit = "caml_tolk_metal_hcq_resource"
  external hcq_wait : nativeint -> int64 -> unit = "caml_tolk_metal_hcq_wait"
  external hcq_symbol : string -> nativeint = "caml_tolk_metal_hcq_symbol"
  external profile_clock : unit -> float = "caml_tolk_metal_profile_clock"
  external buffer_address : nativeint -> nativeint = "caml_tolk_metal_buffer_address"

end

module Metal_buffer = struct
  type t = { handle : nativeint; size : int; offset : int }
  let kind : t Type.Id.t = Type.Id.make ()
end

module State = struct
  type t = {
    device : nativeint;
    queue : nativeint;
    context : nativeint;
    mutable timeline : Device.Buffer.t option;
    mutable context_buffer : Device.Buffer.t option;
    mutable closed : bool;
    programs : (string * string, nativeint) Hashtbl.t;
    program_lock : Mutex.t;
    needs_icb_fix : bool;
    arch : string;
  }

  let create () =
    let device = Ffi.create_device () in
    try
      let queue = Ffi.create_command_queue device in
      try
        let needs_icb_fix = Ffi.needs_icb_fix device in
        let arch = Ffi.device_arch device in
        {
          device;
          queue;
          context = Ffi.hcq_create queue;
          timeline = None;
          context_buffer = None;
          closed = false;
          programs = Hashtbl.create 16;
          program_lock = Mutex.create ();
          needs_icb_fix;
          arch;
        }
      with exn ->
        Ffi.release_command_queue queue;
        raise exn
    with exn ->
      Ffi.release_device device;
      raise exn

  let with_program_lock t f =
    Tolk_uop.Storage.with_operation (fun () -> Mutex.protect t.program_lock f)

  let timeline t = with_program_lock t (fun () -> t.timeline)

  let synchronize t =
    if not t.closed then begin
      Option.iter (fun timeline ->
          let value = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 8 in
          Ffi.hcq_wait t.context value) (timeline t)
    end

  let shutdown t =
    if not t.closed then (
      synchronize t;
      Mutex.protect t.program_lock (fun () ->
          if not t.closed then begin
            Ffi.hcq_release t.context;
            t.closed <- true;
            Hashtbl.iter (fun _ program -> Ffi.program_free program) t.programs;
            Hashtbl.clear t.programs;
            Ffi.release_command_queue t.queue;
            Ffi.release_device t.device
          end))
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
    let offset buf size byte_offset =
      if byte_offset < 0 then
        invalid_arg "Metal buffer offset must be non-negative";
      if byte_offset + size > buf.Metal_buffer.size then
        invalid_arg "Metal buffer view exceeds base buffer";
      Metal_buffer.{handle = buf.handle; size; offset = buf.offset + byte_offset}
    in
    (* The tinygrad counterpart has no mapping: it copies host storage into a
       Metal buffer. Where the GPU shares the host's memory, the pages that
       hold host storage are wrapped in place instead, so a mapped file is not
       held twice. Two wraps may share a page. *)
    let map source =
      let unavailable () =
        raise (Tolk_uop.Storage.Mapping_unavailable
            "Metal maps host memory on unified-memory devices only")
      in
      match Device.Buffer.host_addr source with
      | None -> unavailable ()
      | Some address ->
          let size = Device.Buffer.nbytes source in
          let handle = Ffi.buffer_wrap state.State.device address size in
          if handle = 0n then unavailable ();
          (try Ffi.hcq_resource state.State.context handle true
           with exn -> Ffi.buffer_free handle; raise exn);
          let offset =
            Nativeint.to_int (Nativeint.sub address (Ffi.buffer_contents handle))
          in
          Metal_buffer.{handle; size; offset}
    in
    let unmap buf =
      State.synchronize state;
      if not state.State.closed then
        Ffi.hcq_resource state.State.context buf.Metal_buffer.handle false;
      Ffi.buffer_free buf.Metal_buffer.handle
    in
    {
      Device.Allocator.kind = Metal_buffer.kind;
      host = (fun buf -> Some (Nativeint.add (Ffi.buffer_contents buf.Metal_buffer.handle)
          (Nativeint.of_int buf.offset)));
      mapping = Some {map; unmap};
      synchronize = (fun () -> State.synchronize state);
      alloc;
      free;
      addr = Some (fun buf -> Nativeint.add (Ffi.buffer_address buf.Metal_buffer.handle)
          (Nativeint.of_int buf.offset));
      offset = Some offset;
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
  let key (obj : Tolk_uop.Tiny_elf.t) = Bytes.to_string obj.lib, obj.name

  let load state obj =
    let key = key obj in
    Mutex.protect state.State.program_lock (fun () ->
        if state.State.closed then invalid_arg "Metal device is closed";
        match Hashtbl.find_opt state.State.programs key with
        | Some program -> program
        | None ->
            let program = Ffi.program_create state.State.device obj.name (fst key) in
            (try Hashtbl.add state.State.programs key program
             with exn -> Ffi.program_free program; raise exn);
            program)

  let args_size (obj : Tolk_uop.Tiny_elf.t) =
    List.fold_left (fun size (field : Tolk_uop.Tiny_elf.field) ->
        max size ((field.offset + field.size + 7) / 8 * 8)) 8
        (Tolk_uop.Tiny_elf.layout obj.signature)
end

module Icb = struct
  type t = { handle : nativeint; count : int }

  let create state ~count =
    let handle = Ffi.icb_create state.State.device count in
    { handle; count }

  let encode t ~index ~program ~arg_buf ~arg_offset ~global ~local ~args_size =
    Ffi.icb_encode t.handle index program arg_buf arg_offset global local args_size

  let release t = Ffi.icb_release t.handle
end

module Queue = struct
  open Tolk_uop
  module U = Uop
  module B = Device.Buffer

  type command = { object_ : Tiny_elf.t; global : int array; local : int array;
    offset : int; sizes : int }
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
        Some (State.with_program_lock state (fun () ->
            if state.State.closed then invalid_arg "Metal device is closed";
            match state.State.context_buffer with
            | Some buffer -> buffer
            | None ->
                let buffer = word state.context in
                state.context_buffer <- Some buffer;
                buffer))
    | Some {param = {allocation = Some ("metal_icb", data); _}; _} ->
        let desc = (Marshal.from_string data 0 : descriptor) in
        let allocator = Allocator.raw state in
        let live = ref None in
        let alloc size spec =
          let raw = allocator.alloc size spec in
          let icb = ref None in
          (try
             Ffi.buffer_copyin raw.Metal_buffer.handle 0 (Bytes.make size '\000');
             let commands = Array.of_list desc.commands in
             let indirect = Icb.create state ~count:(Array.length commands) in
             icb := Some indirect;
             let programs = Array.map (fun c -> Program.load state c.object_) commands in
             Array.iteri (fun i c ->
                 Icb.encode indirect ~index:i ~program:programs.(i)
                   ~arg_buf:raw.handle ~arg_offset:c.offset ~global:c.global ~local:c.local
                   ~args_size:(Program.args_size c.object_)) commands;
             Array.iter (fun c ->
                 let dimensions = Array.append c.global c.local in
                 let bytes = Bytes.create 48 in
                 Array.iteri (fun i n -> Bytes.set_int64_le bytes (8 * i) (Int64.of_int n)) dimensions;
                 Ffi.buffer_copyin raw.handle c.sizes bytes) commands;
             let records = List.map2 (fun program c ->
                 [program; Nativeint.of_int (List.length c.object_.signature);
                  Nativeint.of_int c.offset; Nativeint.of_int c.sizes])
                 (Array.to_list programs) desc.commands |> List.concat in
             let programs = Helpers.dedup_by Nativeint.equal (Array.to_list programs) in
             (* Header: ICB, count, workaround, pipelines, argument buffer;
                pipeline handles; program/count/argument offset/size offset per command. *)
             let values = [indirect.handle; Nativeint.of_int indirect.count;
               Nativeint.of_int (Helpers.getenv "FIX_METAL_ICB" (Bool.to_int state.State.needs_icb_fix));
               Nativeint.of_int (List.length programs); raw.handle] @ programs @ records in
             let bytes = Bytes.create (8 * List.length values) in
             List.iteri (fun i v -> Bytes.set_int64_le bytes (8 * i) (Int64.of_nativeint v)) values;
             Ffi.buffer_copyin raw.handle desc.header bytes;
             live := Some indirect;
             raw
           with exn ->
             Option.iter Icb.release !icb;
             allocator.free raw size spec;
             raise exn) in
        let free raw size spec =
          State.synchronize state;
          Option.iter Icb.release !live;
          live := None;
          allocator.free raw size spec in
        let spec = {Device.Buffer_spec.default with nolru = true; cpu_access = true} in
        Some (B.create ~device:name ~size:(U.max_numel u) ~dtype:(U.dtype u) ~spec
          (Device.Allocator.Pack {allocator with alloc; free}))
    | Some _ when U.node_tag u = Some "timeline" ->
        Some (State.with_program_lock state (fun () ->
            if state.State.closed then invalid_arg "Metal device is closed";
            match state.State.timeline with
            | Some buffer -> buffer
            | None ->
                let buffer = host_buffer 2 in
                state.timeline <- Some buffer;
                buffer))
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
    | Ops.Custom_function, U.Arg.String "submit_metal_compute_0", [linear; dependency] ->
        let commands = ref [] and rows = ref [] and sizes = ref [] and used = ref 0 in
        let signal = ref None and stamps = ref [] in
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
                let at = align !used 8 in
                used := at + 48;
                if List.exists (function U.Launch_sym _ -> true | _ -> false) dims then begin
                  let values ds = List.init 3 (fun i ->
                      if i >= List.length ds then U.const (Const.int Dtype.uint64 1) else
                      match List.nth ds i with
                      | U.Launch_int n -> U.const (Const.int Dtype.uint64 n)
                      | U.Launch_float f -> U.const (Const.int Dtype.uint64 (int_of_float f))
                      | U.Launch_sym v -> U.cast ~src:v ~dtype:Dtype.uint64) in
                  List.iteri (fun i v -> rows := (at + 8 * i, v) :: !rows)
                    (values info.global_size @ values info.local_size);
                  sizes := (List.length !commands, at) :: !sizes
                end;
                commands := !commands @ [{object_; global; local; offset; sizes = at}]
            | _, U.Arg.Typed ("timestamp", _) ->
                let timestamp = U.shrink ~src:(U.src node).(0) ~offset:(U.const_int 1)
                    ~size:(U.const_int 1) in
                stamps := U.getaddr ~device:"CPU" ~src:timestamp () :: !stamps
            | _, U.Arg.Typed ("store", _) -> signal := Some (U.src node).(1)
            | _, U.Arg.Typed (("barrier" | "wait"), _) -> ()
            | _ -> invalid_arg "Metal queue: unsupported instruction") (U.children linear);
        let header = align !used 8 in
        let profile = !stamps <> [] in
        if profile && List.length !stamps <> 2 * List.length !commands then
          invalid_arg "Metal queue: timestamps must bracket each command";
        let pipeline_count = List.length (Helpers.dedup_by Stdlib.(=)
            (List.map (fun c -> Program.key c.object_) !commands)) in
        let stamp_offset = header + 8 * (5 + pipeline_count + 4 * List.length !commands) in
        List.iteri (fun i stamp -> rows := (stamp_offset + 8 * i, stamp) :: !rows)
          (List.rev !stamps);
        let size = stamp_offset + 8 * List.length !stamps in
        let desc = {commands = !commands; header} in
        let buffer = U.placeholder ~shape:[size] ~dtype:Dtype.uint8
            ~slot:(U.fresh_buffer_slot ()) ~device:(U.Single name) ~volatile:true
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
          [load (context name) 0; header_ptr; Option.get !signal;
           U.const (Const.int Dtype.uint64 (Bool.to_int profile))])
    | _ -> None

  let create state device_name =
    let host = try Device.get "CPU" with Failure _ -> Tolk_cpu.create "CPU" in
    let completion () =
      let value = match State.timeline state with
        | None -> 0L
        | Some timeline -> Bytes.get_int64_le (B.as_bytes timeline) 8 in
      fun timeout -> ignore timeout; Ffi.hcq_wait state.State.context value in
    (* The shared encoder dispatches kernels above 15 arguments directly;
       all calls still use the same queue, timeline and resource ownership. *)
    Device.{timestamp_divider = 1000.; profile_offset = (fun () -> Profile.calibrate (fun () -> Ffi.profile_clock));
      completion; prepare = (fun () -> Ffi.hcq_wait state.State.context 0L); host = Device.name host; max_kernel_bindings = None; config = (fun () -> "ICB_MAX_BINDINGS=15");
      copy = (fun _ -> None); encode = encode device_name; lower = lower device_name;
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
  let synchronize () = State.synchronize state in
  let queue, bufferize = Queue.create state name in
  Device.make ~name ~allocator ~renderer_set ~synchronize:(fun timeout -> ignore timeout; synchronize ())
    ~shares_host_memory:(Ffi.has_unified_memory state.State.device) ~queue ~bufferize ()
