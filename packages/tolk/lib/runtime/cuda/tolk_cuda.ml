(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

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

  external host_write : nativeint -> bytes -> unit
    = "caml_tolk_cuda_host_write"

  external memcpy_htod_async : nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_htod_async"

  external memcpy_dtoh : bytes -> nativeint -> unit
    = "caml_tolk_cuda_memcpy_dtoh"

  external memcpy_dtod_async : nativeint -> nativeint -> int -> unit
    = "caml_tolk_cuda_memcpy_dtod_async"

  external module_load : bytes -> nativeint = "caml_tolk_cuda_module_load"

  external module_get_function : nativeint -> string -> nativeint
    = "caml_tolk_cuda_module_get_function"

  external module_unload : nativeint -> unit = "caml_tolk_cuda_module_unload"

  external launch_kernel :
    nativeint ->
    nativeint array ->
    int array ->
    int array ->
    int array ->
    bool ->
    float option = "caml_tolk_cuda_launch_kernel_bc" "caml_tolk_cuda_launch_kernel"

  external nvrtc_version : unit -> int * int = "caml_tolk_cuda_nvrtc_version"

  external nvrtc_compile : string -> string array -> (bytes, string) result
    = "caml_tolk_cuda_nvrtc_compile"
end

(* Compute capabilities newer than the highest supported source-generation
   tier fall back to that tier; the driver JIT-compiles its PTX for the actual
   GPU. Extend both functions when a new tier is added. *)
let target_of_capability ~major ~minor =
  match (major * 10) + minor with
  | v when v >= 90 -> Gpu_target.SM90
  | v when v >= 89 -> Gpu_target.SM89
  | v when v >= 80 -> Gpu_target.SM80
  | _ -> Gpu_target.SM75

let arch_of_target = function
  | Gpu_target.SM75 -> "sm_75"
  | Gpu_target.SM80 -> "sm_80"
  | Gpu_target.SM89 -> "sm_89"
  | Gpu_target.SM90 -> "sm_90"

module State = struct
  type t = {
    context : nativeint;
    target : Gpu_target.cuda;
    mutable pending_copyin : (nativeint * int * Device.Buffer_spec.t) list;
    (* The device's LRU-wrapped allocator; set right after creation and used
       by copyin staging and pending-buffer release. *)
    mutable allocator : nativeint Device.Allocator.t option;
  }

  let devices : t list ref = ref []

  let create device_id =
    Ffi.init ();
    let cu_device = Ffi.device_get device_id in
    let context = Ffi.ctx_create cu_device in
    let major, minor = Ffi.compute_capability cu_device in
    let target =
      match Gpu_target.cuda_of_env () with
      | Some target -> target
      | None -> target_of_capability ~major ~minor
    in
    let state = { context; target; pending_copyin = []; allocator = None } in
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
end

module Allocator = struct
  let host_spec = { Device.Buffer_spec.default with host = true }

  let raw state =
    let alloc size spec =
      Ffi.ctx_set_current state.State.context;
      match spec.Device.Buffer_spec.external_ptr with
      | Some ptr -> ptr
      | None ->
          if spec.Device.Buffer_spec.host then Ffi.mem_host_alloc size
          else Ffi.mem_alloc size
    in
    let free buf _size spec =
      match spec.Device.Buffer_spec.external_ptr with
      | Some _ -> ()
      | None ->
          if spec.Device.Buffer_spec.host then Ffi.mem_free_host buf
          else Ffi.mem_free buf
    in
    (* Host-to-device copies stage through a pinned host buffer drawn from the
       device allocator's LRU cache so the copy can be issued asynchronously;
       the staging buffer is released back to the cache at the next
       synchronize. *)
    let copyin buf bytes =
      Ffi.ctx_set_current state.State.context;
      let size = Bytes.length bytes in
      let host =
        (Option.get state.State.allocator).Device.Allocator.alloc size
          host_spec
      in
      state.State.pending_copyin <-
        (host, size, host_spec) :: state.State.pending_copyin;
      Ffi.host_write host bytes;
      Ffi.memcpy_htod_async buf host size
    in
    let copyout bytes buf =
      State.synchronize_system ();
      Ffi.ctx_set_current state.State.context;
      Ffi.memcpy_dtoh bytes buf
    in
    let transfer ~dest ~src nbytes =
      Ffi.ctx_set_current state.State.context;
      Ffi.memcpy_dtod_async dest src nbytes
    in
    let offset buf _size byte_offset =
      if byte_offset < 0 then
        invalid_arg "CUDA buffer offset must be non-negative";
      Nativeint.add buf (Nativeint.of_int byte_offset)
    in
    {
      Device.Allocator.alloc;
      free;
      copyin;
      copyout;
      addr = Fun.id;
      offset = Some offset;
      transfer = Some transfer;
      supports_transfer = true;
      copy_from_disk = None;
      supports_copy_from_disk = false;
    }

  let create state =
    let allocator = Device.Lru_allocator.wrap (raw state) in
    state.State.allocator <- Some allocator;
    Device.Allocator.Pack allocator
end

module Compiler_nvrtc = struct
  let compile_options arch =
    let includes =
      match Helpers.getenv_str "CUDA_PATH" "" with
      | "" -> [ "-I/usr/local/cuda/include"; "-I/usr/include"; "-I/opt/cuda/include" ]
      | cuda_path -> [ "-I" ^ cuda_path ^ "/include" ]
    in
    let options = ("--gpu-architecture=" ^ arch) :: includes in
    let major, minor = Ffi.nvrtc_version () in
    if (major, minor) >= (12, 4) then options @ [ "--minimal" ] else options

  let create arch =
    let options = Array.of_list (compile_options arch) in
    let compile src =
      match Ffi.nvrtc_compile src options with
      | Ok ptx -> ptx
      | Error msg -> raise (Compiler.Compile_error msg)
    in
    Compiler.make ~name:"CUDA" ~cachekey:("compile_cuda_" ^ arch) ~compile ()
end

module Program = struct
  let runtime state entry_name lib ~runtimevars:_ =
    Ffi.ctx_set_current state.State.context;
    let module_ = Ffi.module_load lib in
    let func = Ffi.module_get_function module_ entry_name in
    let default_local = [| 1; 1; 1 |] in
    let call bufs ~global ~local ~vals ~wait ~timeout:_ =
      let local = Option.value local ~default:default_local in
      Ffi.ctx_set_current state.State.context;
      Ffi.launch_kernel func bufs (Array.map Int64.to_int vals) global local
        wait
    in
    let free () = Ffi.module_unload module_ in
    Device.{ call; free }
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
  let allocator = Allocator.create state in
  let compiler = Compiler_nvrtc.create (arch_of_target state.State.target) in
  let renderer =
    Renderer.with_compiler compiler (Cstyle.cuda state.State.target)
  in
  let renderer_set = Device.Renderer_set.make [ (renderer, None) ] in
  let runtime = Program.runtime state in
  let synchronize () = State.synchronize state in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ()
