(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* backend_intf.ml *)

open Ir

type 'a compiled_artifact = { native_artifact : 'a; entry_points : string list }
type 'a callable_kernel = { native_kernel : 'a; name : string }

type ('a, 'b) device_buffer = {
  native_buffer : 'b;
  size_in_bytes : int;
  dtype : 'a Dtype.t;
}

type 'b any_device_buffer =
  | Any_Device_Buffer : ('a, 'b) device_buffer -> 'b any_device_buffer
[@@unboxed]

module type S = sig
  val name : string

  (* ───── Shared Opaque Handles ───── *)

  type device_info
  type device_buffer_native
  type compiled_artifact_native
  type callable_kernel_native

  (* ───── High-level Buffer Wrappers ───── *)

  type nonrec 'a device_buffer = ('a, device_buffer_native) device_buffer
  type nonrec any_device_buffer = device_buffer_native any_device_buffer
  type nonrec compiled_artifact = compiled_artifact_native compiled_artifact
  type nonrec callable_kernel = callable_kernel_native callable_kernel

  (* ───── Sub-module Interfaces ───── *)

  module Device_info : sig
    val get_default : unit -> device_info
    val max_shared_memory : device_info -> int
    val max_workgroup_size : device_info -> int array
    val supports_dtype : device_info -> Dtype.any -> bool

    (* helpers for the renderer *)
    val renderer_float4_str : device_info -> string option
    val renderer_smem_prefix : device_info -> string
    val renderer_barrier_str : device_info -> string
  end

  module Renderer : sig
    val render :
      device_info:device_info ->
      lowered_ir:Ir.Lowered.graph_t ->
      kernel_name:string ->
      string
  end

  module Compiler : sig
    type compile_options

    val default_options : device_info -> compile_options

    val compile :
      device_info:device_info ->
      source_code:string ->
      options:compile_options ->
      (compiled_artifact, string) result
  end

  module Runtime : sig
    val allocate_buffer :
      device_info:device_info ->
      size_in_bytes:int ->
      dtype:'a Dtype.t ->
      ('a device_buffer, string) result

    val copy_to_device :
      dest_buffer:'a device_buffer ->
      host_data:nativeint ->
      host_data_offset_bytes:int ->
      copy_size_bytes:int ->
      (unit, string) result

    val copy_from_device :
      src_buffer:'a device_buffer ->
      host_dest_ptr:nativeint ->
      device_data_offset_bytes:int ->
      copy_size_bytes:int ->
      (unit, string) result

    val get_kernel :
      artifact:compiled_artifact ->
      kernel_name:string ->
      (callable_kernel, string) result

    val launch_kernel :
      ?local_dims:int array (* None = let backend pick *) ->
      device_info:device_info ->
      global_dims:int array (* [|gx; gy; gz|] *) ->
      args:any_device_buffer list ->
      callable_kernel ->
      (unit, string) result

    val synchronize : device_info:device_info -> unit
  end
end
