(* This file is now part of the rune_jit library and independent of Nx_core. *)
(* It refers to types defined in `ir.ml` within the same library. *)

module type S = sig
  val name : string
  (** The unique name of this backend (e.g., "CLANG_CPU", "METAL_GPU"). *)

  (** Information and capabilities of the target device for this backend. *)
  module Device_info : sig
    type t

    val get_default :
      unit -> t (* Get the default configuration for this backend *)

    val max_shared_memory : t -> int
    val max_workgroup_size : t -> int array (* e.g., [|1024; 1024; 64|] *)

    val supports_dtype : t -> Ir.Dtype.any -> bool
    (** Checks if the backend supports a given Rune_jit.Ir.Dtype.any. *)

    val renderer_float4_str :
      t -> string option (* e.g., Some "(float4)" or None *)

    val renderer_smem_prefix :
      t -> string (* e.g., "__local " or "__shared__ " *)

    val renderer_barrier_str : t -> string
    (* e.g., "barrier(CLK_LOCAL_MEM_FENCE);" *)
  end

  type device_buffer_native
  (** Opaque type representing a native device buffer for this backend. *)

  type ('a_elt, 'b_layout_phantom) device_buffer = {
    native_buffer : device_buffer_native;
    size_in_bytes : int;
    dtype : ('a_elt, 'b_layout_phantom) Ir.Dtype.t;
    device_info : Device_info.t;
  }
  (** Represents a buffer on the device, using Rune_jit.Ir.Dtype.t. The type
      parameters ('a_elt, 'b_layout_phantom) match those in Ir.Dtype.t. *)

  type compiled_artifact_native
  (** Opaque type representing a compiled kernel artifact. *)

  type compiled_artifact = {
    native_artifact : compiled_artifact_native;
    entry_points : string list; (* Names of callable kernels within *)
  }

  type callable_kernel_native
  (** Opaque type representing a callable kernel function. *)

  type callable_kernel = {
    native_kernel : callable_kernel_native;
    name : string;
  }

  (** Renders the lowered IR (from Rune_jit.Ir) into source code. *)
  module Renderer : sig
    val render :
      device_info:Device_info.t ->
      lowered_ir:Ir.Lowered.graph_t ->
      (* Uses Rune_jit.Ir.Lowered.graph_t *)
      kernel_name:string ->
      string
  end

  (** Compiles source code into a loadable artifact. *)
  module Compiler : sig
    type compile_options

    val default_options : Device_info.t -> compile_options

    val compile :
      device_info:Device_info.t ->
      source_code:string ->
      options:compile_options ->
      (compiled_artifact, string) result
  end

  (** Manages device interaction: memory, kernel launch. *)
  module Runtime : sig
    val allocate_buffer :
      device_info:Device_info.t ->
      size_in_bytes:int ->
      dtype:('a_elt, 'b_layout_phantom) Ir.Dtype.t ->
      (('a_elt, 'b_layout_phantom) device_buffer, string) result
    (** Allocates a buffer on the device. Uses Rune_jit.Ir.Dtype.t for
        specifying the data type. *)

    val deallocate_buffer : ('a_elt, 'b_layout_phantom) device_buffer -> unit

    val copy_to_device :
      dest_buffer:('a_elt, 'b_layout_phantom) device_buffer ->
      host_data:nativeint ->
      host_data_offset_bytes:int ->
      copy_size_bytes:int ->
      (unit, string) result

    val copy_from_device :
      src_buffer:('a_elt, 'b_layout_phantom) device_buffer ->
      host_dest_ptr:nativeint ->
      device_data_offset_bytes:int ->
      copy_size_bytes:int ->
      (unit, string) result

    val get_kernel :
      artifact:compiled_artifact ->
      kernel_name:string ->
      (callable_kernel, string) result

    val launch_kernel :
      kernel:callable_kernel ->
      global_dims:int array ->
      local_dims:int array option ->
      args:nativeint list ->
      (unit, string) result

    val synchronize : device_info:Device_info.t -> unit
  end
end
