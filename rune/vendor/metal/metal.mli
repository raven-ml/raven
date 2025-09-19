(** Metal bindings for OCaml *)

(** {2 Basic Structures} *)

(** Represents the dimensions of a grid, region, or threadgroup. See
    {{:https://developer.apple.com/documentation/metal/mtlsize} MTLSize}. *)
module Size : sig
  type t = { width : int; height : int; depth : int }
  type mtl

  val from_struct : mtl -> t
  val to_value : t -> mtl
  val make : width:int -> height:int -> depth:int -> mtl
end

(** Represents the origin of a region in 3D space. See
    {{:https://developer.apple.com/documentation/metal/mtlorigin} MTLOrigin}. *)
module Origin : sig
  type t = { x : int; y : int; z : int }
  type mtl

  val from_struct : mtl -> t
  val to_value : t -> mtl
  val make : x:int -> y:int -> z:int -> mtl
end

(** Represents a 3D region. See
    {{:https://developer.apple.com/documentation/metal/mtlregion} MTLRegion}. *)
module Region : sig
  type t = { origin : Origin.t; size : Size.t }
  type mtl

  val make :
    x:int -> y:int -> z:int -> width:int -> height:int -> depth:int -> mtl

  val from_struct : mtl -> t
  val to_value : t -> mtl
end

(** Represents a range with location and length. See
    {{:https://developer.apple.com/documentation/foundation/nsrange} NSRange}.
*)
module Range : sig
  type ns
  type t = { location : int; length : int }

  val from_struct : ns -> t
  val make : location:int -> length:int -> ns
  val to_value : t -> ns
end

(** Represents the GPU device capable of executing Metal commands. See
    {{:https://developer.apple.com/documentation/metal/mtldevice} MTLDevice}. *)
module Device : sig
  type t

  val create_system_default : unit -> t
  (** Returns the default Metal device for the system. See
      {{:https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice}
       MTLCreateSystemDefaultDevice}. *)

  val copy_all_devices : unit -> t array
  (** Returns an array of all the Metal device instances in the system. See
      {{:https://developer.apple.com/documentation/metal/mtlcopyalldevices()/}
       MTLCopyAllDevices}. *)

  (** Represents a Metal GPU family, categorizing devices by feature set and
      capability. See
      {{:https://developer.apple.com/documentation/metal/mtlgpufamily}
       MTLGPUFamily}. *)
  module GPUFamily : sig
    type t =
      | Apple1
      | Apple2
      | Apple3
      | Apple4
      | Apple5
      | Apple6
      | Apple7
      | Apple8
      | Apple9
      | Mac1
      | Mac2
      | Common1
      | Common2
      | Common3
      | MacCatalyst1
      | MacCatalyst2
      | Metal3

    val to_int : t -> int
    val from_int : int -> t
  end

  val supports_family : t -> GPUFamily.t -> bool
  (** Checks if the device supports the specified GPU family. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/supportsfamily(_:)?language=objc}
       supportsFamiliy:}. *)

  (** Describes the level of support for argument buffers. See
      {{:https://developer.apple.com/documentation/metal/mtlargumentbufferstier}
       MTLArgumentBuffersTier}. *)
  module ArgumentBuffersTier : sig
    type t = Tier1 | Tier2

    val to_ulong : t -> Unsigned.ulong
  end

  type attributes = {
    name : string;
    registry_id : Unsigned.ULLong.t;
    max_threads_per_threadgroup : Size.t;
    max_buffer_length : Unsigned.ULong.t;
    max_threadgroup_memory_length : Unsigned.ULong.t;
    argument_buffers_support : ArgumentBuffersTier.t;
    recommended_max_working_set_size : Unsigned.ULLong.t;
    is_low_power : bool;
    is_removable : bool;
    is_headless : bool;
    has_unified_memory : bool;
    peer_count : Unsigned.ULong.t;
    peer_group_id : Unsigned.ULLong.t;
    supported_gpu_families : GPUFamily.t list;
  }

  (** Metal device attributes relevant for compute tasks. *)

  val get_attributes : t -> attributes
  (** Fetches the static compute-relevant attributes of the device. *)
end

(** {2 Resource Configuration} *)

(** Options for configuring Metal resources like buffers and textures. See
    {{:https://developer.apple.com/documentation/metal/mtlresourceoptions}
     MTLResourceOptions}. *)
module ResourceOptions : sig
  type t

  (* Storage Modes (MTLStorageMode) *)
  val storage_mode_shared : t
  (** Shared between CPU and GPU. See
      {{:https://developer.apple.com/documentation/metal/mtlstoragemode/shared}
       MTLStorageModeShared}. *)

  val storage_mode_managed : t
  (** Managed by the system, requiring synchronization. See
      {{:https://developer.apple.com/documentation/metal/mtlstoragemode/managed}
       MTLStorageModeManaged}. *)

  val storage_mode_private : t
  (** Private to the GPU. See
      {{:https://developer.apple.com/documentation/metal/mtlstoragemode/private}
       MTLStorageModePrivate}. *)

  val storage_mode_memoryless : t
  (** On-chip storage only (TBDR). See
      {{:https://developer.apple.com/documentation/metal/mtlstoragemode/memoryless}
       MTLStorageModeMemoryless}. *)

  (* CPU Cache Modes (MTLCPUCacheMode) *)
  val cache_mode_default_cache : t
  (** Default CPU cache mode. See
      {{:https://developer.apple.com/documentation/metal/mtlcpucachemode/defaultcache}
       MTLCPUCacheModeDefaultCache}. *)

  val cache_mode_write_combined : t
  (** Write-combined CPU cache mode. See
      {{:https://developer.apple.com/documentation/metal/mtlcpucachemode/writecombined}
       MTLCPUCacheModeWriteCombined}. *)

  (* Hazard Tracking Modes (MTLHazardTrackingMode) *)
  val hazard_tracking_mode_default : t
  (** Default hazard tracking mode. See
      {{:https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/default}
       MTLHazardTrackingModeDefault}. *)

  val hazard_tracking_mode_untracked : t
  (** Manual hazard tracking via fences/events. See
      {{:https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/untracked}
       MTLHazardTrackingModeUntracked}. *)

  val hazard_tracking_mode_tracked : t
  (** Automatic hazard tracking. See
      {{:https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/tracked}
       MTLHazardTrackingModeTracked}. *)

  val ( + ) : t -> t -> t
  (** Combines resource options using bitwise OR. *)

  val make :
    ?storage_mode:t -> ?cache_mode:t -> ?hazard_tracking_mode:t -> unit -> t
  (** Creates a combined resource option value. Defaults to shared, default
      cache, default hazard tracking. *)
end

(** Options controlling pipeline state creation. See
    {{:https://developer.apple.com/documentation/metal/mtlpipelineoption}
     MTLPipelineOption}. *)
module PipelineOption : sig
  type t

  val none : t
  val argument_info : t
  val buffer_type_info : t

  val fail_on_binary_archive_miss : t
  (** Fail creation if compiled code is not in binary archive. *)

  val ( + ) : t -> t -> t
  (** Combines pipeline options using bitwise OR. *)
end

(** Options for compiling Metal Shading Language (MSL) source code. See
    {{:https://developer.apple.com/documentation/metal/mtlcompileoptions}
     MTLCompileOptions}. *)
module CompileOptions : sig
  type t

  val init : unit -> t
  (** Creates a new, default set of compile options. *)

  (** Specifies the version of the Metal Shading Language to use. See
      {{:https://developer.apple.com/documentation/metal/mtllanguageversion}
       MTLLanguageVersion}. *)
  module LanguageVersion : sig
    type t

    val version_1_0 : t
    val version_1_1 : t
    val version_1_2 : t
    val version_2_0 : t
    val version_2_1 : t
    val version_2_2 : t
    val version_2_3 : t
    val version_2_4 : t
    val version_3_0 : t
    val version_3_1 : t
    val version_3_2 : t
  end

  (** Specifies the type of library to produce. See
      {{:https://developer.apple.com/documentation/metal/mtllibrarytype}
       MTLLibraryType}. *)
  module LibraryType : sig
    type t (* Uses ullong *)

    val executable : t
    val dynamic : t
    val to_ulong : t -> Unsigned.ulong
  end

  (** Specifies the optimization level for the compiler. See
      {{:https://developer.apple.com/documentation/metal/mtllibraryoptimizationlevel}
       MTLLibraryOptimizationLevel}. *)
  module OptimizationLevel : sig
    type t (* Uses ullong *)

    val default : t
    (** Default optimization level optimized for performance. **)

    val size : t
    (** Optimize for size. *)

    val to_ulong : t -> Unsigned.ulong
  end

  (** Specifies the math mode for floating-point optimizations. See
      {{:https://developer.apple.com/documentation/metal/mtlmathmode}
       MTLMathMode}. *)
  module MathMode : sig
    type t = Safe | Relaxed | Fast

    val to_ulong : t -> Unsigned.ulong
    val from_ulong : Unsigned.ulong -> t
  end

  (** Specifies which math functions to use for single precision floating-point.
      See
      {{:https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions}
       MTLMathFloatingPointFunctions}. *)
  module MathFloatingPointFunctions : sig
    type t = Fast | Precise

    val to_ulong : t -> Unsigned.ulong
    val from_ulong : Unsigned.ulong -> t
  end

  val set_fast_math_enabled : t -> bool -> unit
  val get_fast_math_enabled : t -> bool

  val set_math_mode : t -> MathMode.t -> unit
  (** Sets the floating-point arithmetic optimizations mode. *)

  val get_math_mode : t -> MathMode.t

  val set_math_floating_point_functions :
    t -> MathFloatingPointFunctions.t -> unit
  (** Sets the default math functions for single precision floating-point. *)

  val get_math_floating_point_functions : t -> MathFloatingPointFunctions.t
  val set_enable_logging : t -> bool -> unit
  val get_enable_logging : t -> bool
  val set_max_total_threads_per_threadgroup : t -> int -> unit
  val get_max_total_threads_per_threadgroup : t -> int
  val set_language_version : t -> LanguageVersion.t -> unit
  val get_language_version : t -> LanguageVersion.t
  val set_library_type : t -> LibraryType.t -> unit
  val get_library_type : t -> LibraryType.t

  val set_install_name : t -> string -> unit
  (** Sets the install name for dynamic libraries. *)

  val get_install_name : t -> string
  val set_optimization_level : t -> OptimizationLevel.t -> unit
  val get_optimization_level : t -> OptimizationLevel.t
end

(** {2 Resources} *)

(** Common interface for Metal resources like buffers and textures. See
    {{:https://developer.apple.com/documentation/metal/mtlresource} MTLResource}.
*)
module Resource : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string

  val get_device : t -> Device.t
  (** Gets the device the resource belongs to. *)

  (** Resource purgeability states. See
      {{:https://developer.apple.com/documentation/metal/mtlpurgeablestate}
       MTLPurgeableState}. *)
  module PurgeableState : sig
    type t = KeepCurrent | NonVolatile | Volatile | Empty
  end

  val set_purgeable_state : t -> PurgeableState.t -> PurgeableState.t
  (** Sets the purgeable state and returns the previous state. *)

  (** CPU cache modes. See
      {{:https://developer.apple.com/documentation/metal/mtlcpucachemode}
       MTLCPUCacheMode}. *)
  module CPUCacheMode : sig
    type t = DefaultCache | WriteCombined

    val to_ulong : t -> Unsigned.ulong
    val from_ulong : Unsigned.ulong -> t
  end

  val get_cache_mode : t -> CPUCacheMode.t

  (** Resource storage modes. See
      {{:https://developer.apple.com/documentation/metal/mtlstoragemode}
       MTLStorageMode}. *)
  module StorageMode : sig
    type t = Shared | Managed | Private | Memoryless

    val to_ulong : t -> Unsigned.ulong
    val from_ulong : Unsigned.ulong -> t
  end

  val get_storage_mode : t -> StorageMode.t

  (** Resource hazard tracking modes. See
      {{:https://developer.apple.com/documentation/metal/mtlhazardtrackingmode}
       MTLHazardTrackingMode}. *)
  module HazardTrackingMode : sig
    type t = Default | Untracked | Tracked

    val to_ulong : t -> Unsigned.ulong
    val from_ulong : Unsigned.ulong -> t
  end

  val get_hazard_tracking_mode : t -> HazardTrackingMode.t
  val get_resource_options : t -> ResourceOptions.t

  val get_heap : t -> Objc.object_t
  (** Gets the heap the resource was allocated from (if any). Result type needs
      Heap module. *)

  val get_heap_offset : t -> int
  (** Gets the offset within the heap (if placed). *)

  val get_allocated_size : t -> int

  val make_aliasable : t -> unit
  (** Allows future heap allocations to alias this resource's memory. *)

  val is_aliasable : t -> bool
end

(** Represents a block of untyped memory accessible by the GPU. See
    {{:https://developer.apple.com/documentation/metal/mtlbuffer} MTLBuffer}. *)
module Buffer : sig
  type t

  val super : t -> Resource.t
  val on_device : Device.t -> length:int -> ResourceOptions.t -> t

  val on_device_with_bytes :
    Device.t -> bytes:unit Ctypes.ptr -> length:int -> ResourceOptions.t -> t
  (** Creates a buffer and initializes its contents by copying from a pointer.
  *)

  val on_device_with_bytes_no_copy :
    Device.t ->
    bytes:unit Ctypes.ptr ->
    length:int ->
    ?deallocator:(unit -> unit) ->
    ResourceOptions.t ->
    t
  (** Creates a buffer that wraps existing memory without copying. *)

  val length : t -> int
  (** Gets the length of the buffer in bytes. *)

  val contents : t -> unit Ctypes.ptr
  (** Gets a pointer to the buffer's contents (requires appropriate
      storage/cache mode and synchronization). *)

  val did_modify_range : t -> Range.t -> unit
  (** Informs Metal that a range of a managed buffer was modified by the CPU.
      This is not needed for shared buffers (NOTE: validation layer will report
      an error). *)

  val add_debug_marker : t -> marker:string -> Range.t -> unit
  val remove_all_debug_markers : t -> unit

  val get_gpu_address : t -> Unsigned.ULLong.t
  (** Gets the GPU virtual address of the buffer. *)
end

(** {2 Libraries and Functions} *)

(** Identifies the type of a Metal function. See
    {{:https://developer.apple.com/documentation/metal/mtlfunctiontype}
     MTLFunctionType}. *)
module FunctionType : sig
  type t = Vertex | Fragment | Kernel | Visible | Intersection | Mesh | Object

  val to_ulong : t -> Unsigned.ulong
  val from_ulong : Unsigned.ulong -> t
end

(** Represents a single, named function (shader or kernel) within a Metal
    library. See
    {{:https://developer.apple.com/documentation/metal/mtlfunction} MTLFunction}.
*)
module Function : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t
  val get_function_type : t -> FunctionType.t

  val get_name : t -> string
  (** Gets the name of the function as defined in the source code. *)

  val get_options : t -> Unsigned.ULLong.t
  (** Gets the options the function was created with (MTLFunctionOptions). *)
end

(** Represents a compiled Metal library containing one or more functions. See
    {{:https://developer.apple.com/documentation/metal/mtllibrary} MTLLibrary}.
*)
module Library : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t

  val on_device : Device.t -> source:string -> CompileOptions.t -> t
  (** Creates a library by compiling Metal Shading Language source code. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/1433431-newlibrarywithsource}
       newLibraryWithSource:options:error:}. *)

  val on_device_with_data :
    Device.t ->
    unit Ctypes.ptr ->
    t (* dispatch_data_t is tricky, using unit ptr *)
  (** Creates a library from pre-compiled data (e.g., a .metallib file loaded
      into memory). Defensively, the returned value holds on to the data
      pointer. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/makelibrary(data:)?language=objc}
       newLibraryWithData:error:}. *)

  val new_function_with_name : t -> string -> Function.t
  (** Retrieves a specific function from the library by name. See
      {{:https://developer.apple.com/documentation/metal/mtllibrary/makefunction(name:)?language=objc}
       newFunctionWithName:}. *)

  val get_function_names : t -> string array

  val get_library_type : t -> CompileOptions.LibraryType.t
  (** Gets the type of the library (Executable or Dynamic). *)

  val get_install_name : t -> string option
  (** Gets the install name if it's a dynamic library. *)
end

(** {2 Compute Pipeline} *)

(** Describes the configuration for creating a compute pipeline state. See
    {{:https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor}
     MTLComputePipelineDescriptor}. *)
module ComputePipelineDescriptor : sig
  type t

  val create : unit -> t
  val set_label : t -> string -> unit
  val get_label : t -> string
  val set_compute_function : t -> Function.t -> unit
  val get_compute_function : t -> Function.t
  val set_support_indirect_command_buffers : t -> bool -> unit
  val get_support_indirect_command_buffers : t -> bool
end

(** Represents a compiled compute pipeline state object. See
    {{:https://developer.apple.com/documentation/metal/mtlcomputepipelinestate}
     MTLComputePipelineState}. *)
module ComputePipelineState : sig
  type t

  val on_device_with_function :
    Device.t ->
    ?options:PipelineOption.t ->
    ?reflection:bool ->
    Function.t ->
    t * Objc.object_t Ctypes.ptr
  (** Creates a pipeline state from a function. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/makecomputepipelinestate(function:options:reflection:)?language=objc}
       newComputePipelineStateWithFunction:options:reflection:error:}. *)
  (* Returns PSO and optional reflection object *)

  val on_device_with_descriptor :
    Device.t ->
    ?options:PipelineOption.t ->
    ?reflection:bool ->
    ComputePipelineDescriptor.t ->
    t * Objc.object_t Ctypes.ptr
  (** Creates a pipeline state from a descriptor. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/makecomputepipelinestate(descriptor:options:reflection:)?language=objc}
       newComputePipelineStateWithDescriptor:options:reflection:error:}. *)
  (* Returns PSO and optional reflection object *)

  val get_label : t -> string
  val get_device : t -> Device.t
  val get_max_total_threads_per_threadgroup : t -> int

  val get_thread_execution_width : t -> int
  (** Gets the execution width (SIMD group size) for this pipeline. *)

  val get_static_threadgroup_memory_length : t -> int
  (** Gets the amount of statically allocated threadgroup memory in bytes. *)

  val get_support_indirect_command_buffers : t -> bool
  (** Checks if the pipeline supports indirect command buffers. *)
end

(** Log levels for shader debugging. See
    {{:https://developer.apple.com/documentation/metal/mtlloglevel} MTLLogLevel}.
*)
module LogLevel : sig
  (** Log levels for shader logging. See
      {{:https://developer.apple.com/documentation/metal/mtlloglevel}
       MTLLogLevel}. *)
  type t = Undefined | Debug | Info | Notice | Error | Fault

  val from_long : Signed.long -> t
  val to_long : t -> Signed.long
end

(** Descriptor for configuring shader logging. See
    {{:https://developer.apple.com/documentation/metal/mtllogstatedescriptor}
     MTLLogStateDescriptor}. *)
module LogStateDescriptor : sig
  type t
  (** Configuration for creating a log state object. See
      {{:https://developer.apple.com/documentation/metal/mtllogstatedescriptor}
       MTLLogStateDescriptor}. *)

  val create : unit -> t
  (** Creates a new log state descriptor with default values. *)

  val set_level : t -> LogLevel.t -> unit
  (** Sets the minimum log level to capture. *)

  val get_level : t -> LogLevel.t
  (** Gets the minimum log level to capture. *)

  val set_buffer_size : t -> int -> unit
  (** Sets the size (in bytes) of the internal buffer for log messages. Minimum
      1KB. *)

  val get_buffer_size : t -> int
  (** Gets the size (in bytes) of the internal buffer for log messages. *)
end

(** Container for shader log messages. See
    {{:https://developer.apple.com/documentation/metal/mtllogstate} MTLLogState}.
*)
module LogState : sig
  type t
  (** A container for shader log messages. See
      {{:https://developer.apple.com/documentation/metal/mtllogstate}
       MTLLogState}. *)

  val on_device_with_descriptor : Device.t -> LogStateDescriptor.t -> t
  (** Creates a log state object using the specified descriptor. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/4379071-newlogstatewithdescriptor}
       newLogStateWithDescriptor:error:}. *)

  val add_log_handler :
    t ->
    (sub_system:string option ->
    category:string option ->
    level:LogLevel.t ->
    message:string ->
    unit) ->
    unit
  (** Adds a handler block to process log messages. See
      {{:https://developer.apple.com/documentation/metal/mtllogstate/4379067-addloghandler}
       addLogHandler:}. *)
end

(** Configuration for creating command queues. See
    {{:https://developer.apple.com/documentation/metal/mtlcommandqueuedescriptor}
     MTLCommandQueueDescriptor}. *)
module CommandQueueDescriptor : sig
  type t
  (** Configuration for creating a command queue. See
      {{:https://developer.apple.com/documentation/metal/mtlcommandqueuedescriptor}
       MTLCommandQueueDescriptor}. *)

  val create : unit -> t
  (** Creates a new command queue descriptor with default values. *)

  val set_max_command_buffer_count : t -> int -> unit
  (** Sets the maximum number of uncompleted command buffers allowed in the
      queue. *)

  val get_max_command_buffer_count : t -> int
  (** Gets the maximum number of uncompleted command buffers allowed in the
      queue. *)

  val set_log_state : t -> LogState.t option -> unit
  (** Sets the log state object for the command queue (nullable). *)

  val get_log_state : t -> LogState.t option
  (** Gets the log state object for the command queue (nullable). *)
end

(** {2 Command Infrastructure} *)

(** A queue for submitting command buffers to a device. See
    {{:https://developer.apple.com/documentation/metal/mtlcommandqueue}
     MTLCommandQueue}. *)
module CommandQueue : sig
  type t

  val on_device : Device.t -> t

  val on_device_with_max_buffer_count : Device.t -> int -> t
  (** Creates a command queue with a specific maximum number of uncompleted
      command buffers. *)

  val on_device_with_descriptor : Device.t -> CommandQueueDescriptor.t -> t
  (** Creates a command queue using the specified descriptor. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/makecommandqueue(descriptor:)?language=objc}
       newCommandQueueWithDescriptor:}.

      The returned OCaml value will ensure that any OCaml objects referenced by
      the descriptor's lifetime (e.g., for log handlers) are kept alive as long
      as this command queue value is alive. *)

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t
end

(** An object used for GPU-GPU synchronization within a single device. See
    {{:https://developer.apple.com/documentation/metal/mtlevent} MTLEvent}. *)
module Event : sig
  type t

  val get_device : t -> Device.t
  val set_label : t -> string -> unit
  val get_label : t -> string
end

(** A container for encoded commands that the GPU executes. See
    {{:https://developer.apple.com/documentation/metal/mtlcommandbuffer}
     MTLCommandBuffer}. *)
module CommandBuffer : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t
  val get_command_queue : t -> CommandQueue.t

  val get_retained_references : t -> bool
  (** Checks if the buffer retains references to its resources. *)

  val on_queue : CommandQueue.t -> t

  val on_queue_with_unretained_references : CommandQueue.t -> t
  (** Creates a command buffer that does not retain references to its resources.
  *)

  val enqueue : t -> unit
  (** Adds the command buffer to the end of the command queue. *)

  val commit : t -> unit
  (** Commits the command buffer for execution as soon as possible. *)

  val add_scheduled_handler : t -> (t -> unit) -> unit
  (** Registers a block to be called when the buffer is scheduled. *)

  val add_completed_handler : t -> (t -> unit) -> unit
  (** Registers a block to be called when the buffer finishes execution. *)

  val wait_until_scheduled : t -> unit
  (** Blocks the calling thread until the buffer is scheduled. *)

  val wait_until_completed : t -> unit
  (** Blocks the calling thread until the buffer finishes execution. *)

  (** The execution status of the command buffer. See
      {{:https://developer.apple.com/documentation/metal/mtlcommandbufferstatus}
       MTLCommandBufferStatus}. *)
  module Status : sig
    type t =
      | NotEnqueued
      | Enqueued
      | Committed
      | Scheduled
      | Completed
      | Error

    val from_ulong : Unsigned.ulong -> t
    val to_ulong : t -> Unsigned.ulong
  end

  val get_status : t -> Status.t

  val get_error : t -> string option
  (** Gets the error object if the status is Error, otherwise None. *)

  val get_gpu_start_time : t -> float
  (** Gets the host time (seconds) when the GPU started executing the buffer. *)

  val get_gpu_end_time : t -> float
  (** Gets the host time (seconds) when the GPU finished executing the buffer.
  *)

  val encode_wait_for_event : t -> Event.t -> Unsigned.ULLong.t -> unit
  val encode_signal_event : t -> Event.t -> Unsigned.ULLong.t -> unit
  val push_debug_group : t -> string -> unit
  val pop_debug_group : t -> unit
end

(** Base protocol for objects that encode commands into a command buffer. See
    {{:https://developer.apple.com/documentation/metal/mtlcommandencoder}
     MTLCommandEncoder}. *)
module CommandEncoder : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t
  val end_encoding : t -> unit
  val insert_debug_signpost : t -> string -> unit
  val push_debug_group : t -> string -> unit
  val pop_debug_group : t -> unit
end

(** Usage flags for resources within a command encoder. See
    {{:https://developer.apple.com/documentation/metal/mtlresourceusage}
     MTLResourceUsage}. *)
module ResourceUsage : sig
  type t

  val read : t
  val write : t
  val ( + ) : t -> t -> t
end

(** {2 Indirect Command Buffers} *)

(** Types of commands that can be encoded into an indirect command buffer. See
    {{:https://developer.apple.com/documentation/metal/mtlindirectcommandtype}
     MTLIndirectCommandType}. *)
module IndirectCommandType : sig
  type t

  val draw : t
  val draw_indexed : t
  val draw_patches : t
  val draw_indexed_patches : t
  val concurrent_dispatch : t
  val concurrent_dispatch_threads : t
  val ( + ) : t -> t -> t
end

(** Describes the configuration for an indirect command buffer. See
    {{:https://developer.apple.com/documentation/metal/mtlindirectcommandbufferdescriptor}
     MTLIndirectCommandBufferDescriptor}. *)
module IndirectCommandBufferDescriptor : sig
  type t

  val create : unit -> t
  val set_command_types : t -> IndirectCommandType.t -> unit
  val get_command_types : t -> IndirectCommandType.t

  val set_inherit_pipeline_state : t -> bool -> unit
  (** Configures whether commands inherit the pipeline state from the encoder.
  *)

  val get_inherit_pipeline_state : t -> bool

  val set_inherit_buffers : t -> bool -> unit
  (** Configures whether commands inherit buffer bindings from the encoder. *)

  val get_inherit_buffers : t -> bool

  val set_max_kernel_buffer_bind_count : t -> int -> unit
  (** Sets the maximum number of buffer bindings commands can specify. *)

  val get_max_kernel_buffer_bind_count : t -> int
end

(** Represents a command within an indirect command buffer specific to compute
    operations. See
    {{:https://developer.apple.com/documentation/metal/mtlindirectcomputecommand}
     MTLIndirectComputeCommand}. *)
module IndirectComputeCommand : sig
  type t

  val set_compute_pipeline_state : t -> ComputePipelineState.t -> unit

  val set_kernel_buffer : t -> ?offset:int -> index:int -> Buffer.t -> unit
  (** Sets a buffer argument for the command. *)

  val concurrent_dispatch_threadgroups :
    t -> threadgroups_per_grid:Size.t -> threads_per_threadgroup:Size.t -> unit
  (** Specifies dispatch dimensions for the command. *)

  val set_barrier : t -> unit
end

(** A buffer containing pre-encoded commands that can be executed efficiently by
    the GPU. See
    {{:https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer}
     MTLIndirectCommandBuffer}. *)
module IndirectCommandBuffer : sig
  type t

  val on_device_with_descriptor :
    Device.t ->
    IndirectCommandBufferDescriptor.t ->
    max_command_count:int ->
    options:ResourceOptions.t ->
    t
  (** Creates an indirect command buffer. See
      {{:https://developer.apple.com/documentation/metal/mtldevice/3088425-newindirectcommandbuffer}
       newIndirectCommandBufferWithDescriptor:maxCommandCount:options:}. *)

  val get_size : t -> int
  val indirect_compute_command_at_index : t -> int -> IndirectComputeCommand.t

  val reset_with_range : t -> Range.t -> unit
  (** Resets a range of commands in the buffer, making them no-ops. *)
end

(** Encodes compute commands. See
    {{:https://developer.apple.com/documentation/metal/mtlcomputecommandencoder}
     MTLComputeCommandEncoder}. *)
module ComputeCommandEncoder : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t

  (** Possible ways to dispatch commands within an encoder. See
      {{:https://developer.apple.com/documentation/metal/mtldispatchtype}
       MTLDispatchType}. *)
  module DispatchType : sig
    type t = Serial | Concurrent

    val to_ulong : t -> Unsigned.ulong
  end

  val on_buffer : CommandBuffer.t -> t
  val on_buffer_with_dispatch_type : CommandBuffer.t -> DispatchType.t -> t
  val end_encoding : t -> unit
  val insert_debug_signpost : t -> string -> unit
  val push_debug_group : t -> string -> unit
  val pop_debug_group : t -> unit
  val set_compute_pipeline_state : t -> ComputePipelineState.t -> unit

  val set_buffer : t -> ?offset:int -> index:int -> Buffer.t -> unit
  (** Sets a buffer argument for the compute function. *)

  val set_buffers : t -> offsets:int list -> index:int -> Buffer.t list -> unit
  (** Sets multiple buffer arguments. The offsets length must match the number
      of buffers. *)

  val set_bytes : t -> bytes:unit Ctypes.ptr -> length:int -> index:int -> unit
  (** Sets inline constant data as a buffer argument. *)

  val set_threadgroup_memory_length : t -> length:int -> index:int -> unit
  (** Sets the length of a threadgroup memory argument. *)

  val dispatch_threadgroups :
    t -> threadgroups_per_grid:Size.t -> threads_per_threadgroup:Size.t -> unit
  (** Dispatches threadgroups for execution. *)

  val use_resource : t -> Resource.t -> ResourceUsage.t -> unit
  (** Declares that a resource will be used by the following commands. *)

  val use_resources : t -> Resource.t list -> ResourceUsage.t -> unit
  (** Declares that multiple resources will be used by the following commands.
  *)

  val execute_commands_in_buffer :
    t -> IndirectCommandBuffer.t -> Range.t -> unit
end

(** An object used for fine-grained resource synchronization within a command
    encoder. See
    {{:https://developer.apple.com/documentation/metal/mtlfence} MTLFence}. *)
module Fence : sig
  type t

  val on_device : Device.t -> t
  val get_device : t -> Device.t
  val set_label : t -> string -> unit
  val get_label : t -> string
end

(** Encodes resource copy and synchronization commands. See
    {{:https://developer.apple.com/documentation/metal/mtlblitcommandencoder}
     MTLBlitCommandEncoder}. *)
module BlitCommandEncoder : sig
  type t

  val set_label : t -> string -> unit
  val get_label : t -> string
  val get_device : t -> Device.t
  val end_encoding : t -> unit
  val insert_debug_signpost : t -> string -> unit
  val push_debug_group : t -> string -> unit
  val pop_debug_group : t -> unit
  val on_buffer : CommandBuffer.t -> t

  val copy_from_buffer :
    t ->
    source_buffer:Buffer.t ->
    source_offset:int ->
    destination_buffer:Buffer.t ->
    destination_offset:int ->
    size:int ->
    unit

  val fill_buffer : t -> Buffer.t -> Range.t -> value:int -> unit
  (** Fills a buffer range with a {i byte} value (0-255). *)

  val synchronize_resource : t -> Resource.t -> unit
  (** Synchronizes a managed resource between CPU and GPU. *)

  val update_fence : t -> Fence.t -> unit
  val wait_for_fence : t -> Fence.t -> unit
end

(** {2 Synchronization} *)

(** An object used for GPU-GPU or CPU-GPU synchronization, potentially across
    multiple devices or processes. See
    {{:https://developer.apple.com/documentation/metal/mtlsharedevent}
     MTLSharedEvent}. *)
module SharedEvent : sig
  type t

  (** Provides a simple interface for handling MTLSharedEvent notifications. See
      {{:https://developer.apple.com/documentation/metal/mtlsharedeventlistener?language=objc}
       MTLSharedEventListener}. *)
  module SharedEventListener : sig
    type t

    val init : unit -> t
    (** Creates a default listener with its own dispatch queue. See
        {{:https://developer.apple.com/documentation/metal/mtlsharedeventlistener/2966578-init?language=objc}
         init}. *)
  end

  (** A serializable object used to recreate a MTLSharedEvent object in another
      process. See
      {{:https://developer.apple.com/documentation/metal/mtlsharedeventhandle?language=objc}
       MTLSharedEventHandle}. *)
  module SharedEventHandle : sig
    type t

    val get_label : t -> string option
    (** The label associated with the original shared event. *)
  end

  val super : t -> Event.t
  val on_device : Device.t -> t
  val get_device : t -> Device.t
  val set_label : t -> string -> unit
  val get_label : t -> string

  val set_signaled_value : t -> Unsigned.ULLong.t -> unit
  (** Sets the event's signaled value from the CPU. *)

  val get_signaled_value : t -> Unsigned.ULLong.t

  val notify_listener :
    t ->
    SharedEventListener.t ->
    value:Unsigned.ULLong.t ->
    (t -> Unsigned.ULLong.t -> unit) ->
    unit
  (** Schedules a notification handler block to be called when the event's
      signaled value reaches or exceeds the specified value. See
      {{:https://developer.apple.com/documentation/metal/mtlsharedevent/notify(_:atvalue:block:)?language=objc}
       notifyListener:atValue:block:}. *)

  val new_shared_event_handle : t -> SharedEventHandle.t
  (** Creates a serializable handle for this shared event. See
      {{:https://developer.apple.com/documentation/metal/mtlsharedevent/makesharedeventhandle()?language=objc}
       newSharedEventHandle}. *)

  val wait_until_signaled_value :
    t -> value:Unsigned.ULLong.t -> timeout_ms:Unsigned.ULLong.t -> bool
  (** Synchronously waits until the signaled value reaches or exceeds the target
      value, or the timeout elapses. Returns [true] if the value was reached,
      [false] on timeout. See
      {{:https://developer.apple.com/documentation/metal/mtlsharedevent/wait(untilsignaledvalue:timeoutms:)?language=objc}
       waitUntilSignaledValue:timeoutMS:}. *)
end

(** {2 Dynamic Library Placeholder} *)

(** Represents a dynamically linkable Metal library. Bindings are complex and
    omitted for now. See
    {{:https://developer.apple.com/documentation/metal/mtldynamiclibrary}
     MTLDynamicLibrary}. *)
module DynamicLibrary : sig
  type t
end
