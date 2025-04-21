(** Metal bindings for OCaml.

    This module provides OCaml bindings to Apple's Metal framework, enabling GPU
    computation through a high-level interface. It supports buffer management,
    compute pipeline setup, and command encoding for GPU operations.

    The module allows you to:
    - Create and manage Metal devices
    - Allocate and manipulate GPU buffers using pointers
    - Compile and execute compute kernels
    - Configure thread groups for parallel computation
    - Coordinate command execution and synchronization

    **Important Note on Memory Management:** Metal objects (devices, queues,
    buffers, etc.) are managed automatically via OCaml's garbage collector. You
    do not need to manually release them. However, the timing of release is
    non-deterministic.

    Example usage:
    {[
      let device = Metal.create_device () in
      let queue = Metal.create_command_queue device in
      let options_int = Metal.resource_options_to_int [Storage_Mode_Shared] in
      let buffer = Metal.create_buffer device 1024L options_int in
      ...
    ]} *)

exception Metal_error of string
(** Exception raised when a Metal operation fails. *)

(** Options that control how a resource allocates and stores its data in the CPU
    and GPU memory. Use {!resource_options_to_int} to convert a list of these
    options into the integer bitmask required by buffer creation functions. *)
type resource_options =
  | CPU_Cache_Mode_Default
      (** The default CPU cache mode (MTLCPUCacheModeDefaultCache) *)
  | CPU_Cache_Mode_Write_Combined
      (** Write-combined CPU cache mode (MTLCPUCacheModeWriteCombined) *)
  | Storage_Mode_Shared  (** Shared memory (MTLStorageModeShared) *)
  | Storage_Mode_Managed
      (** Managed memory (MTLStorageModeManaged). Requires synchronization using
          {!buffer_did_modify_range} or blit commands. *)
  | Storage_Mode_Private  (** GPU-private memory (MTLStorageModePrivate) *)
  | Storage_Mode_Memoryless  (** Memoryless (MTLStorageModeMemoryless) *)
  | Hazard_Tracking_Default
      (** Default hazard tracking (MTLHazardTrackingModeDefault) *)
  | Hazard_Tracking_Untracked
      (** Untracked hazards (MTLHazardTrackingModeUntracked) *)
  | Hazard_Tracking_Tracked
      (** Tracked hazards (MTLHazardTrackingModeTracked) *)

type device
(** Represents a GPU device *)

type command_queue
(** Schedules commands for a device *)

type buffer
(** A region of GPU-accessible memory *)

type library
(** Contains compiled Metal shader code *)

type function_
(** A handle to a specific Metal kernel function *)

type pipeline_state
(** Compiled pipeline state for a compute function *)

type command_buffer
(** Stores a sequence of commands *)

type command_encoder
(** Encodes compute commands into a command buffer *)

type blit_command_encoder
(** Encodes memory transfer/fill commands *)

type compile_options
(** Options for compiling Metal source code *)

type grid_size = { x : int64; y : int64; z : int64 }
(** Dimensions for thread dispatch grid *)

type thread_group_size = { x : int64; y : int64; z : int64 }
(** Dimensions for a thread group *)

(** {1 Core Device Operations} *)

val create_device : unit -> device
(** [create_device ()] creates a new Metal device representing the default
    system GPU. Raises [Metal_error] on failure. *)

val get_device_name : device -> string
(** [get_device_name device] returns the name of the Metal device. Raises
    [Metal_error] on failure. *)

(** {1 Command Queue Management} *)

val create_command_queue : device -> command_queue
(** [create_command_queue device] creates a new command queue for the given
    device. Raises [Metal_error] on failure. *)

(** {1 Buffer Operations} *)

val create_buffer : device -> int64 -> resource_options list -> buffer
(** [create_buffer device size_bytes options] creates a new Metal buffer with
    the given size and options. Raises [Metal_error] on failure. *)

val create_buffer_with_pointer :
  device -> nativeint -> int64 -> resource_options list -> buffer
(** [create_buffer_with_pointer device ptr size_bytes options] creates a new
    Metal buffer using the provided CPU memory pointer [ptr] and size. The
    memory **must remain valid** for the lifetime of the buffer and meet Metal's
    alignment requirements (often page-aligned). This typically only works
    reliably with [Storage_Mode_Shared] options. Raises [Metal_error] on
    failure. *)

val create_buffer_with_data :
  device ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
  resource_options list ->
  buffer
(** [create_buffer_with_data device bigarray options] creates a new Metal buffer
    initialized with the contents of the provided [bigarray]. The buffer's size
    matches the bigarray's size in bytes. This typically works best with
    [Storage_Mode_Shared] options. Raises [Metal_error] on failure. *)

val create_buffer_with_bytes :
  device -> bytes -> resource_options list -> buffer
(** [create_buffer_with_bytes device bytes options] creates a new Metal buffer
    initialized with the contents of the provided [bytes]. The buffer's size
    matches the bytes' length. This typically works best with
    [Storage_Mode_Shared] options. Raises [Metal_error] on failure. *)

val buffer_length : buffer -> int64
(** [buffer_length buffer] returns the size in bytes of the given buffer. *)

val buffer_contents : buffer -> nativeint
(** [buffer_contents buffer] returns a CPU-accessible pointer ([nativeint]) to
    the buffer's contents. **Warning:** This function will raise [Metal_error]
    if the buffer storage mode does not allow direct CPU access (e.g.,
    [Storage_Mode_Private]). For [Storage_Mode_Managed] buffers, this pointer
    accesses the CPU copy; ensure synchronization after GPU writes or before GPU
    reads if modifications are made via this pointer. *)

val copy_to_buffer :
  buffer -> offset:int64 -> ptr:nativeint -> num_bytes:int64 -> unit
(** [copy_to_buffer buffer ~offset ~ptr ~num_bytes] copies [num_bytes] from the
    CPU memory pointed to by [ptr] into the Metal buffer at [offset].
    **Warning:** This function will raise [Metal_error] if the buffer storage
    mode does not allow direct CPU access (e.g., [Storage_Mode_Private]). For
    [Storage_Mode_Managed] buffers, this modifies the CPU copy. You **must**
    call {!buffer_did_modify_range} afterwards to notify Metal before the GPU
    accesses this data. *)

val copy_from_buffer :
  buffer -> offset:int64 -> ptr:nativeint -> num_bytes:int64 -> unit
(** [copy_from_buffer buffer ~offset ~ptr ~num_bytes] copies [num_bytes] from
    the Metal buffer at [offset] into the CPU memory pointed to by [ptr].
    **Warning:** This function will raise [Metal_error] if the buffer storage
    mode does not allow direct CPU access (e.g., [Storage_Mode_Private]). For
    [Storage_Mode_Managed] buffers, this reads the CPU copy; ensure the GPU
    commands that produced the data have completed and potentially synchronize
    if necessary before calling this. *)

(** {1 Managed Buffer Synchronization} *)

val buffer_did_modify_range : buffer -> offset:int64 -> length:int64 -> unit
(** [buffer_did_modify_range buffer ~offset ~length] notifies Metal that the CPU
    has modified the specified range (in bytes) of a buffer with
    [Storage_Mode_Managed]. This **must** be called after CPU writes (e.g., via
    {!copy_to_buffer} or {!buffer_contents}) and before the GPU reads from that
    range within a command buffer. Does nothing for non-managed buffers. *)

(** {1 Compute Pipeline Setup} *)

val create_compile_options : unit -> compile_options
(** [create_compile_options ()] creates a new set of compile options. Raises
    [Metal_error] on failure. *)

val set_compile_option_fast_math_enabled : compile_options -> bool -> unit
(** [set_compile_option_fast_math_enabled opts enabled] enables or disables fast
    (less precise) math optimizations during compilation. *)

val create_library_with_source :
  device -> string -> ?options:compile_options -> unit -> library
(** [create_library_with_source device source ?options ()] creates a new Metal
    library by compiling the provided [source] code string. Optional [options]
    can control compilation behavior. Raises [Metal_error] on failure (e.g.,
    compile errors). *)

val create_library_with_data : device -> string -> library
(** [create_library_with_data device data] creates a new Metal library from
    pre-compiled metallib data (provided as a string/bytes). Raises
    [Metal_error] on failure. *)

val create_function_with_name : library -> string -> function_
(** [create_function_with_name library name] looks up and creates a function
    handle for the kernel named [name] within the library. Raises [Metal_error]
    if the function is not found. *)

val create_compute_pipeline_state : device -> function_ -> pipeline_state
(** [create_compute_pipeline_state device function_] creates a compute pipeline
    state object for the given function. This compiles the function for the
    device. Raises [Metal_error] on failure. *)

val get_pipeline_state_max_total_threads_per_threadgroup :
  pipeline_state -> int64
(** [get_pipeline_state_max_total_threads_per_threadgroup state] returns the
    maximum total number of threads ([width * height * depth]) allowed in a
    single threadgroup for this pipeline state. *)

(** {1 Command Buffer Management} *)

val create_command_buffer : command_queue -> command_buffer
(** [create_command_buffer queue] creates a new, empty command buffer associated
    with the queue. Raises [Metal_error] on failure. *)

(** {1 Compute Command Encoding} *)

val create_compute_command_encoder : command_buffer -> command_encoder
(** [create_compute_command_encoder buffer] creates an encoder for appending
    compute commands to the command buffer. Raises [Metal_error] on failure. *)

val set_compute_pipeline_state : command_encoder -> pipeline_state -> unit
(** [set_compute_pipeline_state encoder state] sets the active compute pipeline
    state for subsequent dispatch calls. *)

val set_buffer : command_encoder -> buffer -> offset:int64 -> index:int -> unit
(** [set_buffer encoder buffer ~offset ~index] binds the [buffer] to the
    argument table at [index], starting from the specified byte [offset]. *)

val set_bytes :
  command_encoder -> bytes_ptr:nativeint -> length:int -> index:int -> unit
(** [set_bytes encoder ~bytes_ptr ~length ~index] copies [length] bytes directly
    from the CPU memory pointed to by [bytes_ptr] into the argument table at
    [index]. The data is copied immediately; [bytes_ptr] only needs to be valid
    during this call. Suitable for small amounts of uniform data. *)

val dispatch_thread_groups :
  command_encoder ->
  grid_size:grid_size ->
  thread_group_size:thread_group_size ->
  unit
(** [dispatch_thread_groups encoder ~grid_size ~thread_group_size] encodes a
    compute dispatch command using the specified grid and threadgroup
    dimensions. *)

val end_encoding : command_encoder -> unit
(** [end_encoding encoder] completes the encoding process for this compute
    encoder. No more commands can be added after this. *)

(** {1 Blit Command Encoding (Memory Operations)} *)

val create_blit_command_encoder : command_buffer -> blit_command_encoder
(** [create_blit_command_encoder buffer] creates an encoder for appending memory
    transfer and fill commands (blit operations) to the command buffer. Raises
    [Metal_error] on failure. *)

val copy_from_buffer_to_buffer :
  blit_command_encoder ->
  src_buffer:buffer ->
  src_offset:int64 ->
  dst_buffer:buffer ->
  dst_offset:int64 ->
  size:int64 ->
  unit
(** [copy_from_buffer_to_buffer encoder ~src_buffer ~src_offset ~dst_buffer
     ~dst_offset ~size] encodes a GPU-side command to copy [size] bytes from
    [src_buffer] at [src_offset] to [dst_buffer] at [dst_offset]. Works
    efficiently for all buffer storage modes, including [Storage_Mode_Private].
*)

val end_blit_encoding : blit_command_encoder -> unit
(** [end_blit_encoding encoder] completes the encoding process for this blit
    encoder. *)

(** {1 Command Execution and Synchronization} *)

val commit : command_buffer -> unit
(** [commit buffer] submits the command buffer and its encoded commands to the
    command queue for execution on the GPU. *)

val wait_until_completed : command_buffer -> unit
(** [wait_until_completed buffer] blocks the calling CPU thread until the
    command buffer has finished executing on the GPU. *)
