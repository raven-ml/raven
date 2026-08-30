(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPU runtime.

    Building blocks for driving NVIDIA GPUs through their hardware
    command queues: generic queue machinery ({!Hcq}), the generated
    driver tables ({!Nv_tables}), kernel launch descriptors ({!Qmd}),
    and the command-stream builders ({!Compute_queue}, {!Copy_queue})
    that translate work into the method streams the compute and copy
    engines execute.

    The builders are pure: they read a {!type-device} description,
    append dwords to an in-memory {!Hcq.Q.t}, and patch launch
    descriptors through their CPU mappings. Their [submit] functions
    stage the accumulated stream in the device's command buffer, point
    a channel ring entry ({!Queue_desc}) at it, and ring the
    work-submission doorbell. *)

module Hcq = Tolk_hcq.Hcq
module Nv_tables = Nv_tables

(** {1:qmd Launch descriptors} *)

(** Kernel launch descriptors.

    The compute engine consumes launches as fixed-size descriptors of
    packed bitfields: grid and block geometry, program and constant
    buffer addresses, dependency links, and release semaphores. A
    descriptor is a lens over caller-provided mapped bytes, either a
    program's long-lived template or the per-launch copy the engine
    fetches from device memory; fields are addressed by the names of
    the generated tables ({!Nv_tables.Defs}). *)
module Qmd : sig
  type t
  (** The type for launch descriptors. *)

  val sizeof : compute_class:int -> int
  (** [sizeof ~compute_class] is the descriptor size in bytes for the
      layout [compute_class] consumes: [0x100] before Blackwell,
      [0x180] from Blackwell on. *)

  val create : view:Hcq.Mmio.t -> compute_class:int -> t
  (** [create ~view ~compute_class] is the descriptor stored in the
      first [sizeof ~compute_class] bytes of [view], read and written
      in place. Raises [Invalid_argument] if [view] is smaller than the
      descriptor. *)

  val version : t -> int
  (** [version t] is the descriptor layout version: [3] before
      Blackwell, [5] from Blackwell on. *)

  val read : t -> string -> int
  (** [read t name] is the value of field [name]. Field names are
      case-insensitive; per-slot fields carry their slot index as a
      suffix (for example ["release0_enable"]). Raises
      [Invalid_argument] on an unknown name. *)

  val write : t -> (string * int) list -> unit
  (** [write t fields] sets each named field to its value, in list
      order. Raises [Invalid_argument] on an unknown name or a value
      that does not fit the field's width. *)

  val field_offset : t -> string -> int
  (** [field_offset t name] is the byte offset of the first byte of
      field [name] within the descriptor. Raises [Invalid_argument] on
      an unknown name. *)

  val set_constant_buf_addr : t -> int -> nativeint -> unit
  (** [set_constant_buf_addr t i addr] binds constant buffer slot [i]
      to device address [addr]. Version 5 descriptors store the
      address shifted right by 6, so [addr] must be 64-byte aligned
      there. *)

  val to_bytes : t -> bytes
  (** [to_bytes t] is a copy of the descriptor's bytes. *)
end

(** {1:devices Devices} *)

type 'meta device = {
  compute_class : int;
      (** Compute engine class id; selects the descriptor layout. *)
  dma_class : int;  (** Copy engine class id. *)
  gpfifo_class : int;  (** Channel class id. *)
  sass_version : int;  (** Shader ISA version of the chip. *)
  mutable slm_per_thread : int;
      (** Per-thread local-memory bytes the device is currently sized
          for; starts at [0]. *)
  shared_mem_window : nativeint;
      (** Virtual-address window shared-memory accesses go through. *)
  local_mem_window : nativeint;
      (** Virtual-address window local-memory accesses go through. *)
  cmdq_page : 'meta Hcq.Buffer.t;
      (** CPU-mapped device memory where submissions stage their
          command streams. *)
  cmdq_allocator : Tolk.Bump.t;
      (** Wrapping bump allocator handing out stream addresses inside
          [cmdq_page]. *)
  cmdq : Hcq.Mmio.t;  (** CPU view of [cmdq_page]. *)
  gpu_mmio : Hcq.Mmio.t;
      (** Usermode register region carrying the work-submission
          doorbell. *)
}
(** The device description the queue builders read. ['meta] is the
    driver metadata carried by the device's buffers. *)

val device :
  compute_class:int ->
  dma_class:int ->
  gpfifo_class:int ->
  sass_version:int ->
  ?slm_per_thread:int ->
  shared_mem_window:nativeint ->
  local_mem_window:nativeint ->
  cmdq_page:'meta Hcq.Buffer.t ->
  gpu_mmio:Hcq.Mmio.t ->
  unit ->
  'meta device
(** [device ~compute_class ~dma_class ~gpfifo_class ~sass_version
    ~shared_mem_window ~local_mem_window ~cmdq_page ~gpu_mmio ()] is a
    device description over the given engine classes and mappings. The
    command-stream allocator wraps over [cmdq_page], whose CPU view
    must exist. [slm_per_thread] defaults to [0].

    Raises [Invalid_argument] if [cmdq_page] has no CPU view. *)

(** {1:programs Programs} *)

type 'meta program = {
  dev : 'meta device;  (** Device the program was loaded on. *)
  qmd : Qmd.t;
      (** Launch descriptor template: the geometry-independent fields,
          filled at load time. {!Compute_queue.exec} copies it behind
          the staged arguments and patches the per-launch fields into
          the copy. *)
  cbuf0_size : int;
      (** Size of the kernel's first constant buffer in bytes. Kernel
          arguments are staged in it, and the descriptor copy lands at
          the next 256-byte boundary after it. *)
}
(** The launch parameters of a loaded kernel. *)

(** {1:queue_desc Mapped queues} *)

(** Hardware channels mapped into the process.

    A descriptor bundles what a submission needs: the channel's entry
    ring, its put pointer, and the token that identifies the channel
    to the work-submission doorbell. Tests may build descriptors over
    any mapped memory. *)
module Queue_desc : sig
  type t = {
    ring : Hcq.Mmio.t;
        (** The channel's entry ring: 64-bit entries, each pointing at
            a staged command stream. *)
    gpput : Hcq.Mmio.t;
        (** 32-bit producer position, published after each entry. *)
    token : int;
        (** Work-submission token naming the channel to the
            doorbell. *)
    mutable put_value : int;  (** Number of entries submitted. *)
  }
  (** The type for mapped channels. *)
end

(** {1:queues Queue builders} *)

(** Compute-engine command streams.

    Each function appends one logical command to the queue's dword
    stream; {!Compute_queue.q} exposes the accumulated stream for
    submission. Values that do not fit their 32-bit dword raise
    [Invalid_argument] (see {!Hcq.Q.push}).

    Successive launches coalesce: while a launch is pending, the next
    {!Compute_queue.exec} links itself into the pending descriptor as
    its dependent instead of appending stream methods, and
    {!Compute_queue.signal} rides in a free release slot of the
    pending descriptor instead of appending a semaphore method.
    {!Compute_queue.wait}, {!Compute_queue.write},
    {!Compute_queue.poll_bit} and {!Compute_queue.memory_barrier} end
    the pending launch, so later commands go back to the stream. *)
module Compute_queue : sig
  type 'meta t
  (** The type for compute command streams under construction. *)

  val create : 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  val setup :
    'meta t ->
    ?compute_class:int ->
    ?local_mem_window:nativeint ->
    ?shared_mem_window:nativeint ->
    ?local_mem:nativeint ->
    ?local_mem_tpc_bytes:int ->
    unit ->
    unit
  (** [setup t ()] appends the engine set-up methods for each argument
      given: bind the compute class to the channel, set the two
      virtual-address windows, and point the engine at the
      local-memory backing store and its per-TPC size. *)

  val exec :
    'meta t ->
    'meta program ->
    kernargs:'a Hcq.Buffer.t ->
    global_size:int * int * int ->
    local_size:int * int * int ->
    unit
  (** [exec t prg ~kernargs ~global_size ~local_size] launches [prg]
      over a [global_size] grid of [local_size] blocks, with the
      kernel arguments staged at the start of [kernargs]. The launch
      descriptor is copied into [kernargs] at the 256-byte boundary
      after the argument bytes ([prg.cbuf0_size]) and its geometry and
      constant-buffer-0 address are patched into the copy, so
      [kernargs] must be CPU-mapped and have room for the descriptor.

      Raises [Invalid_argument] if the descriptor's device address
      does not fit in 40 bits, or if a dimension does not fit its
      field (32 bits per grid dimension, 16 bits for the first two
      block dimensions, 8 bits for the third). *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] writes [value] (defaults to [0]) to [sg]'s value
      slot once all prior work retired. After a launch, the release is
      carried by the launch descriptor when one of its two release
      slots is free; otherwise a semaphore-release method is appended,
      which also stamps [sg]'s timestamp and raises a non-stalling
      interrupt. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the channel until [sg]'s value reaches
      [value] (defaults to [0]), comparing 64-bit values with
      wrap-around. *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] releases [sg] with value [0], stamping its
      timestamp slot. *)

  val write : 'meta t -> ?b64:bool -> 'a Hcq.Buffer.t -> int64 -> unit
  (** [write t buf v] writes [v] to the start of [buf] once all prior
      work retired: the full 64 bits when [b64] is [true], the low 32
      otherwise (defaults to [false]). *)

  val poll_bit : 'meta t -> 'a Hcq.Buffer.t -> value:int -> mask:int -> unit
  (** [poll_bit t buf ~value ~mask] stalls the channel until the bits
      selected by [mask] in the first dword of [buf] are all set
      ([value = mask]) or all clear ([value = 0]). *)

  val memory_barrier : 'meta t -> unit
  (** [memory_barrier t] invalidates the engine's instruction, global
      data, and constant caches, making prior memory writes visible to
      subsequent launches. *)

  val submit : 'meta t -> Queue_desc.t -> unit
  (** [submit t qd] stages the accumulated stream in the device's
      command buffer, writes the next ring entry of [qd] to point at
      it, publishes the new put position, fences, and rings the
      work-submission doorbell with [qd]'s token. The stream is kept:
      submitting again stages it again. *)
end

(** Copy-engine command streams.

    Each function appends one logical command to the queue's dword
    stream; {!Copy_queue.q} exposes the accumulated stream for
    submission. *)
module Copy_queue : sig
  type 'meta t
  (** The type for copy command streams under construction. *)

  val create : 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  val setup : 'meta t -> ?copy_class:int -> unit -> unit
  (** [setup t ()] binds [copy_class], when given, to the channel. *)

  val copy :
    'meta t -> dest:'a Hcq.Buffer.t -> src:'b Hcq.Buffer.t -> int -> unit
  (** [copy t ~dest ~src size] copies [size] bytes from the start of
      [src] to the start of [dest], split into transfers of at most
      2 GiB. *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] writes [value] (defaults to [0]) to [sg]'s value
      slot once prior transfers completed, stamping its timestamp
      slot. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the channel until [sg]'s value reaches
      [value] (defaults to [0]), comparing 64-bit values with
      wrap-around. *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] releases [sg] with value [0], stamping its
      timestamp slot. *)

  val submit : 'meta t -> Queue_desc.t -> unit
  (** [submit t qd] stages the accumulated stream and rings the
      doorbell, exactly as {!Compute_queue.submit}. *)
end
