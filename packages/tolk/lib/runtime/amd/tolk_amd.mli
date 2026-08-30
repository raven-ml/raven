(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** AMD GPU runtime.

    Building blocks for driving AMD GPUs through their hardware command
    queues: generic queue machinery ({!Hcq}), kernel compilation
    ({!Compiler_amd}), hardware tables ({!Amd_tables}), and the
    command-stream builders ({!Compute_queue}, {!Copy_queue}) that
    translate work into the packet formats the compute and DMA engines
    execute.

    The builders are pure: they read a {!type-device} description and
    append dwords to an in-memory {!Hcq.Q.t}. Mapping queues, submitting
    streams, and synchronizing with the hardware happen elsewhere. *)

module Hcq = Hcq
module Compiler_amd = Compiler_amd
module Amd_tables = Amd_tables

(** {1:devices Devices} *)

type queue_event = { event_id : int }
(** An interrupt event registered with the driver. [event_id] names the
    event slot a packet can fire to wake waiters. *)

type 'meta device = {
  target : int * int * int;  (** Target graphics version, e.g. [(11, 0, 0)]. *)
  xccs : int;  (** Number of accelerated-compute dies; 1 on consumer chips. *)
  soc : (module Amd_tables.Soc);  (** Event ids for the generation. *)
  pm4 : (module Amd_tables.Pm4);  (** Compute-packet constants. *)
  sdma : (module Amd_tables.Sdma);  (** DMA-packet constants. *)
  gc : Amd_tables.Ip.t;  (** Graphics-core register family. *)
  nbio : Amd_tables.Ip.t;  (** Bus-interface register family. *)
  max_copy_size : int;
      (** Largest byte count a single DMA copy packet can move. *)
  sqtt_enabled : bool;  (** Thread-trace capture; not supported yet. *)
  mutable tmpring_size : int;
      (** Scratch-ring size register value, written verbatim on launch. *)
  mutable scratch : 'meta Hcq.Buffer.t;
      (** Backing store for kernel scratch (private segments). *)
  is_am : bool;
      (** [true] when this process drives the GPU directly rather than
          through the kernel driver; such devices have no queue event. *)
  queue_event_mailbox_ptr : nativeint;
      (** Address the driver polls for the queue event's payload. *)
  queue_event : queue_event;
      (** Event fired to signal completion interrupts. *)
}
(** The device description the queue builders read. ['meta] is the
    driver metadata carried by the device's buffers. *)

val device :
  target:int * int * int ->
  xccs:int ->
  gc_version:int * int * int ->
  nbio_version:int * int * int ->
  sdma_version:int * int * int ->
  ?sqtt_enabled:bool ->
  tmpring_size:int ->
  scratch:'meta Hcq.Buffer.t ->
  is_am:bool ->
  queue_event_mailbox_ptr:nativeint ->
  queue_event:queue_event ->
  unit ->
  'meta device
(** [device ~target ~xccs ~gc_version ~nbio_version ~sdma_version ...]
    resolves the hardware tables for a chip: event ids and packet
    constants from [target], register families from the discovered
    [gc_version] and [nbio_version] (the [nbif] family on generation 12
    and later), the DMA packet format from [sdma_version], and
    [max_copy_size] ([0x40000000] for DMA engines of major version 5 and
    later, [0x400000] before). [sqtt_enabled] defaults to [false].

    Raises [Invalid_argument] when a version has no table. *)

type 'meta program = {
  dev : 'meta device;  (** Device the program was loaded on. *)
  prog_addr : nativeint;  (** Machine-code address, 256-byte aligned. *)
  rsrc1 : int;  (** COMPUTE_PGM_RSRC1 register value. *)
  rsrc2 : int;  (** COMPUTE_PGM_RSRC2 register value. *)
  rsrc3 : int;  (** COMPUTE_PGM_RSRC3 register value. *)
  wave32 : bool;  (** Dispatch in wave32 mode (generation 10 and later). *)
  enable_private_segment_sgpr : bool;
      (** The kernel expects a flat-scratch descriptor in its first user
          registers. *)
  enable_dispatch_ptr : bool;
      (** The kernel expects a dispatch-packet pointer; not supported
          yet. *)
}
(** The launch parameters of a loaded kernel. *)

(** {1:queues Queue builders} *)

(** Compute-engine command streams.

    Each function appends one logical command to the queue's dword
    stream; {!Compute_queue.q} exposes the accumulated stream for
    submission. Values that do not fit their 32-bit dword raise
    [Invalid_argument] (see {!Hcq.Q.push}). *)
module Compute_queue : sig
  type 'meta t
  (** The type for compute command streams under construction. *)

  val create : 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  (** {2:commands Commands} *)

  val exec :
    'meta t ->
    'meta program ->
    kernargs:'a Hcq.Buffer.t ->
    global_size:int * int * int ->
    local_size:int * int * int ->
    unit
  (** [exec t prg ~kernargs ~global_size ~local_size] launches [prg]
      over a [global_size] grid of [local_size] workgroups, with the
      kernel arguments staged at [kernargs]. The launch invalidates
      stale caches first and drains the pipeline afterwards, so
      successive launches see each other's writes.

      Raises [Invalid_argument] if [prg] wants a dispatch pointer or
      thread-trace capture (neither is supported), or if it wants a
      private-segment descriptor on a multi-die device. *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] flushes caches and writes [value] (defaults to [0])
      to [sg]'s value slot once all prior work retired. For a timeline
      signal owned by a driver-managed device, also fires the owner's
      queue event so blocked waiters wake up. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the queue until [sg]'s value reaches [value]
      (defaults to [0]). *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] records the GPU clock counter in [sg]'s
      timestamp slot once all prior work retired. *)

  val write : 'meta t -> ?b64:bool -> 'a Hcq.Buffer.t -> int64 -> unit
  (** [write t buf v] writes [v] to the start of [buf] once all prior
      work retired: the full 64 bits when [b64] is [true], the low 32
      otherwise (defaults to [false]). *)

  val poll_bit : 'meta t -> 'a Hcq.Buffer.t -> value:int -> mask:int -> unit
  (** [poll_bit t buf ~value ~mask] stalls the queue until the first
      dword of [buf], masked with [mask], equals [value]. *)

  val memory_barrier : 'meta t -> unit
  (** [memory_barrier t] flushes the host-data-path caches and
      invalidates every GPU cache, making host writes visible to
      subsequent commands. *)

  (** {2:packets Packet-level interface}

      Building blocks for commands not covered above. *)

  val pkt3 : 'meta t -> int -> int array -> unit
  (** [pkt3 t op payload] appends a type-3 packet: the header for [op]
      sized to [payload], then [payload] itself. *)

  val wreg : 'meta t -> Amd_tables.Reg.t -> int array -> unit
  (** [wreg t reg vals] sets [vals] into the register file starting at
      [reg]: consecutive dwords land in consecutive registers. The
      packet type follows [reg]'s range (shader or universal config).
      Raises [Invalid_argument] for a register outside both ranges. *)

  val wreg_fields : 'meta t -> Amd_tables.Reg.t -> (string * int) list -> unit
  (** [wreg_fields t reg fields] is {!wreg} with the value assembled
      from named fields via {!Amd_tables.Reg.encode}. *)

  val pred_exec : 'meta t -> xcc_mask:int -> (unit -> unit) -> unit
  (** [pred_exec t ~xcc_mask f] runs [f] with the commands it emits
      predicated to the dies selected by [xcc_mask]. On single-die
      devices the commands are emitted unpredicated. *)

  val acquire_mem :
    'meta t ->
    ?addr:nativeint ->
    ?sz:int64 ->
    ?gli:int ->
    ?glm:int ->
    ?glk:int ->
    ?glv:int ->
    ?gl1:int ->
    ?gl2:int ->
    unit ->
    unit
  (** [acquire_mem t ()] stalls until prior work retired and invalidates
      the selected caches for the [sz] bytes at [addr] (defaults: the
      whole address space, every cache). Each [gl*] flag selects one
      cache level; pass [0] to leave it untouched. *)

  val release_mem :
    'meta t ->
    ?address:nativeint ->
    ?value:int64 ->
    ?data_sel:int ->
    ?int_sel:int ->
    ?ctxid:int ->
    ?cache_flush:bool ->
    unit ->
    unit
  (** [release_mem t ()] appends an end-of-pipe event: once prior work
      retired, write the datum selected by [data_sel] ([value], or the
      GPU clock) to [address] and raise the interrupt selected by
      [int_sel], optionally flushing caches first. *)

  val wait_reg_mem :
    'meta t ->
    ?mask:int ->
    ?mem:nativeint ->
    ?reg:int ->
    ?reg_done:int ->
    ?op:int ->
    int ->
    unit
  (** [wait_reg_mem t value] stalls the queue until a masked location
      compares against [value] under [op] (defaults to
      {!wait_reg_mem_function_geq}): the dword at address [mem] when
      given, the register [reg] otherwise. Without [mem], a non-zero
      [reg_done] register is written back when the wait completes. *)

  val wait_reg_mem_function_eq : int
  (** Comparison: masked location equals the reference value. *)

  val wait_reg_mem_function_geq : int
  (** Comparison: masked location is at least the reference value. *)
end

(** DMA-engine command streams.

    Each function appends one packet to the queue's dword stream;
    {!Copy_queue.q} exposes the accumulated stream and
    {!Copy_queue.cmd_sizes} its packet boundaries, which submission
    needs to split the stream across a ring's wrap point. *)
module Copy_queue : sig
  type 'meta t
  (** The type for DMA command streams under construction. *)

  val create : ?max_copy_size:int -> 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. [max_copy_size] caps
      the bytes per copy packet and defaults to the device's. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  val cmd_sizes : 'meta t -> int list
  (** [cmd_sizes t] is the dword count of each packet, in stream
      order. *)

  val copy :
    'meta t -> dest:'a Hcq.Buffer.t -> src:'b Hcq.Buffer.t -> int -> unit
  (** [copy t ~dest ~src size] copies [size] bytes from the start of
      [src] to the start of [dest], split into as many packets as the
      queue's [max_copy_size] requires. *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] writes [value] (defaults to [0]) to [sg]'s value
      slot once prior packets completed. For a timeline signal owned by
      a driver-managed device, also writes the owner's event mailbox and
      fires its queue event. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the engine until [sg]'s value reaches [value]
      (defaults to [0]). *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] records the global engine clock in [sg]'s
      timestamp slot. *)

  val write : 'meta t -> ?b64:bool -> 'a Hcq.Buffer.t -> int64 -> unit
  (** [write t buf v] writes [v] to the start of [buf]: the full 64 bits
      when [b64] is [true], the low 32 otherwise (defaults to
      [false]). *)
end
