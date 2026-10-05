(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPU command packets.

    The words a GPU's queues read: PM4 packets on a compute queue ({!Pm4}), AQL
    packets on a queue that takes them ({!Aql}), and SDMA packets on a copy
    engine ({!Sdma}); and the GC registers that PM4 packets write ({!Gc}).

    A packet is constant words around the caller's values, of any type ['v]:
    integers for a runtime that writes its queue at once, or values that a
    compiler computes later. Where a packet's layout computes on a value, such
    as an address a register holds from its bit 8, or the offset of each piece
    of a copy, the word is a {!term}, which the consumer evaluates: {!dwords}
    over integers, a compiler as nodes. *)

(** {1:words Terms and words} *)

(** The type for computations on a value, as 64-bit unsigned integers.

    A term lists each operation a layout applies, in the order it applies them:
    an address offset for a piece of a copy is an [Add] node, an address shifted
    for its field a [Shift] node. An interpreter applies every node as it
    stands, never omitting one or merging two, so that what it builds of a term
    mirrors the term. *)
type 'v term =
  | Value of 'v  (** The value. *)
  | Add of 'v term * int64  (** [Add (t, n)] is [t + n], modulo [2{^64}]. *)
  | Shift of 'v term * int  (** [Shift (t, n)] is [t] shifted right by [n]. *)
  | Or of 'v term * int64  (** [Or (t, n)] is the bitwise or of [t] and [n]. *)

val eval : int term -> int64
(** [eval t] is the 64-bit unsigned integer [t] computes, its values taken as
    non-negative integers. *)

(** The type for the words of a packet. *)
type 'v word =
  | Dword of int  (** A word known when encoding: the integer's low 32 bits. *)
  | W32 of 'v term  (** A term's low 32 bits. *)
  | W64 of 'v term  (** A term's 64 bits, as two words, low first. *)

val dwords : int word list -> int list
(** [dwords ws] is the 32-bit words of [ws]. *)

type version = int * int * int
(** The type for the versions of a GPU's IP blocks, as its discovery table gives
    them: [(9, 4, 3)] for the GC of an MI300. *)

(** The type for the comparisons of a wait: the value read is equal to, or at
    least, the reference. *)
type comparison = Equal | Greater_equal

(** The type for what a packet reads or writes: a register, at its dword address
    ({!Gc.address}), or memory, at the caller's address. *)
type 'v location = Register of int | Memory of 'v

(** {1:pm4 PM4} *)

(** PM4 packets: the commands of a compute queue. *)
module Pm4 : sig
  val set_reg : int -> 'v word list -> 'v word list
  (** [set_reg reg ws] writes [ws] to the consecutive registers from the one at
      address [reg]: an SH register's packet, or a UCONFIG register's.

      Raises [Invalid_argument] if [reg] is in neither range. *)

  val set_program : gc:version -> 'v -> 'v word list
  (** [set_program ~gc addr] points [COMPUTE_PGM_LO] and [_HI] at the code at
      [addr], 256-byte aligned, which they hold from its bit 8. *)

  val set_scratch : gc:version -> 'v -> 'v word list
  (** [set_scratch ~gc addr] points [COMPUTE_DISPATCH_SCRATCH_BASE_LO] and [_HI]
      at the scratch memory at [addr], 256-byte aligned, which they hold from
      its bit 8. *)

  val wait :
    gc:version ->
    'v location ->
    comparison ->
    'v ->
    mask:int ->
    interval:int ->
    'v word list
  (** [wait ~gc loc cmp v ~mask ~interval] waits until the 32 bits at [loc],
      masked by [mask], compare to the low 32 bits of [v] as [cmp] says, reading
      them again after [interval], the packet's poll interval. *)

  (** The type for the caches an acquire invalidates: the data caches (scalar,
      vector and the L1s), or all of them, the instruction cache and the L2
      (written back, then invalidated) too. *)
  type caches = Data_caches | All_caches

  val acquire_mem : gc:version -> caches -> 'v word list
  (** [acquire_mem ~gc caches] invalidates [caches] of all memory for the work
      after it. *)

  (** The type for the signal a release writes: the low 32 bits of a value, or
      all 64. *)
  type 'v data = Low_32 of 'v | Data_64 of 'v

  val release_mem : gc:version -> 'v -> 'v data -> 'v word list
  (** [release_mem ~gc addr d] writes the signal [d] to memory at [addr] once
      the work before it is complete, at the end of the pipe: it first writes
      back and invalidates the GPU's caches, so that the work's writes are
      visible to whoever reads the signal, and raises an interrupt once it is
      written. *)

  val pred_exec : xcc_mask:int -> dwords:int -> 'v word list
  (** [pred_exec ~xcc_mask ~dwords] has the [dwords] words after it run only on
      the dies of [xcc_mask], bit [i] for die [i]. *)

  (** The type for the events a queue signals. *)
  type event =
    | Cs_partial_flush  (** Waits for the dispatches before it to complete. *)
    | Thread_trace_marker  (** Marks the thread trace. *)
    | Thread_trace_finish  (** Finishes the thread trace. *)

  val event_write : event -> 'v word list
  (** [event_write e] signals [e]. *)

  (** The type for how a write completes: posted, or confirmed before the
      packets after it run. *)
  type write = Posted | Confirmed

  (** The type for what a copy reads: the 32 bits of a counter register, at its
      address, or the 64 bits of the GPU's clock counter. *)
  type source = Counter of int | Clock

  val copy_data : write -> source -> 'v -> 'v word list
  (** [copy_data w src addr] copies [src] to memory at [addr] through the L2, as
      [w] says, as the queue reaches the packet: the work before it has reached
      the queue's end, and the work after it has not started. *)

  val write_data : 'v location -> 'v -> 'v word list
  (** [write_data loc v] writes the low 32 bits of [v] to [loc], waiting for a
      write to memory to be confirmed. *)

  val indirect_buffer : 'v -> dwords:int -> 'v word list
  (** [indirect_buffer addr ~dwords] runs the [dwords] words of PM4 packets at
      [addr]. *)

  (** The type for the lanes of a wave. *)
  type wave = Wave32 | Wave64

  val dispatch_direct : gc:version -> wave -> 'v * 'v * 'v -> 'v word list
  (** [dispatch_direct ~gc wave (x, y, z)] dispatches the kernel the [COMPUTE_*]
      registers describe on a grid of [x * y * z] workgroups, in waves of
      [wave]'s lanes. *)

  val dispatch :
    gc:version ->
    Nx_amd_code_object.kernel ->
    program:'v ->
    scratch:'v ->
    packet:'v ->
    args:'v ->
    tmpring:int ->
    limits:int ->
    threads:'v * 'v * 'v ->
    groups:'v * 'v * 'v ->
    'v word list
  (** [dispatch ~gc k ~program ~scratch ~packet ~args ~tmpring ~limits ~threads
       ~groups] runs the kernel [k], whose first instruction is at [program], on
      a single die. It sets the [COMPUTE_*] registers from [k]'s descriptor: its
      resource words, with the privilege GFX11 runs kernels with and the LDS its
      workgroups take, and the user SGPRs it enables, in their order: a buffer
      descriptor of the scratch memory at [scratch], the address [packet] of its
      dispatch packet, and the address [args] of its arguments. It also sets the
      scratch ring's word [tmpring] ({!Gc.tmpring_size}) and its memory at
      [scratch] ({!set_scratch}), at most [limits] waves on each shader array,
      [0] for no limit, and workgroups of [threads] work-items, then dispatches
      a grid of [groups] workgroups in waves of [k]'s lanes
      ({!dispatch_direct}). [packet] is read only when [k] reads its dispatch
      packet. *)
end

(** {1:aql AQL} *)

(** AQL packets: the 64-byte packets of a queue that takes them. *)
module Aql : sig
  val dispatch :
    threads:int * int * int ->
    grid:'v * 'v * 'v ->
    private_segment:int ->
    group_segment:int ->
    descriptor:'v ->
    args:'v ->
    'v word list
  (** [dispatch ~threads ~grid ~private_segment ~group_segment ~descriptor
       ~args] dispatches the kernel whose descriptor is at [descriptor] on
      arguments at [args], over a grid of [grid] work-items in workgroups of
      [threads] work-items, each work-item taking [private_segment] bytes of
      scratch and each workgroup [group_segment] bytes of LDS. It waits for the
      packets before it and makes memory coherent across the system before and
      after the kernel. *)

  val indirect_buffer : 'v -> dwords:int -> 'v word list
  (** [indirect_buffer addr ~dwords] runs the [dwords] words of PM4 packets at
      [addr], ordered as {!dispatch} is. *)
end

(** {1:sdma SDMA} *)

(** SDMA packets: the commands of a copy engine. *)
module Sdma : sig
  val copy : sdma:version -> dst:'v -> src:'v -> int -> 'v word list
  (** [copy ~sdma ~dst ~src n] copies [n] bytes from [src] to [dst], in linear
      copies of at most what an engine of version [sdma] moves at once: 1 GiB
      from version 4.4.2 below 5 and from 5.2, 4 MiB otherwise. It is [[]] for
      [n = 0].

      Raises [Invalid_argument] if [n < 0]. *)

  val poll : 'v -> comparison -> 'v -> mask:int -> 'v word list
  (** [poll addr cmp v ~mask] waits until the 32 bits at [addr], masked by
      [mask], compare to the low 32 bits of [v] as [cmp] says. *)

  val fence : sdma:version -> 'v -> 'v -> 'v word list
  (** [fence ~sdma addr v] writes the low 32 bits of [v] to [addr] once the
      packets before it are complete, uncached on engines that take a memory
      type (version 5 on). *)

  val trap : 'v word list
  (** [trap] raises an interrupt. *)

  val timestamp : 'v -> 'v word list
  (** [timestamp addr] writes the GPU's global timestamp, 64 bits, to [addr]. *)
end

(** {1:gc GC registers} *)

(** The registers of the GC, the block that runs a compute queue. *)
module Gc : sig
  type register = {
    name : string;
        (** Its name in the GC's headers, as ["regGRBM_GFX_INDEX"]. *)
    offset : int;  (** Its dword offset in its segment. *)
    segment : int;  (** Its segment. *)
    fields : (string * (int * int)) list;
        (** Its fields by name, each as its lowest and highest bit. *)
  }
  (** The type for registers. *)

  val family : version -> version option
  (** [family gc] is the version whose registers a GC of version [gc] has: the
      latest of its major version at or before it, if any. *)

  val registers : version -> register list
  (** [registers gc] is the registers of a GC of version [gc], of its {!family}:
      those a compute queue's commands and a runtime that brings the GPU up read
      and write. It is [[]] for a GC of no family. *)

  val find : version -> string -> register option
  (** [find gc name] is the register [name] of {!registers}[ gc], if any. *)

  val address : version -> register -> int
  (** [address gc r] is [r]'s dword address in the register space of a GC of
      version [gc], which {!Pm4.set_reg} takes: its offset from the base its
      generation gives its segment. *)

  val encode : register -> (string * int) list -> int
  (** [encode r fs] is the word of [r] with each field of [fs] set to its value,
      cut to the field's width, and its other bits zero.

      Raises [Invalid_argument] if [r] has no field of one of [fs]. *)

  val tmpring_size :
    gc:version ->
    compute_units:int ->
    slots:int ->
    shader_engines:int ->
    xccs:int ->
    int ->
    int
  (** [tmpring_size ~gc ~compute_units ~slots ~shader_engines ~xccs n] is the
      word of [COMPUTE_TMPRING_SIZE] for kernels of [n] bytes of scratch per
      lane, on a GPU of [xccs] dies, each of [shader_engines] shader engines and
      [compute_units] compute units of [slots] scratch wave slots. A lane takes
      at least 128 bytes, rounded up to the generation's alignment (1024 bytes a
      wave on GFX9, 256 after); the word holds the size of a wave's scratch in
      units of that alignment, and the waves one die's scratch serves, divided
      among its shader engines after GFX9, and at most every slot's.

      Raises [Invalid_argument] if a GC of version [gc] has no such register. *)

  (** The values a thread trace sets in the fields of its registers, as the SOC
      enumerations define them. *)
  module Thread_trace : sig
    val rt_freq_4096_clk : int
    (** [SQ_THREAD_TRACE_CTRL.rt_freq]: a time token every 4096 clocks. *)

    val wtype_include_cs_bit : int
    (** [SQ_THREAD_TRACE_MASK.wtype_include]'s bit of compute waves. *)

    val token_mask_sqdec_bit : int
    (** [SQ_THREAD_TRACE_TOKEN_MASK.reg_include]'s bit of the SQ's registers. *)

    val token_mask_shdec_bit : int
    (** [reg_include]'s bit of the shader registers. *)

    val token_mask_gfxudec_bit : int
    (** [reg_include]'s bit of the graphics user-config registers. *)

    val token_mask_comp_bit : int
    (** [reg_include]'s bit of the compute registers. *)

    val token_mask_context_bit : int
    (** [reg_include]'s bit of the context registers. *)

    val token_exclude_vmemexec_shift : int
    (** [SQ_THREAD_TRACE_TOKEN_MASK.token_exclude]'s bit of vector memory
        execution tokens. *)

    val token_exclude_aluexec_shift : int
    (** [token_exclude]'s bit of ALU execution tokens. *)

    val token_exclude_valuinst_shift : int
    (** [token_exclude]'s bit of vector ALU instruction tokens. *)

    val token_exclude_immediate_shift : int
    (** [token_exclude]'s bit of immediate tokens. *)

    val token_exclude_inst_shift : int
    (** [token_exclude]'s bit of instruction tokens. *)
  end
end
