(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AQL packets: the commands of a queue that dispatches over every die.

    An AQL packet is 16 words, 64 bytes, that never straddle the ring's end. Its
    first word holds its header, which names its type: the queue may read a
    packet as soon as that word is written, before the doorbell. A writer writes
    words 1 to 15 first, then the first word in one 32-bit store with release
    order. Every packet here waits for the packets before it to complete (the
    header's barrier bit) and makes memory coherent across the system before and
    after it. *)

val dispatch :
  Code_object.kernel ->
  descriptor:'v ->
  args:'v ->
  threads:int * int * int ->
  grid:'v * 'v * 'v ->
  'v Packet.t
(** [dispatch k ~descriptor ~args ~threads ~grid] launches the kernel [k], whose
    descriptor is at address [descriptor], on the arguments at address [args],
    over a grid of [grid] work-items in workgroups of [threads] work-items: a
    kernel dispatch packet with [k]'s private and group segment sizes and no
    completion signal.

    It is also the dispatch packet a kernel that reads one ({!Pm4.dispatch})
    finds in memory.

    Raises [Invalid_argument] if a component of [threads] is not in
    \[[1];[65535]\]. *)

(** The type for the fields of a kernel dispatch packet that its writer may
    set after encoding it ({!dispatch}): a launch's workgroup sizes, and the
    LDS its workgroups take. *)
type field =
  | Workgroup_size of Code_object.axis
      (** 16 bits: the work-items of a workgroup along the axis. *)
  | Group_segment_size  (** 32 bits: the bytes of LDS a workgroup takes. *)

val offset : field -> int
(** [offset f] is the byte offset of [f] in a kernel dispatch packet, the 64
    bytes of hsa.h's [hsa_kernel_dispatch_packet_t], little-endian. *)

val indirect_buffer : 'v -> dwords:'v -> 'v Packet.t
(** [indirect_buffer addr ~dwords] runs the [dwords] words of PM4 packets at
    [addr] ({!Pm4}), in a packet of the vendor's format. Every die the queue
    runs on runs them; {!Pm4.pred_exec} keeps words to some dies. [addr] is a
    multiple of 4 and [dwords] is in \[[0];[1048575]\]. *)
