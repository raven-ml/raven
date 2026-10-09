(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ConnectX NICs' formats, as values.

    What a process and a ConnectX NIC (the [mlx5] family) share in memory: the
    work entries a queue pair's send ring holds, the completion entries a
    completion queue's ring holds, the doorbell records that count both, and the
    doorbell registers of the NIC's access region, whose stores tell the NIC
    that work waits. Nothing here acts on a NIC. Each value describes bytes, and
    its caller writes or reads them.

    {v
     send ring: n entries of 64 bytes        doorbell record    access region
     | entry i | entry i+1 | ...   --post-->  send count  --ring-->  register
                                                                       |
     completion ring: m entries of 64 bytes  <--- the NIC writes ------'
     | completion j | ...           --poll-->  consumed count
    v}

    A process writes entries at its ring's producer count, stores the count in
    the queue pair's doorbell record ({!Doorbell}), then stores the first 8
    bytes of the last entry in the queue pair's doorbell register ({!Uar}). The
    NIC runs the entries in order and writes a completion for each entry that
    asks for one, and for every entry that fails. The process reads a completion
    once {!Completion.owned} says the NIC wrote it, and stores how many it read
    in the completion queue's doorbell record.

    {1:conventions Conventions}

    Every field the NIC reads or writes is big-endian, and the encoders and
    decoders here read and write it so. Data carried inline is copied as given.

    Misuse raises [Invalid_argument]: an integer outside the range its function
    states.

    The library has no state: any domain may use any of its values.

    {1:references References}

    - rdma-core v56.0's [providers/mlx5]: [mlx5dv.h] (work entries, completions,
      doorbell records), [mlx5.h] (the access region's doorbell offset), [qp.c]
      (posting and ringing) and [cq.c] (polling and arming).
    - Linux v6.12's [include/uapi/rdma/mlx5-abi.h] (the access region's mapping
      commands), [include/linux/mlx5/device.h] (doorbell registers per access
      region) and [drivers/infiniband/hw/mlx5/main.c] (the mapping of access
      region pages). *)

type buffer =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for memory a process and a NIC share, such as a ring. *)

(** Work entries: the operations of a queue pair's send ring.

    Every entry here takes one basic block of {!size} bytes, so that no entry
    wraps around the ring's end: entry [i] of a ring of [n] entries is at byte
    [64 * (i mod n)]. *)
module Entry : sig
  val size : int
  (** [size] is [64], the bytes of a basic block. *)

  type local = {
    address : int;  (** The address the NIC knows the first byte by. *)
    bytes : int;  (** The length, in bytes. *)
    key : int;  (** The local key of the region that holds it. *)
  }
  (** The type for memory this process registered with its NIC. *)

  type remote = {
    address : int;  (** The address the peer's NIC knows the first byte by. *)
    key : int;  (** The remote key of the peer's region that holds it. *)
  }
  (** The type for memory a peer registered with its NIC. *)

  (** The type for operations. *)
  type op =
    | Write of { src : local; dst : remote }
        (** [Write {src; dst}] copies [src.bytes] bytes from [src] to [dst]. *)
    | Write_inline of { data : string; dst : remote }
        (** [Write_inline {data; dst}] copies [data], which the entry holds, to
            [dst]: the NIC reads no memory of this process. *)
    | Read of { src : remote; dst : local }
        (** [Read {src; dst}] copies [dst.bytes] bytes from [src] to [dst]. It
            reads what the writes posted before it on its queue pair wrote. *)

  type t = {
    op : op;
    signal : bool;
        (** The NIC writes a completion once the entry completes. It writes one
            for a failed entry whatever [signal] says. *)
  }
  (** The type for work entries. *)

  val max_inline : int
  (** [max_inline] is the most bytes a [Write_inline] carries in one basic
      block: [28]. *)

  val write : buffer -> int -> qp:int -> index:int -> t -> unit
  (** [write b at ~qp ~index e] writes [e] in the {!size} bytes of [b] at byte
      [at], as entry number [index] of queue pair number [qp]. [index] is the
      ring's producer count before the entry: the NIC reads its low 16 bits.

      Raises [Invalid_argument] if the bytes are not in [b], [at] is not a
      multiple of {!size}, [qp] is not in \[[0];[2{^24}-1]\], [index] is
      negative, a length is not in \[[0];[2{^31}-1]\], a key is not in
      \[[0];[2{^32}-1]\], an address is negative, or a [Write_inline] carries
      more than {!max_inline} bytes. *)
end

(** Completions: what the NIC writes once a work entry completes or fails.

    The completion ring holds [m] entries of {!size} bytes, [m] a power of two.
    Its consumer count [c] is how many completions the process read: the next is
    at byte [64 * (c mod m)], and the NIC wrote it once {!owned} says so. *)
module Completion : sig
  val size : int
  (** [size] is [64], the bytes of a completion. *)

  (** The type for the reasons an entry failed. *)
  type error =
    | Local_length  (** A local length is wrong. *)
    | Local_qp_operation  (** The entry is malformed for its queue pair. *)
    | Local_protection  (** Local memory is outside its region or its keys. *)
    | Flushed  (** The queue pair failed before the entry ran. *)
    | Bad_response  (** The peer answered with an unexpected response. *)
    | Local_access  (** The peer's request violated local access rights. *)
    | Remote_invalid_request
        (** The peer refused the request as malformed for its queue pair. *)
    | Remote_access  (** The peer refused the request's memory or key. *)
    | Remote_operation  (** The peer could not complete the request. *)
    | Retry_exceeded
        (** The peer acknowledged nothing within the queue pair's retries. *)
    | Remote_aborted  (** The peer aborted the operation. *)
    | Other_error of int  (** Another syndrome, by its number. *)

  (** The type for the outcomes of entries. *)
  type status =
    | Done  (** The entry completed. *)
    | Failed of { error : error; vendor : int }
        (** The entry failed: [vendor] is the NIC's own syndrome. *)
    | Unexpected of int
        (** The completion is of another kind than an entry of a send ring, such
            as a receive's, by its opcode. *)

  type t = {
    qp : int;  (** The number of the entry's queue pair. *)
    index : int;  (** The low 16 bits of the entry's number. *)
    status : status;
  }
  (** The type for completions. *)

  val owned : buffer -> int -> count:int -> entries:int -> bool
  (** [owned b at ~count ~entries] is [true] iff the completion at byte [at] of
      [b] is the one the NIC wrote for consumer count [count] of a ring of
      [entries] entries: its opcode is a valid one and its owner bit is bit
      [log2 entries] of [count]. A process reads the completion after [owned]
      answers [true], with loads ordered after the one that read the owner.

      Raises [Invalid_argument] if the bytes are not in [b], [count] is negative
      or [entries] is not a power of two. *)

  val read : buffer -> int -> t
  (** [read b at] is the completion at byte [at] of [b].

      Raises [Invalid_argument] if the bytes are not in [b]. *)

  val invalidate : buffer -> int -> unit
  (** [invalidate b at] marks the completion at byte [at] of [b] as none the NIC
      wrote for consumer count [0]: a new ring's entries are marked so. *)

  val message : t -> string
  (** [message c] describes [c] for a person, such as
      ["queue pair 0x1c3, entry 7: retry counter exceeded (vendor syndrome
       0x81)"]. *)
end

(** Doorbell records: the counts a process stores for the NIC in its memory.

    A queue pair's record and a completion queue's record are 8 bytes each,
    aligned to 8. *)
module Doorbell : sig
  val size : int
  (** [size] is [8], the bytes of a record. *)

  val send : int
  (** [send] is the byte offset in a queue pair's record of its send count, a
      32-bit word holding the low 16 bits of the ring's producer count. *)

  val consumed : int
  (** [consumed] is the byte offset in a completion queue's record of its
      consumer count, a 32-bit word holding the count's low 24 bits. *)

  val armed : int
  (** [armed] is the byte offset in a completion queue's record of its arm word,
      the first word of {!arm}. *)

  val arm : sequence:int -> count:int -> cq:int -> int * int
  (** [arm ~sequence ~count ~cq] is the two 32-bit words of the store that arms
      completion queue number [cq] at consumer count [count]: the NIC raises an
      event at the next completion after it. [sequence] counts the events the
      queue raised; its low 2 bits are stored. The first word is also stored at
      {!armed}. *)
end

(** Access regions (UAR): the NIC's pages through which a process rings
    doorbells.

    The kernel gives a process access regions as {e system pages} it maps from
    the NIC's device file; a system page holds [per_page] access regions of 4096
    bytes. Each access region has 4 doorbell registers, numbered across the
    system pages a process maps: register [r] is in access region [r / 4]. *)
module Uar : sig
  val per_region : int
  (** [per_region] is [4], the doorbell registers of an access region. *)

  val mapping : page:int -> int -> int
  (** [mapping ~page i] is the offset in the NIC's device file of system page
      [i] of the process's access regions, mapped uncached, on a host whose
      pages are [page] bytes.

      Raises [Invalid_argument] if [i] is not in \[[0];[255]\] or [page] is not
      a positive power of two. *)

  val register : per_page:int -> size:int -> int -> int * int
  (** [register ~per_page ~size r] is the system page and the byte offset in it
      of doorbell register [r], with [per_page] access regions per system page
      and registers of [size] bytes. A doorbell of a queue pair is 8 bytes
      stored at that offset.

      Raises [Invalid_argument] if [r] is negative, or [r mod 4] is [2] or [3]
      (registers a process never rings), [per_page < 1] or [size < 0]. *)

  val cq_doorbell : int
  (** [cq_doorbell] is the byte offset in an access region of the register a
      process stores {!Doorbell.arm} in. *)
end
