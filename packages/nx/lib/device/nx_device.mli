(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, their memory and their programs.

    A device is hardware with memory: the {!host}, or a GPU that a vendor
    library such as [nx.metal.device] opens. Memory is held in {!Buffer}s, a
    number of elements of one storage format ({!Nx_dtype.Scalar.t}) on one
    device, and copied between devices by {!Buffer.copy}. A GPU also loads
    {!Program}s, which the libraries that submit work to it launch.

    Work runs on a device asynchronously. Each device has a {e timeline}: the
    value its last submitted work signals when it completes. {!synchronize}
    waits for it, and for the work of other devices that touched the device's
    memory. Copies and the return of memory to the system synchronize the
    devices involved first.

    Every function may be called from any domain. The operations on one device
    run one at a time: each takes the device for its whole duration, so they are
    ordered as they take it. An operation on several devices takes them in a
    fixed order.

    {b Reclamation.} Nothing frees a buffer by hand. Once a buffer and all its
    views are unreachable, the garbage collector hands its memory back to the
    device, which reclaims it at the start of its next operation: a GPU keeps it
    in a cache for reuse, the host returns it to the heap, and a borrowing
    device unmaps a borrow once its work is done. Memory goes back to the system
    only once no work can still use it. If that work cannot be waited for, the
    memory is {e retained}: kept, and never freed or reused. An allocation that
    the device's {!budget} or its driver refuses first releases the cache to the
    system, then collects garbage and tries again, and raises {!Out_of_memory}
    only after that. The collector finalises a buffer in the domain that created
    it, so memory dropped by a domain that is not allocating returns when that
    domain next runs its finalisers.

    {b Hangs.} {!synchronize} and {!Buffer.copy} wait for the work of the
    devices involved. They raise [Failure "NAME hang detected"], where [NAME] is
    a device's name, if one does not signal within its timeout: 30 seconds
    unless its vendor library sets another. *)

(** {1:devices Devices} *)

type t
(** The type for devices. There is one value per device: every open of a device
    returns the value its first open made. *)

val host : t
(** [host] is the host, named ["CPU"], with a budget of [max_int]. Its buffers
    are memory of the process's heap, which it does not cache. It loads no
    programs. *)

val name : t -> string
(** [name d] is [d]'s name: ["CPU"] for the host, ["METAL"] for the Metal GPU.
*)

val arch : t -> string
(** [arch d] is the architecture of [d]'s processor: the machine's instruction
    set for the host, such as ["arm64"] or ["x86_64"], and the GPU family for
    Metal, such as ["Apple7"]. *)

val equal : t -> t -> bool
(** [equal d d'] is [true] iff [d] and [d'] are the same device. *)

val synchronize : t -> unit
(** [synchronize d] returns once the work submitted to [d], and the work
    submitted to other devices that touched [d]'s memory, has completed.

    Raises [Failure "NAME hang detected"] if [d] or one of those devices does
    not signal in time. *)

(** {1:memory Memory} *)

val budget : t -> int
(** [budget d] is the most bytes [d]'s allocator holds at once, in live buffers
    and in its cache together. It is [max_int] for the host, and defaults to a
    device's recommended working set otherwise. *)

val set_budget : t -> int -> unit
(** [set_budget d n] sets [d]'s budget to [n], releasing cached memory to the
    system until [d] holds at most [n] bytes or its cache is empty. Live buffers
    are never released: an allocation fails until enough of them are collected.

    Raises [Invalid_argument] if [n < 0]. *)

val free_cache : t -> unit
(** [free_cache d] returns all of [d]'s cached memory to the system. *)

exception Out_of_memory of t * int
(** [Out_of_memory (d, n)] is raised by {!Buffer.create} when [d] cannot
    allocate [n] bytes: at once if [n] exceeds [d]'s {!budget}, and otherwise if
    the budget or the driver refuses them after [d]'s cache was released and
    unreachable buffers collected. *)

(** {1:buffers Buffers} *)

(** Buffers of device memory. *)
module Buffer : sig
  type device := t

  type t
  (** The type for buffers: {!length} elements of format {!dtype} in a range of
      one device's memory. Their bytes are the elements in their storage
      representation, in order: [Nx_dtype.Scalar.bitsize s / 8] bytes each, and
      two per byte for [Int4] and [UInt4], the first in the low nibble.
      Multi-byte elements are little-endian, the byte order of the host on arm64
      and x86_64 and of Metal; a big-endian host holds its own byte order.

      A buffer is {e owned} when {!create} made it and {e borrowed} when it is
      over memory that something else holds ({!of_bigarray}, {!borrow}); a
      {!view} is as the buffer it views. Owned memory returns to its device once
      the buffer and all its views are unreachable. Borrowed memory is never
      cached, and never counted in a device's budget or statistics. *)

  val create : device -> Nx_dtype.Scalar.t -> int -> t
  (** [create d s n] is an owned buffer of [n] elements of format [s] on [d].
      Its contents are unspecified. A buffer of no bytes allocates nothing.

      Raises [Invalid_argument] if [n < 0] or if [n] elements of [s] take more
      than [max_int] bytes, and {!Out_of_memory} if [d] cannot allocate its
      bytes. *)

  val of_bigarray :
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t -> t
  (** [of_bigarray ba] is a borrowed buffer of format [UInt8] on {!host} over
      [ba]'s bytes, without a copy. A write through either is seen through the
      other. The buffer keeps [ba] reachable. When [ba] is over memory that
      OCaml does not manage, such as memory a C library allocated, its owner
      must keep that memory alive for as long as the buffer and its views are
      reachable. *)

  val borrow : device -> t -> t
  (** [borrow d b] is a borrowed buffer on [d] over the memory of the host
      buffer [b], without a copy, of [b]'s format and length. A write through
      either is seen through the other, once the devices involved are
      synchronized. The result keeps [b] reachable. [borrow host b] is [b]. Work
      that reads or writes through the result touches the host's memory: its
      {!submit} lists {!host} in [touches].

      Raises [Invalid_argument] if [b] is not on {!host}, or if [d] cannot
      address the host's memory. *)

  val device : t -> device
  (** [device b] is the device whose memory [b] is. *)

  val dtype : t -> Nx_dtype.Scalar.t
  (** [dtype b] is the format of [b]'s elements. *)

  val length : t -> int
  (** [length b] is the number of elements of [b]. *)

  val nbytes : t -> int
  (** [nbytes b] is [b]'s size in bytes. *)

  val is_borrowed : t -> bool
  (** [is_borrowed b] is [true] iff [b] is borrowed. *)

  val view : t -> offset:int -> Nx_dtype.Scalar.t -> int -> t
  (** [view b ~offset s n] is the [n] elements of format [s] from byte [offset]
      of [b] on. It shares [b]'s memory: a write through either is seen through
      the other.

      Raises [Invalid_argument] if [offset] or [n] is negative, if [n] elements
      of [s] take more than [max_int] bytes, if the view's bytes do not lie
      inside [b]'s, or if its first byte is not aligned to the size of one
      element of [s]. *)

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they are
      there. It first synchronizes the devices of [src] and [dst]. A copy
      between two devices counts in [src]'s [bytes_out] and in [dst]'s
      [bytes_in].

      Raises [Invalid_argument] if [src] and [dst] have different sizes in
      bytes, and [Failure "NAME hang detected"] if a device it waits for does
      not signal in time. *)

  (** {2:low Low-level}

      For the libraries that submit work over buffers. *)

  val address : t -> nativeint
  (** [address b] is the address of [b]'s first byte in its device's address
      space, as the device's work reads it. *)

  val host_address : t -> nativeint
  (** [host_address b] is the address of [b]'s first byte in the host's address
      space. Reading or writing it is outside the device's ordering: synchronize
      first. *)

  val handle : t -> nativeint
  (** [handle b] is the driver's object for the memory [b] lies in, such as a
      [MTLBuffer], and [0n] on the host. [b] starts {!offset} bytes into it. *)

  val offset : t -> int
  (** [offset b] is the byte offset of [b]'s first byte in {!handle}[ b]. *)
end

(** {1:programs Programs} *)

(** Programs loaded on a device. *)
module Program : sig
  type device := t

  type t
  (** The type for programs: a function of a binary, loaded on a device. The
      libraries that submit work launch them; this module does not. *)

  val load : device -> binary:string -> name:string -> t
  (** [load d ~binary ~name] is the function [name] of [binary], a compiled
      library in [d]'s format, such as a metallib for Metal. Loading the same
      binary and name on [d] again returns the same program.

      Raises [Invalid_argument] if [d] loads no programs, and [Failure] with the
      driver's message if it rejects [binary] or has no function [name]. *)

  val device : t -> device
  (** [device p] is the device [p] is loaded on. *)

  val name : t -> string
  (** [name p] is the name of [p]'s function. *)

  val handle : t -> nativeint
  (** [handle p] is the driver's object for [p], such as a
      [MTLComputePipelineState]. *)
end

(** {1:stats Statistics} *)

(** Device statistics. *)
module Stats : sig
  type t
  (** The type for a snapshot of a device's statistics. *)

  val allocated : t -> int
  (** [allocated s] is the bytes of owned memory in buffers that
      {!Buffer.create} returned and that were not yet returned to the device. *)

  val cached : t -> int
  (** [cached s] is the bytes in the device's cache: memory allocated from the
      driver and held for reuse. *)

  val retained : t -> int
  (** [retained s] is the bytes of the device's own memory that it retains
      because the work that last used them could not be waited for. They count
      against its budget. Retained borrows are not counted. *)

  val bytes_in : t -> int
  (** [bytes_in s] is the bytes copied into the device from another one. *)

  val bytes_out : t -> int
  (** [bytes_out s] is the bytes copied from the device to another one. *)

  val diff : t -> t -> t
  (** [diff s s'] is what changed from [s] to [s']: each count of [s'] minus
      that of [s]. The [allocated] of a diff is the bytes allocated in between
      and not returned. *)
end

val stats : t -> Stats.t
(** [stats d] is a snapshot of [d]'s statistics. The memory of buffers collected
    before the call counts as returned. *)

(** {1:submitting Submitting work}

    For the libraries that submit work to a device. Work is submitted inside
    {!submit}, which orders it after the device's earlier work and gives it the
    value it must signal on completion. *)

val submit : t -> touches:t list -> (int -> 'a) -> 'a
(** [submit d ~touches f] is [f v], run with [d] and the devices of [touches]
    taken, where [v] is the next value of [d]'s timeline. [f] submits work to
    [d] that signals [v] when it completes, and whose memory accesses reach [d]
    and the devices of [touches]. Work through a buffer borrowed from the host
    ({!Buffer.borrow}) touches the host: list {!host} in [touches]. Once [f]
    returns, [v] is [d]'s submitted value, and {!synchronize} on each device of
    [touches] waits for [v] on [d].

    If [f] raises, nothing is recorded, so [f] may raise only before it commits
    any work: committed work that signals [v] would signal a value the next
    submission takes again.

    [f] must not use [d] or the devices of [touches] through this module, which
    they are taken by: it submits through the handles its vendor library gives.
*)

val submitted : t -> int
(** [submitted d] is the value [d]'s last submitted work signals, [0] before any
    submission. *)

val signaled : t -> int
(** [signaled d] is the last value [d] signaled. Work that signals
    [v <= signaled d] has completed. *)

val timeline : t -> Buffer.t
(** [timeline d] is a host buffer of two [UInt64]: a signal word, then [d]'s
    submitted value. On a device made without its own [signal], work signals by
    storing its value into the signal word, and {!signaled} reads it. A device
    with its own signal, such as Metal's shared event, reports through
    {!signaled} alone and leaves the signal word at [0]. *)

(** {1:vendors Vendor runtimes}

    A library that opens devices of some kind makes each of them with {!make},
    once, and returns that value from every later open. *)

type memory = {
  host : nativeint;  (** The memory's first byte, as the host addresses it. *)
  device : nativeint;  (** Its first byte, as the device's work addresses it. *)
  handle : nativeint;  (** The driver's object for it. *)
}
(** The type for memory that a driver allocated or mapped. The host addresses
    all of it: the device's copies are host memory copies. *)

type signal = {
  signaled : unit -> int;  (** The last value the device signaled. *)
  wait : int -> timeout_ms:int -> bool;
      (** [wait v ~timeout_ms] waits until the device signaled [v], for at most
          [timeout_ms] milliseconds; [false] if it did not. *)
}
(** The type for a device's own completion signal. *)

val make :
  name:string ->
  arch:string ->
  budget:int ->
  alloc:(int -> memory option) ->
  free:(memory -> unit) ->
  ?borrow:(nativeint -> int -> memory option) ->
  ?load:(binary:string -> name:string -> nativeint) ->
  ?signal:signal ->
  ?timeout_ms:int ->
  ?synchronized:(unit -> unit) ->
  unit ->
  t
(** [make ~name ~arch ~budget ~alloc ~free ?borrow ?load ?signal ?timeout_ms
     ?synchronized ()] is a new device:
    - [alloc n] is [n > 0] bytes of new memory, or [None] if the driver has
      none; [free m] returns [m] to the driver. Memory the device frees is
      cached for reuse first, and the device synchronizes before it calls
      [free], so no work still uses [m].
    - [borrow a n] maps the [n] bytes of host memory at [a] for the device:
      memory whose [host] is at or below [a], [None] if the driver cannot.
      [free] unmaps it. Without [borrow], the device cannot {!Buffer.borrow}.
    - [load ~binary ~name] loads a program, raising [Failure] if the driver
      rejects it. Without [load], the device loads no programs.
    - [signal] is how the device signals completion. Without it, the device's
      work signals by storing into its {!timeline}.
    - [timeout_ms] is how long a wait for its work lasts before the device is
      considered hung. Defaults to [30_000]. Without [signal], the timeout
      restarts whenever the signal word moves.
    - [synchronized ()] runs at the end of each synchronization of the device.
      Defaults to doing nothing.

    These functions run while the device is taken, and must not use it through
    this module. Blocking driver calls should release the OCaml runtime.

    Raises [Invalid_argument] if [budget < 0] or [timeout_ms <= 0]. *)
