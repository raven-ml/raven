(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the GPU suites and benches share: host memory by address, and a GPU of
    one driver opened through rig. *)

val opened : 'a Rig_edge.opener -> ('a, string) result
(** [opened o] is [o.make] given a {!Host.word}: the device opened outside
    rig, for a test that calls its driver itself. *)

(** {1:dmabufs Dma-bufs} *)

(** The dma-bufs a driver exports ({!Rig.Buffer.dma}), by descriptor. *)
module Dmabuf : sig
  val size : int -> int option
  (** [size fd] is the bytes of the dma-buf open as [fd], from Linux's
      [/proc/self/fdinfo/FD], or [None] if no exporter is named there
      ([exp_name]), as for a descriptor that is no dma-buf. *)

  val read : int -> at:int -> int -> string option
  (** [read fd ~at n] is the [n] bytes from byte [at] of the dma-buf [fd],
      read through a mapping of it inside [DMA_BUF_IOCTL_SYNC]'s start and
      end, or [None] if its exporter maps none, or off Linux. *)

  val cloexec : int -> bool
  (** [cloexec fd] is [true] iff [fd] closes on [exec]. *)

  val close : int -> unit
  (** [close fd] closes [fd]. *)
end

val bar_bytes : string -> bar:int -> (int * int) list -> int -> string
(** [bar_bytes bus ~bar runs n] is the first [n] bytes at the bus ranges
    [runs], as (address, bytes), of BAR [bar] of the function at [bus],
    read through mappings of its [resourceN] file under [/sys/bus/pci],
    which root may read. Each range lies in the BAR, at its bus address.

    Raises [Unix.Unix_error] if the file cannot be mapped. *)

val has : sub:string -> string -> bool
(** [has ~sub s] is [true] iff [sub] occurs in [s]. *)

val child : string -> unit
(** [child flag] runs this executable again with the one argument [flag]. A
    suite runs a test in a child where the test leaves its process changed
    for good, as an export of a GPU's memory over PCI does. The test fails
    unless the child exits with [0], and skips for the reason the child
    gives if the child ends with {!skipped}. *)

val skipped : string -> 'a
(** [skipped why] ends this process, a child {!child} runs, as a skip of the
    test for [why]. *)

(** {1:host Host memory} *)

(** Host memory the suites own, by address. *)
module Host : sig
  val at : Rig.Buffer.t -> int
  (** [at b] is the host address of [b]'s first byte, with no wait or claim:
      a probe of what the device wrote, read while work may run. *)

  val data : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int
  (** [data a] is the host address of [a]'s bytes while [a] is reachable. *)

  val page : int
  (** [page] is the host's page size, in bytes. *)

  val pages : ?read_only:bool -> int -> int
  (** [pages n] is the address of [n] zeroed bytes from a page boundary,
      read-only iff [read_only] (defaults to [false]). *)

  val word : unit -> nativeint
  (** [word ()] is the address of 8 zeroed bytes of host memory that live
      for the process: the word an opener's [make] is given
      ({!Rig_edge.type-opener}) by a test that calls [make] itself. *)

  val free_pages : int -> int -> unit
  (** [free_pages a n] gives back the [n] bytes {!pages} gave at [a]. *)

  val get8 : int -> int
  (** [get8 a] is the byte at [a]. *)

  val set8 : int -> int -> unit
  (** [set8 a x] stores [x] as the byte at [a]. *)

  val get32 : int -> int
  (** [get32 a] is the unsigned 32-bit word at [a]. *)

  val set32 : int -> int -> unit
  (** [set32 a x] stores [x] as the 32-bit word at [a]. *)

  val get64 : int -> int
  (** [get64 a] is the 64-bit word at [a], read with acquire order. *)

  val set64 : int -> int -> unit
  (** [set64 a x] stores [x] in the 64-bit word at [a] with release order. *)

  val read : int -> int -> string
  (** [read a n] is the [n] bytes at [a]. *)

  val write : int -> string -> unit
  (** [write a s] stores [s] at [a]. *)
end

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for [ms] milliseconds of the monotonic clock
    ({!Rig.Profile.now}), the clock a device's hang bound counts, and fails the
    test if a read is ever other than [x] under [w]. *)

(** {1:gpu A GPU} *)

(** The GPU a suite acts on. *)
module type Gpu = sig
  module D : Rig_edge.Driver

  val class_ : string
  (** [class_] names the GPU, as ["CUDA"]: the skip's reason and rig's name of
      the device. *)

  val present : unit -> bool
  (** [present ()] is [true] iff the machine has a GPU of the class that rig
      drives, from its files and settings alone: it starts no vendor library. *)

  val open_ : word:nativeint -> (D.t, string) result
  (** [open_ ~word] opens the driver's device of the GPU on the timeline
      word at [word] ({!Rig_edge.type-opener}). *)
end

(** {1:work Work} *)

type kept = Kept : 'a -> kept  (** A value of any type. *)

type work = {
  part : int -> Rig.Submission.part;
  buffers : Rig.Buffer.t array;
  fixed : (Rig.Buffer.t * Rig.access) list;
  keep : kept;
  init : Rig.Submission.t -> Rig.Submission.Run.t -> int -> unit;
}
(** Work with its memory, retained objects, and initialization. [part base]
    is the part, its refs naming [buffers] from the slot [base] on. Its
    submission keeps [keep], objects the work uses. [init s r i] writes the
    parameters or fixed words of part [i] after [s] is made. *)

val work : Rig.Submission.part -> work
(** [work p] is [p] with no buffers, extra memory or initialization. *)

val prepare :
  ?fixed:(Rig.Buffer.t * Rig.access) list ->
  Rig.t ->
  work list ->
  Rig.Submission.Run.t * Rig.Buffer.t array
(** [prepare d ws] is a run of the submission that assembles [ws], their
    slots one after the other, with its buffers. It initializes their blocks
    and fixed memory, and the run stays taken by the calling thread. [fixed]
    adds fixed memory. *)

val launching :
  Rig.Image.t ->
  string ->
  params:int ->
  ?groups:int ->
  ?threads:int ->
  (Rig.Buffer.t * Rig.access * int) array ->
  (Rig.Submission.Run.t -> Rig.Submission.block -> unit) ->
  work
(** [launching image f ~params ~groups ~threads uses store] is a launch of
    [image]'s [f] over [groups] (defaults to [1]) groups of [threads] (defaults
    to [1]) threads, each [(b, a, at)] of [uses] the buffer [b] referenced at
    the parameter byte [at] with the access [a], and its [params] bytes of
    parameters [store] stores. *)

val params : Rig.Submission.Run.t -> Rig.Submission.block -> string -> unit
(** [params r b s] stores [s], padded to a multiple of four bytes, into [b]. *)

(** A GPU's device, opened through rig or by its driver alone. *)
module type S = sig
  type gpu
  (** The type for the driver's devices. *)

  type t = { d : Rig.t; g : gpu }
  (** The type for an open GPU: [d] as programs reach it, [g] its driver's
      device, which rig owns. *)

  val class_ : string
  (** [class_] names the GPU, as ["CUDA"]. *)

  val present : unit -> bool
  (** [present ()] is [true] iff the machine has a GPU of the class that rig
      drives, from its files and settings alone: it starts no vendor library. *)

  val open_ : unit -> t
  (** [open_ ()] is the GPU opened by its driver and handed to rig under a name
      of the GPU's. It first ends what the last [open_] or {!driver} made
      ({!release}). It skips the test if the machine has no such GPU, and fails
      it if the GPU does not open. *)

  val open_as : (module Rig_edge.Driver with type t = gpu) -> t
  (** [open_as m] is {!open_}, the device handed to rig under the driver [m],
      such as [D] wrapped to declare other facts. *)

  val open_own : (module Rig_edge.Driver with type t = gpu) -> t
  (** [open_own m] is {!open_as}[ m], the driver's [make] given a word
      outside rig's array that holds the slot word's value, as a caller that
      calls [make] itself may give: a driver whose work writes only words
      it maps then keeps a word of its own ({!Rig_edge.type-facts}). *)

  val close : t -> unit
  (** [close t] is {!Rig.close}[ t.d]. *)

  val with_ : (t -> 'a) -> 'a
  (** [with_ f] is [f t], [t] the {!open_}ed GPU, closed after [f] returns or
      raises. *)

  val driver : unit -> gpu
  (** [driver ()] is the GPU opened by its driver alone, which rig never takes,
      as {!open_} opens it. The caller stops it with {!stop_driver}. *)

  val stop_driver : gpu -> unit
  (** [stop_driver g] stops [g], which {!driver} or {!handed} opened. *)

  val handed : string -> t
  (** [handed name] is the GPU opened in rig under [name], as {!open_} opens
      it, for a test of the driver's own stop: the test stops its handle with
      {!stop_driver}, and the device stays open in rig until the process ends,
      as no driver stops twice. *)

  val with_driver : (gpu -> 'a) -> 'a
  (** [with_driver f] is [f g], [g] {!driver}[ ()], stopped after [f] returns or
      raises. *)

  val with_shared : (t -> 'a) -> 'a
  (** [with_shared f] is [f t], [t] the GPU the last [with_shared] opened if
      nothing opened since, else a new {!open_}. [t] stays open after [f]
      returns, until the next open or {!release}, and is closed if [f]
      raises. A property's cases share one open this way, which a GPU whose
      open boots its firmware for seconds needs. *)

  val with_shared_driver : (gpu -> 'a) -> 'a
  (** [with_shared_driver f] is {!with_shared} for the GPU opened by its driver
      alone, as {!with_driver} opens it. *)

  val release : unit -> unit
  (** [release ()] ends what the last {!open_} or {!driver} made, if a failed
      test left it: it closes a device in rig, and stops one opened alone. *)

  val submit : t -> Rig.Submission.part array -> Rig.Point.t
  (** [submit t ps] submits [ps] through rig on [t] and is their point. *)

  val submit_work : t -> work list -> Rig.Point.t
  (** [submit_work t ws] submits [ws]' parts through rig on [t], with their
      initialized memory, and is their point. *)
end

(** [Make (G)] opens [G]'s GPU under the name [G.class_ ^ ":test"]. *)
module Make (G : Gpu) : S with type gpu = G.D.t

(** {1:watch A driver's word} *)

(** [Watch (D)] is [D], which also counts, for each device whose facts rig
    read, the frees of its word and the reads of it after a free. *)
module Watch (D : Rig_edge.Driver) : sig
  include
    Rig_edge.Driver
      with type t = D.t
       and type region = D.region
       and type image = D.image

  val word_frees : t -> int
  (** [word_frees h] is the number of {!Rig_edge.Driver.free} calls on the
      word [h]'s facts give, by physical equality. *)

  val late_reads : t -> int
  (** [late_reads h] is the number of {!Rig_edge.Driver.signaled} calls on
      [h] after a free of its word. *)
end

(** {1:edge A driver's copies} *)

(** The type for a copy part's fields, as [rig_edge.h] lays them out: each
    side's handle, or host address for a side [local] names, and offset. *)
type raw = {
  dst : nativeint;
  dst_offset : int;
  src : nativeint;
  src_offset : int;
  width : int;
  rows : int;
  src_pitch : int;
  dst_pitch : int;
  local : [ `None | `Src | `Dst ];  (** [copy.local]. *)
  device_memory : bool;  (** [copy.device_memory]. *)
  fill : char option;
      (** [Some c] for a fill of [c]: [copy.fill] and [copy.byte]. *)
}

(** The type for what a hand-over of a part answers. *)
type answer =
  | Handed  (** Submitted and committed. *)
  | Room of int
      (** The room check's answer, other than [RIG_FITS]: [1] for
          [RIG_LATER], [2] for [RIG_NEVER], [3] for [RIG_GROW]. *)
  | Failed  (** The submit or the commit failed. *)

val hand : nativeint -> v:int -> queue:int -> raw -> answer
(** [hand edge ~v ~queue p] hands the device whose C state is [edge] the
    value [v], one copy part [p] on its queue [queue], through its room
    check, submit and commit, as rig calls them. It waits for nothing. *)

(** [Edge (D)] hands copy parts to a device of [D] opened by its driver
    alone ({!hand}), for the laws of copy parts that rig's interface does
    not make, such as boxes. *)
module Edge (D : Rig_edge.Driver) : sig
  type t
  (** The type for a device and the memory allocated for its parts. *)

  (** The type for memory of the device. *)
  type region = {
    r : D.region;
    memory : Rig_edge.memory;  (** Its kind. *)
    bytes : int;  (** Its bytes. *)
  }

  (** The type for a side of a part: the device's memory, or this process's
      memory at a host address, which only a device of another machine
      takes. *)
  type side = Region of region | Host of int

  (** The type for copy parts. A part with a [fill] sets its row to the byte
      and has no [src]. *)
  type part = {
    dst : side;
    dst_offset : int;
    src : side option;
    src_offset : int;
    width : int;
    rows : int;
    src_pitch : int;
    dst_pitch : int;
    fill : char option;
  }

  val make : D.t -> t
  (** [make g] hands work to [g], which its driver opened alone, from the
      value after the one its word holds. *)

  val alloc : t -> Rig_edge.memory -> int -> region
  (** [alloc t m n] is [n] bytes of [t]'s memory [m], until {!release}. It
      fails the test if the device has none. *)

  val release : t -> unit
  (** [release t] frees the memory {!alloc} made. *)

  val run : t -> part -> unit
  (** [run t p] hands [p] to the device's copy engine as the next value, its
      [copy.device_memory] set where both sides are [Device] memory and its
      [copy.local] where a side is [Host], and waits until the word shows
      that value. It fails the test if the device never runs [p] or fails
      it, or the value is not reached within 10 s. *)

  val read : t -> region -> at:int -> int -> string
  (** [read t r ~at n] is the [n] bytes of [r] from [at], copied by a part
      into [Pinned] memory where the host does not address [r]. *)

  val write : t -> region -> at:int -> string -> unit
  (** [write t r ~at s] writes [s] into [r] from [at], as {!read}. *)
end

(** {1:conformance Conformance}

    What the conformance suite needs of a GPU beyond its contract: a binary and
    work on the device's first queue, which a driver's support makes from its
    suite's fixtures, found from the directory a suite runs in. Each work names
    the memory it reads and writes. *)
module type Conformance = sig
  module D : Rig_edge.Driver
  include S with type gpu = D.t

  val binary : unit -> string * string list
  (** [binary ()] is a binary of the driver's fixtures and the kernels it holds.
  *)

  val second : word:nativeint -> (D.t, string) result option
  (** [second ~word] opens another device of the driver beside the one
      {!open_} gave, while that one is open, on the timeline word at [word],
      or is [None] where the machine has none. The caller stops it. *)

  val copy_words : t -> dst:Rig.Buffer.t -> src:Rig.Buffer.t -> work
  (** [copy_words t ~dst ~src] is work on [t]'s first queue that copies [src]'s
      bytes into [dst], buffers of [t] of one length, a positive multiple of 4,
      keeping the memory it addresses. *)

  val spin : t -> ns:int -> work
  (** [spin t ~ns] is work on [t]'s first queue that runs at least [ns]
      nanoseconds. *)

  val clock_error : since:int -> int
  (** [clock_error ~since] is the most, in nanoseconds, by which a span of the
      GPU's work lies outside the host's clock times around its submit and
      its wait, [since] nanoseconds after the device opened, from its
      driver's interface. *)

  val launch_binary : unit -> string option
  (** [launch_binary ()] is a binary of the driver's fixtures holding the launch
      laws' kernels, or [None] where the driver runs no launch. Their parameters
      are 8-byte words at their offsets, and 4-byte ones where named [u32] or
      [f32]; [k] is a thread's index in the grid, [g * T + t]: [g] its group's
      index in the grid, x fastest, [t] its index in its group, x fastest, [T]
      the threads of a group.
      - [ids], parameters [out] (0), [a] (8), [b] (16, u32) and [f] (20, f32):
        stores [a + b * k + f], [f] rounded toward zero, modulo [2{^32}], into
        the 32-bit word [k] at [out].
      - [twice], parameters [dst] (0), [src] (8) and [c] (16, u32), over a grid
        of one dimension: stores [2 x + c] modulo [2{^32}], [x] the 32-bit word
        [k] at [src], into the 32-bit word [k] at [dst]. *)

  val wide_binary : t -> string option
  (** [wide_binary t] is a binary of the driver's fixtures holding [wide], or
      [None] where [t] runs no launch of it: no launch at all, or, on CUDA,
      compute capability below 7.0, where kernels take at most 4,096 bytes
      of parameters. Its parameters are [out] (0) then 32-bit words up to
      byte 8,192; it stores the words at bytes 4,096 and 8,188 into the
      32-bit words 0 and 1 at [out]. *)
end

val loader : (unit -> string) -> Rig.t -> Rig.Image.t
(** [loader bin] is [bin ()] loaded on a device, once per device while it is not
    lost. *)

val arguments : Rig.t -> string -> Rig.Buffer.t
(** [arguments d s] is new [Pinned] memory of [d] holding [s]. *)
