(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a driver is to rig.

    A driver runs one kind of hardware once it is open: it holds the device's
    memory, writes its queues, makes completion observable and reports faults.
    A vendor library links only this library and matches {!Driver}; a program
    passes it to [Rig.open_]. Work crosses in C, through the structures of
    [rig_edge.h]; everything else is the OCaml values below.

    A device's {e timeline word} is eight bytes the driver alone writes: the
    last committed value [v] such that every submission up to [v] completed. It
    never moves backwards, and work never writes it. A driver owes three
    properties:
    - {e Prefix order}: the work it is handed on each queue runs in the order
      handed.
    - {e Device order}: a value's work starts after the previous value's work
      completed, on every queue. The hand-over waits on the queue that released
      the previous value.
    - {e Observable completion}: the host learns that a prefix completed, or
      that the device faulted. *)

(** {1:facts Facts} *)

(** The type for the memories of a device. *)
type memory =
  | Device  (** The device's own memory. *)
  | Pinned
      (** Host memory that both the device's work and the host address,
          page-locked where the host pages. *)
  | Mapped
      (** The device's own memory, which the host also addresses through a
          write-combined window. *)

(** The type for the work of a part, as [rig_edge.h] tags it. *)
type kind =
  | Words  (** 32-bit words placed on the queue. *)
  | Fill  (** A C function called with the queue's context. *)
  | Copy  (** A copy between memory of the device. *)
  | Launch  (** A function of an image run over a grid of groups of threads. *)

type queue = {
  name : string;
      (** The name parts give, such as ["COMPUTE:0"]. rig reads nothing else
          from it. *)
  runs : kind list;
      (** What the queue runs: rig refuses a part of another kind. *)
}
(** The type for a device's queues. A device's first queue is the one rig waits
    through and its default compute queue. Its {e copy queue} is the first
    queue after the first whose [runs] lists [Copy], or, where none does, the
    first queue if its [runs] lists [Copy]: rig's own copies go there. *)

(** The type for how a device's word advances. *)
type completion =
  | Store  (** Its queue stores the word. *)
  | Host
      (** The host writes it, from the driver's handler or before the hand-over
          or the commit returns. *)
  | Object of nativeint
      (** An object of the driver's API completes, and the driver writes the
          word. The object fits in 62 bits: it crosses the hand-over's waits as
          an integer. *)

type waits = {
  stores : bool;  (** Its queues wait for a [Store] producer. *)
  hosts : bool;  (** Its queues wait for a [Host] producer. *)
  objects : bool;
      (** Its queues wait for an [Object] producer of the same driver, whose
          objects they know. *)
  most : int;  (** The most waits one submission carries in its queues. *)
}
(** The type for the producers a device's queues wait for in the queue. rig
    waits on the host for the others, and for those beyond [most]. *)

(** The type for what compiled code needs from a device: a record the driver's
    ABI library declares, under its key. *)
type capability = Capability : 'a Type.Id.t * 'a -> capability

type 'region facts = {
  arch : string;  (** The device's architecture, such as ["gfx1100"]. *)
  budget : int;  (** The bytes of own memory the device should hold at most. *)
  queues : queue list;
      (** The device's queues. A part names its queue by name. *)
  completion : completion;  (** How the word advances. *)
  waits : waits;  (** The producers its queues wait for. *)
  may_block : bool;
      (** Whether the C room check, hand-over and commit may block on the
          device's own earlier work, its own transfers or its library's
          back-pressure. Where they may, they are counted calls: {!Driver.stop}
          never runs while one blocks. *)
  hang_ms : int option;
      (** [Some n] if work whose word has not moved for [n] milliseconds hung,
          where nothing else bounds the device's work; [None] where the
          vendor's driver does. [n] is positive. rig's waits count while a
          committed value stays above an unmoved word, from the first wait
          that saw it so, and the loss may come late but never early. rig then
          loses the device with the reason ["no progress for n ms"], which it
          passes to {!Driver.stop}. The clock does not run while the device's
          queue waits for another device's unreached value: work that waits
          in the queue may wait however long the other device's work runs. *)
  maps_host : bool;
      (** Whether the device maps host memory of its machine that starts on a
          page ({!Driver.map_host}). *)
  host_addresses : bool;
      (** Whether the host addresses every region the device allocates: each
          one's location has a [host] address ({!Driver.locate}). The host then
          reaches the device's memory and copies it, and the device's [Pinned]
          memory is its own, counting in [budget]. An allocation whose region
          breaks it raises [Invalid_argument], the region freed. *)
  capability : capability;
      (** The ABI record, filled when the device opened. *)
  word : 'region;
      (** The timeline word. Other devices may map it, and it is read after a
          loss. rig frees it ({!Driver.free}) after {!Driver.stop}, once nothing
          reads it. It has a host address except behind a transport. *)
  edge : nativeint;
      (** The address of the device's C state, valid while the process runs,
          whose first member points to the driver's [struct rig_driver] of
          [rig_edge.h]: its room check, hand-over and commit, each called with
          [edge]. None calls a function of the OCaml runtime or reads an OCaml
          value.
          - The room check answers whether parts fit the device's queues now,
            once one of its values is reached, or never, for parts that exceed
            its empty queues, name work it does not run or are of no kind, and
            for a launch whose grid or group is empty or exceeds the device's or
            the function's limits, or whose shared memory exceeds the
            function's.
          - The hand-over encodes the parts on the device's queues as the work
            of the value after the last one it received, after the waits and
            the device's earlier work. The work runs without another call. The
            device writes the value into the word once the value is committed
            and the work up to it completed. When it returns, all of the
            value's work is on the queues; a failure loses the device. While a
            profile is taken, the device also writes, before the word shows
            the value, when the value's work started and ended on it, on the
            host's clock, or writes nothing for work it cannot time.
          - The commit, given [v], at most the last value the device received,
            makes the device write [v] or a later value into the word once the
            work up to it completed. Committing a committed value does nothing.
            A hand-over that answers [RIG_COMMITTED] committed every value up
            to its own; a driver whose hand-overs all do is never asked to
            commit. The driver also commits on its own, at least once every [k]
            values it receives, [k] a bound of its own, so the word shows each
            value at most [k] values late while work is submitted. A failure
            loses the device. *)
}
(** The type for the facts of a device, read once when it opens: a fault then is
    the open's [Error]. *)

(** {1:memory Memory} *)

type location = {
  address : int option;
      (** The region's address as the device's work addresses it, or [None]
          for memory named by its handle only. *)
  host : int option;
      (** The host address of the region's first byte, if the host addresses
          it. *)
  handle : nativeint;
      (** The driver's object for the region: a buffer object, a
          [CUdeviceptr], an [MTLBuffer]. *)
}
(** The type for where a region lies. *)

(** {1:code Code} *)

(** The type for a binary a driver took. *)
type ('region, 'image) code =
  | Loaded of 'image  (** The driver's library placed the code itself. *)
  | Place of int * ('region -> 'image * string)
      (** The code goes into [n] bytes of the device's [Device] memory: the
          function is the image over the region rig allocated with at least [n]
          bytes, and the bytes to place at its start, at most [n]. The driver
          makes nothing on the device before the function is called, so a
          [Place] whose function is never called leaves nothing to release; the
          function calls nothing that may block or fault, and raises nothing. *)

type entry = {
  code : int;
      (** What compiled code names the function by: a kernel descriptor's
          address, a code address, a [CUfunction], a pipeline. *)
  launch : nativeint;
      (** What the hand-over reads to launch the function ([launch.launch] in
          [rig_edge.h]), such as its dispatch template and limits, valid while
          the image is loaded; [0n] for a driver whose queues run no [Launch].
      *)
}
(** The type for a function of an image, as a driver names it. *)

(** {1:drivers Drivers} *)

(** Drivers.

    {b Calls.} {!locate} and {!peer} call nothing that may block or fault, and
    rig calls them at any time. Every other call rig makes on a device that is
    not lost is {e counted}: {!stop} waits for none of them. {!stop} runs once,
    with no counted call inside, and after it only {!free}, {!unload},
    {!signaled} and holds' releases follow. A {!Fault} from a counted call, and
    a failed hand-over or commit, lose the device.

    {b What rig guarantees}, so that a driver checks none of it: it frees each
    region once, on its device; it calls {!peer} and {!map_peer} with two
    distinct devices of the driver; it unloads each image once, and calls
    {!entry} only on a loaded image; it allocates [Pinned] memory where [Mapped]
    answers [None]; and it bounds hangs itself ([hang_ms]), so {!sleep} never
    times out work. *)
module type Driver = sig
  type t
  (** The type for open devices of the driver. *)

  type region
  (** The type for memory of a device. *)

  type image
  (** The type for loaded code. *)

  exception Fault of string
  (** [Fault why] reports that the device faulted. Any counted call may raise
      it. *)

  val key : t Type.Id.t
  (** [key] identifies the driver: two devices of one driver map each other's
      memory ({!map_peer}). *)

  val facts : t -> region facts
  (** [facts d] is [d]'s facts. *)

  (** {1:memory Memory} *)

  val alloc : t -> memory -> int -> region option
  (** [alloc d m n] is [n] bytes of [d]'s memory [m], or [None] if [d] has
      none. [n] is positive. Counted. *)

  val free : t -> region -> unit
  (** [free d r] gives back [r], a region {!alloc} made, a mapping {!map_peer}
      or {!map_host} made, or [d]'s word. rig frees [r] once no work of [d]
      that uses it can run; it may free after {!stop}, and frees the word only
      after it. *)

  val locate : region -> location
  (** [locate r] is where [r] lies. *)

  val peer : t -> t -> bool
  (** [peer d d'] is [true] iff {!map_peer}[ d d'] maps [Device] memory of
      [d']. It answers from what [d] and [d'] learned when they opened, also
      once either is lost. *)

  val map_peer : t -> t -> region -> region option
  (** [map_peer d d' r] is a region of [d] over [r], any memory of [d'], or
      [None]. Counted. *)

  val map_host : t -> int -> int -> region option
  (** [map_host d p n] is a region of [d] over the [n] bytes of host memory at
      [p], or [None] if [d] does not map these pages, such as read-only ones.
      [p] starts a page and [n] is positive; it is called only where [d]
      [maps_host]. The memory stays mapped until the region is freed ({!free}).
      Counted. *)

  (** {1:code Code} *)

  val image : t -> string -> ((region, image) code, string) result
  (** [image d b] takes the binary [b], or is [Error why] if [d] refuses it.
      Counted. *)

  val entry : image -> string -> entry option
  (** [entry i f] is the driver's name for [i]'s function [f], or [None] if [i]
      has no function [f]. It makes what a launch of [f] needs, such as a
      compiled pipeline or [f]'s scratch memory, so a hand-over that launches
      [f] makes nothing; a second call for [f] makes nothing new. That first
      call may block. Counted.

      Raises [Invalid_argument] with the driver's reason if [f] needs more than
      the device offers, where the driver finds that only here. *)

  val unload : t -> image -> unit
  (** [unload d i] releases what {!image} and {!entry} made for [i], once no
      work of [d] that runs it can run. The region of a [Place] is not [i]'s:
      rig frees it after. Counted. *)

  (** {1:timeline Timeline} *)

  val signaled : t -> int
  (** [signaled d] is the value in [d]'s word, read with acquire order. rig
      calls it only for a word with no host address, behind a transport, and
      may call it after {!stop}. Counted. *)

  val sleep : t -> seen:int -> still_ms:int -> unit
  (** [sleep d ~seen ~still_ms] blocks on the device's events until the word
      differs from [seen], returning at once if it already does, and returns at
      the latest after [still_ms] milliseconds. It may return earlier, on an
      event of other work or a timer of the driver's own: rig reads the word
      again after every return and sleeps again while it is unmoved. It raises
      {!Fault} once the device faulted. It may run beside the hand-over and the
      commit. Counted. *)

  (** {1:stopping Stopping} *)

  val stop : t -> fault:string option -> unit
  (** [stop d ~fault] stops the lost [d] without waiting. [fault] is the reason
      rig lost [d] for, where it lost it for a {!Fault} of a counted call or for
      its hang bound ([hang_ms]), and [None] for any other loss. The driver
      writes the last value its hand-over received into the word, with release
      order, once no work of [d] runs: before [stop] returns if none does,
      otherwise through the queues' releases or its own drain. Work [d] encoded
      that runs only once committed, such as an open command buffer, is
      dropped. [stop] releases nothing rig made: rig frees each region and
      unloads each image itself, also after [stop]. After [stop] rig calls only
      {!free}, {!unload} and {!signaled}, and calls {!free} and {!unload} only
      once the word reads that last value, uncounted, dropping their faults.
      {!free} and {!unload} may come after an open of the same hardware made a
      new device of this driver; they touch nothing of the new device. *)
end

(** Devices of memory reached by reading and writing.

    An io device holds memory the host does not address and no queue runs,
    such as files. Its reads and writes are synchronous, in the caller. *)
module type Io = sig
  type t
  (** The type for open io devices. *)

  type region
  (** The type for memory of an io device. *)

  exception Fault of string
  (** [Fault why] reports that the device failed, such as a closed connection.
  *)

  val region_key : region Type.Id.t
  (** [region_key] identifies the io library and its regions: [Rig.Buffer.of_io]
      and [Rig.Buffer.io] cast regions by it, and a device's name stays with the
      library that opened it. *)

  val budget : t -> int
  (** [budget d] is the bytes [d] should hold at most. *)

  val alloc : t -> int -> region option
  (** [alloc d n] is [n] new bytes of [d]'s memory, or [None] if [d] has not the
      room. [n] is positive.

      Raises [Invalid_argument] if [d] makes no memory of its own, which
      [Rig.Buffer.create] raises in turn. *)

  val free : t -> region -> unit
  (** [free d r] gives [r] back. *)

  val read : t -> region -> at:int -> dst:int -> len:int -> unit
  (** [read d r ~at ~dst ~len] reads the [len] bytes at [at] in [r] into host
      memory at [dst]. A failure of the memory alone, such as a file truncated
      since it was opened, raises [Sys_error], which loses nothing; {!Fault}
      loses [d]. *)

  val write : t -> region -> at:int -> src:int -> len:int -> unit
  (** [write d r ~at ~src ~len] writes the [len] bytes of host memory at [src]
      at [at] in [r]. It raises as {!read}. *)

  val pages :
    t ->
    region ->
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    option
  (** [pages d r] is [r]'s bytes as host memory, or [None] if [d] maps none. The
      mapping is [r]'s memory: {!read} and {!write} see writes through it, and
      it sees theirs. For [r] that [d] holds for reading ([Rig.Buffer.access]),
      no write through the mapping reaches [r].

      A device other than the host maps the pages where its driver maps host
      memory ({!Driver.map_host}), and on Linux every such driver maps it for
      writing. NVIDIA's kernel driver, for CUDA and NV, pins the pages for
      writing for as long as they are mapped (its os-mlock.c), which the kernel
      refuses for a shared mapping of a file (mm/gup.c,
      [writable_file_mapping_allowed]): those devices map no file that [d] holds
      for writing, and its copies go through the staging memory. AMD's KFD
      faults the pages in for writing, and refuses pages mapped read-only
      (amdgpu_hmm.c), so a region [d] holds for reading is mapped as the
      process's own copy, as a private mapping of a file is.

      [None] is a fact of [r]: rig asks once per memory, at its first
      borrow that answers, and keeps the answer. A failure that may pass, of the
      memory alone, such as too many open files, raises [Sys_error], which loses
      nothing: the borrow raises it, and the next asks again. {!Fault} loses
      [d]. *)

  val prefetch : t -> region -> at:int -> len:int -> unit
  (** [prefetch d r ~at ~len] asks [d] to read the [len] bytes at [at] in [r]
      ahead, before a device other than the host reaches them through {!pages}.
      It is a hint: it raises nothing. *)

  val stop : t -> unit
  (** [stop d] ends [d] once it is lost or closed. *)
end
