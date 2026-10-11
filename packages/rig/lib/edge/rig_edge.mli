(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The contract between rig and the drivers that run its devices.

    A {e driver} is the library that runs one kind of hardware for rig. It
    allocates a device's memory, loads code on it and places work on its
    queues. A vendor library implements {!Driver} and supplies an {!opener},
    which a program passes to [Rig.open_]. The driver, and its ABI library
    that declares the record its compiled code reads ({!type-capability}),
    link this library and never [rig]. [rig], which every program that opens
    a device links, defines the C function [rig_code_sync] that
    [rig_edge.h] declares. A library for a device whose memory the host
    reads and writes through calls, such as files, implements {!Io}
    instead.

    rig reads a device's {!type-facts} once, when the device opens. It then
    calls the driver's OCaml functions to allocate and map memory and to load
    code, and three C functions of the driver to run work
    ({!section-calls}). [room] returns whether a submission's parts fit the
    device's queues, [submit] places them on the queues as the work of the
    next value of the device's timeline, and [commit] lets that work run.

    A device reports progress in its {e timeline word}, eight bytes of memory
    that hold a value of the device's timeline. [Rig]'s preamble and Timelines
    section define values, reaching and committing. While the device is live,
    the word holds the last committed value [v] such that the work of every
    value up to [v] has completed. Only the driver writes the word, and the
    word never moves backwards. rig reads it to learn which values the device
    has reached, and other devices' queues may wait on it. After a loss the
    word may go on to show dropped values ([Rig]'s Loss section).

    The life of a device, as its driver sees it, goes like this:
    {ol
    {- [Rig.open_] calls the opener's [make] and reads the new device's
       facts.}
    {- rig allocates, maps and frees memory, loads and unloads code, and
       hands work to the device through the C functions.}
    {- rig loses the device for a {!Driver.Fault}, a failed [submit] or
       [commit], a hang, a close, a failure of the process, a fork, or the
       loss of a device whose value its queue waits for.}
    {- rig calls {!Driver.stop} once no call of the driver is in flight on
       the device ({!Driver}). Once the word shows the device's last
       submitted value, the last value rig gave the driver's [submit], rig
       frees the device's remaining memory and unloads its images, and it
       frees the word once nothing reads it.}}

    Each function's doc says what it does, and a driver's implementation
    must do that. Further obligations say "must". *)

(** {1:facts Facts}

    A device's {e facts} are what rig reads of it once, when it opens
    ({!type-facts}). *)

(** The type for the kinds of a device's memory. *)
type memory =
  | Device  (** The device's own memory. *)
  | Pinned
      (** Host memory that both the device's work and the host address,
          page-locked where the host pages memory out. *)
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
      (** The queue's name, such as ["COMPUTE:0"]: its lane in a profile. *)
  runs : kind list;  (** The kinds of work the queue runs. *)
}
(** The type for a device's queues. rig places each part of a submission on
    one of them, and orders it after the parts of other queues it follows
    ({!section-calls}):
    {ul
    {- A copy goes on the queue of the last earlier part of its submission
       that it follows, where that queue runs [Copy], and otherwise on the
       device's {e copy engine}, the queue its facts name in [copy_engine]
       ({!type-facts}), where rig's own copies go.}
    {- Words go on the first queue whose [runs] lists both [Launch] and
       [Words], and otherwise on the first that lists [Words].}
    {- A launch or a fill goes on the first queue whose [runs] lists its
       kind.}} *)

(** The type for how a device's word advances. *)
type completion =
  | Store  (** A store that the driver places on a queue writes the word. *)
  | Host
      (** The host writes the word, from the driver's handler or before the
          driver's [submit] or [commit] returns ({!section-calls}). *)
  | Object of nativeint
      (** An object of the driver's API completes, and the driver then writes
          the word. The object must fit in 62 bits, since a wait passes it as
          an integer. *)

(** The type for the producers a device's queues wait for. A {e producer}
    is a device whose value another device's work waits for. rig waits on the
    host for the producers a device's queues do not wait for, and for those
    beyond [most]. *)
type waits = {
  stores : bool;  (** Its queues wait for a [Store] producer. *)
  hosts : bool;  (** Its queues wait for a [Host] producer. *)
  objects : bool;
      (** Its queues wait for an [Object] producer of the same driver, whose
          objects they know. *)
  most : int;  (** The most waits one submission carries in its queues. *)
}

(** The type for what compiled code needs from a device, a record that the
    driver's ABI library declares under its key. *)
type capability = Capability : 'a Type.Id.t * 'a -> capability

type 'region facts = {
  arch : string;  (** The device's architecture, such as ["gfx1100"]. *)
  budget : int;  (** The bytes of own memory the device should hold at most. *)
  queues : queue list;  (** The device's queues. *)
  copy_engine : int option;
      (** [Some i] if the device's copy engine is the queue at index [i] of
          [queues], and [None] if it has none. A copy engine other than the
          first queue lets copies run beside the first queue's work. *)
  completion : completion;  (** How the word advances. *)
  waits : waits;  (** The producers its queues wait for. *)
  may_block : bool;
      (** Whether [room], [submit] and [commit] ({!section-calls}) may block
          on the device's own earlier work, its own transfers or its
          library's back-pressure. Where they may, rig counts them
          ({!Driver}), so {!Driver.stop} never runs while one blocks. *)
  hang_ms : int option;
      (** [Some n] if rig loses the device once its word stays below a
          committed value for [n] milliseconds, for a device whose work
          nothing else bounds, and [None] where the vendor's driver bounds it.
          [n] must be at least [1]. The clock starts at a wait, so the loss
          comes [n] ms or more after the work stalled. [Rig]'s Hangs section
          states when the clock starts again. rig passes the loss's reason,
          ["no progress for n ms"], to {!Driver.stop}. *)
  maps_host : bool;
      (** Whether the device maps host memory of its machine that starts on a
          page ({!Driver.map_host}). *)
  host_addresses : bool;
      (** Whether the host addresses every region the device allocates, with
          a [host] address in each one's location ({!Driver.locate}). The
          host then reaches the device's memory and copies it, and the
          device's [Pinned] memory is its own and counts in [budget]. rig
          frees a region that breaks this, and the allocation raises
          [Invalid_argument]. *)
  capability : capability;
      (** The ABI record, filled when the device opened. *)
  word : 'region;
      (** The timeline word. Other devices may map it, and rig reads it after
          a loss. rig frees it ({!Driver.free}) after {!Driver.stop}, once
          nothing reads it. Where it has no host address, as for a device of
          another machine, rig reads it through {!Driver.signaled}. *)
  edge : nativeint;
      (** The address of the device's C state, valid until rig frees [word].
          The state's first member must point to the driver's
          [struct rig_driver] of [rig_edge.h] ({!section-calls}). *)
}
(** The type for the facts of a device. A {!Driver.Fault} while rig reads
    them makes the open return [Error]. The open raises [Invalid_argument]
    if [copy_engine] names no queue of [queues] or one whose [runs] lacks
    [Copy], [hang_ms] is below [1], [edge] is [0n], or the state's first
    member is [NULL]. rig checks nothing else of [edge]. *)

(** {1:calls The C functions}

    A driver's [struct rig_driver] holds three C functions, which rig calls
    with the facts' [edge]. They return the constants of [rig_edge.h].
    {ul
    {- [room] returns [RIG_FITS] if the parts fit the device's queues now,
       and [RIG_LATER] if they fit once one of its values is reached. It
       returns [RIG_NEVER] for parts that exceed the device's empty queues,
       name work it does not run or have kind [0], and for a launch whose
       grid or group is empty or exceeds the device's or the function's
       limits, or whose shared memory exceeds the function's.}
    {- [submit] encodes the parts on the device's queues as the work of the
       value it is given, after the waits it is given and the device's earlier
       work. Its calls are given the values 1, 2, 3, … in order. Each part
       names its queue, by its index in [queues], and the parts of other
       queues it starts after ([struct rig_part] in [rig_edge.h]). When it
       returns, all of the value's work is on the queues, though the device
       may hold it until it is committed. It returns [RIG_COMMITTED] if every
       value up to its own is committed, [RIG_OK] if its value is only
       encoded, and [RIG_FAILED] if it failed, which loses the device. While a
       profile is taken, it must write when the value's work started and ended
       on the device, on the host's clock, before the word shows the value, or
       leave both times [0] for work it cannot time.}
    {- [commit] is given a value [v] no greater than the last value given to
       [submit], and lets the work up to [v] run. Once that work has
       completed, the device writes [v] or a later value into the word.
       Committing a committed value must do nothing. It returns [RIG_OK], or
       [RIG_FAILED], which loses the device.}}

    rig calls the three one at a time for each device, with [room] before
    each [submit]. They must call no function of the OCaml runtime and read
    no OCaml value, since rig may call them with the runtime released.

    Each submission holds [RIG_SCRATCH] bytes of {e scratch} for each of its
    parts, which rig hands to [room] and [submit] with the parts. Either may
    write the scratch and read it back during the call, such as to keep
    which of the device's engines runs each part, so that neither
    allocates. No other call uses the scratch meanwhile, and a call finds
    what an earlier call left there. A copy part's fields may differ from
    one call to the next, since a copy's ends may be a run's buffers
    ([Rig.Submission.side]). A driver must decide again on each call
    what it derives from them.

    A driver whose [submit] always returns [RIG_COMMITTED] is never asked
    to commit. Every other driver must commit on its own at least once every
    [k] values it receives, so that the word shows each value at most [k]
    values late while work is submitted. [k] is the driver's own constant,
    which rig does not read, and the driver's interface states it.

    A driver must keep three properties:
    {ul
    {- {e Prefix order}: the work handed to each queue runs in the order it
       was handed.}
    {- {e Device order}: a value's work starts after the previous value's
       work has completed, on every queue. The driver's [submit] meets it by
       making its work wait for the queue that completed the previous
       value.}
    {- {e Observable completion}: the host learns, through the word, that a
       prefix of the values has completed, or learns that the device
       faulted.}} *)

(** {1:memory Memory} *)

type location = {
  address : int option;
      (** The region's address as the device's work addresses it, or [None]
          for memory named by its handle only. *)
  host : int option;
      (** The host address of the region's first byte, if the host addresses
          it. *)
  handle : nativeint;
      (** The driver's object for the region, such as a buffer object, a
          [CUdeviceptr] or an [MTLBuffer]. *)
}
(** The type for where a region lies. *)

(** {1:code Code} *)

(** The type for a binary a driver took. *)
type ('region, 'image) code =
  | Loaded of 'image  (** The driver's library placed the code itself. *)
  | Place of int * ('region -> 'image * string)
      (** The code goes into [n] bytes of the device's [Device] memory. rig
          allocates a region of at least [n] bytes and calls the function
          with it, and the function returns the image and the bytes, at most
          [n], that rig places at the region's start. The driver must make
          nothing on the device before the function is called, so a [Place]
          whose function is never called leaves nothing to release. The
          function must call nothing that may block or fault, and must raise
          nothing. *)

type entry = {
  code : int;
      (** What compiled code names the function by, such as a kernel
          descriptor's address, a code address, a [CUfunction] or a
          pipeline. *)
  launch : nativeint;
      (** What the driver's [submit] reads to launch the function
          ([launch.launch] in [rig_edge.h]), such as its dispatch template and
          limits, valid while the image is loaded; [0n] for a driver whose
          queues run no [Launch]. *)
  params : int option;
      (** The number of parameter bytes, from byte [0], that the function
          reads, or [None] if the driver does not know it. rig refuses a
          launch that gives fewer, or that places a ref past them. *)
  threads : int option;
      (** The most threads a group of the function may have, or [None] if
          the driver does not know it. rig refuses a run that sets more. *)
}
(** The type for a function of an image, as a driver names it. *)

(** The type for a driver's answer when asked for a function of an image. *)
type entry_answer =
  | Entry of entry  (** The function. *)
  | Absent  (** The image has no function of that name. *)
  | No_memory of int
      (** The device has no memory for the bytes the function's launches
          need, such as its scratch. rig reclaims memory and asks again
          ({!Driver.val-entry}). *)

(** {1:drivers Drivers} *)

(** Drivers.

    rig counts the calls in flight on each device that is not lost, and
    calls {!Driver.val-stop} once, when none is. Every call of the driver's
    OCaml functions is {e counted} except {!Driver.locate} and
    {!Driver.peer}, which must call nothing that may block or fault, and
    which rig calls at any time. Calls of the C functions are counted where
    the device declares [may_block]. After [stop], rig calls only
    {!Driver.val-free}, {!Driver.unload} and {!Driver.signaled}, and lets go
    of the values the device's submissions keep ([Rig.Submission.make]). A
    {!Driver.Fault} from a counted call, and a failed [submit] or [commit] of
    the driver, lose the device.

    rig guarantees these properties, so that a driver checks none of them:
    {ul
    {- it frees each region once, on its device;}
    {- it calls {!Driver.peer} and {!Driver.val-map_peer} with two distinct
       devices of the driver;}
    {- it unloads each image once, and calls {!Driver.val-entry} only on a
       loaded image;}
    {- it allocates [Pinned] memory where [alloc] of [Mapped] returns
       [None];}
    {- it bounds hangs itself ([hang_ms]), so {!Driver.sleep} never times out
       work.}} *)
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
  (** [key] identifies the driver. rig asks {!val-map_peer} only of two
      devices with the same key. *)

  val facts : t -> region facts
  (** [facts d] is [d]'s facts. *)

  (** {1:driver_memory Memory} *)

  val alloc : t -> memory -> int -> region option
  (** [alloc d m n] is [n] bytes of [d]'s memory [m], or [None] if [d] has
      none. [n] is positive. Counted. *)

  val free : t -> region -> unit
  (** [free d r] gives back [r], which is a region {!val-alloc} made, a
      mapping {!val-map_peer} or {!val-map_host} made, or [d]'s word. rig
      frees [r] once no work of [d] that uses it can run. It may free after
      {!val-stop}, and frees the word only after it. *)

  val locate : region -> location
  (** [locate r] is where [r] lies. *)

  val peer : t -> t -> bool
  (** [peer d d'] is [true] iff {!val-map_peer}[ d d'] maps [Device] memory of
      [d']. It must return its result from what [d] and [d'] learned when they
      opened, also once either is lost. *)

  val map_peer : t -> t -> region -> region option
  (** [map_peer d d' r] is a region of [d] over [r], any memory of [d'], or
      [None]. [d]'s GPU may already map [r]'s memory, through [r] or another
      region. Each region must keep the memory mapped until rig frees it.
      Counted. *)

  val map_host : t -> int -> int -> region option
  (** [map_host d p n] is a region of [d] over the [n] bytes of host memory at
      [p], or [None] if [d] does not map these pages, such as read-only ones.
      [p] starts a page and [n] is positive. rig calls it only where [d]
      declares [maps_host]. The memory must stay mapped until the region is
      freed ({!val-free}). Counted. *)

  (** {1:driver_code Code} *)

  val image : t -> string -> ((region, image) code, string) result
  (** [image d b] takes the binary [b], or is [Error why] if [d] refuses it.
      Counted. *)

  val entry : image -> string -> entry_answer
  (** [entry i f] is [Entry e] with what the driver needs to launch [i]'s
      function [f], or [Absent] if [i] has no function [f]. It must make what
      a launch of [f] needs, such as a compiled pipeline or [f]'s scratch
      memory, so that the driver's [submit] allocates nothing to launch [f].
      A second call for [f] must make nothing new. The first call may block.
      Counted.

      It is [No_memory n], having made nothing, if the device has no memory
      for the [n] bytes [f]'s launches need. rig then runs the rounds of
      reclamation an allocation does, asks again after each, and raises
      [Rig.Out_of_memory] after the last.

      Raises [Invalid_argument] with the driver's reason if [f] needs more
      than the device offers, where the driver finds that only here. *)

  val unload : t -> image -> unit
  (** [unload d i] releases what {!val-image} and {!val-entry} made for [i].
      rig calls it once no work of [d] that runs [i] can run. The region of a
      [Place] is not [i]'s, and rig frees it after. Counted. *)

  (** {1:timeline Timeline} *)

  val signaled : t -> int
  (** [signaled d] is the value in [d]'s word, read with acquire order. rig
      calls it only for a word with no host address, and may call it after
      {!val-stop}. Counted. *)

  val sleep : t -> seen:int -> still_ms:int -> unit
  (** [sleep d ~seen ~still_ms] blocks on the device's events until the word
      differs from [seen], and must return at the latest after [still_ms]
      milliseconds. It must return immediately if the word already differs.
      It may return earlier, on an event of other work or a timer of the
      driver's own, since rig reads the word again after every return and
      sleeps again while it is unmoved. rig may call it while the driver's
      [submit] or [commit] runs. Counted.

      Raises {!Fault} once the device has faulted. *)

  (** {1:stopping Stopping} *)

  val stop : t -> fault:string option -> unit
  (** [stop d ~fault] stops the lost [d] and must not wait for its work.
      [fault] is [Some why] if rig lost [d] for a {!Fault} of a counted call
      or for its hang bound ([hang_ms]), and [None] otherwise.
      {ul
      {- Once no work of [d] runs, the driver must write the last value
         given to its [submit] into the word, with release order, also when
         that [submit] failed. If no work
         runs, it must write it before [stop] returns, and otherwise through
         the queues' releases or by itself once that work has finished.}
      {- The driver must drop the work [d] encoded that runs only once
         committed, such as an open command buffer.}
      {- [stop] must release nothing rig made. rig frees each region and
         unloads each image itself, also after [stop].}}
      After [stop], rig calls {!val-free} and {!unload} only once the word
      reads that last value. It calls them uncounted and drops their faults.
      They may come after an open of the same hardware has made a new device
      of this driver, and must touch nothing of the new device. *)
end

(** Devices of memory that the host reads and writes through calls.

    An io device holds memory that the host does not address, such as a
    file, and runs no work. Its reads and writes are synchronous, in the
    caller. *)
module type Io = sig
  type t
  (** The type for open io devices. *)

  type region
  (** The type for memory of an io device. *)

  exception Fault of string
  (** [Fault why] reports that the device failed, such as a closed
      connection. *)

  val region_key : region Type.Id.t
  (** [region_key] identifies the io library and its regions.
      [Rig.Buffer.of_io] and [Rig.Buffer.io] cast regions by it, and a
      device's name stays with the library that opened it. *)

  val budget : t -> int
  (** [budget d] is the bytes [d] should hold at most. *)

  val alloc : t -> int -> region option
  (** [alloc d n] is [n] new bytes of [d]'s memory, or [None] if [d] has no
      room. [n] is positive.

      Raises [Invalid_argument] if [d] makes no memory of its own, which
      [Rig.Buffer.create] raises in turn. *)

  val free : t -> region -> unit
  (** [free d r] gives [r] back. *)

  val read : t -> region -> at:int -> dst:int -> len:int -> unit
  (** [read d r ~at ~dst ~len] reads the [len] bytes at [at] in [r] into host
      memory at [dst].

      Raises [Sys_error] for a failure of the memory alone, such as a file
      truncated since it was opened, which loses nothing, and {!Fault}, which
      loses [d]. *)

  val write : t -> region -> at:int -> src:int -> len:int -> unit
  (** [write d r ~at ~src ~len] writes the [len] bytes of host memory at
      [src] at [at] in [r].

      Raises as {!read}. *)

  val pages :
    t ->
    region ->
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    option
  (** [pages d r] is [r]'s bytes as host memory, or [None] if [d] maps none.
      The mapping is [r]'s memory, so {!read} and {!write} see writes through
      it, and it sees theirs. For [r] that [d] holds for reading
      ([Rig.Buffer.access]), a write through the mapping must not reach [r].
      rig asks once per allocation, at its first borrow that gets a result,
      and keeps the result.

      A device other than the host maps the pages where its driver maps host
      memory ({!Driver.map_host}). Some drivers refuse pages mapped
      read-only, so a library maps a region it holds for reading as the
      process's own copy, as a private mapping of a file is. Some refuse a
      shared mapping of a file, and their devices then copy through staging
      memory.

      Raises [Sys_error] for a transient failure of the memory alone, such as
      too many open files, which loses nothing. The borrow raises it,
      and the next borrow asks again. Raises {!Fault}, which loses [d]. *)

  val prefetch : t -> region -> at:int -> len:int -> unit
  (** [prefetch d r ~at ~len] asks [d] to read the [len] bytes at [at] in [r]
      ahead, before a device other than the host reaches them through
      {!pages}. It is a hint, and must raise nothing. *)

  val stop : t -> unit
  (** [stop d] ends [d]. rig calls it once [d] is lost or closed. *)
end

(** {1:opening Opening} *)

type 'a opener = {
  name : string;  (** The device's name, without a machine suffix. *)
  driver : (module Driver with type t = 'a);  (** The device's driver. *)
  make : unit -> ('a, string) result;  (** Opens a new device handle. *)
}
(** The type for openers of named devices. Making an opener does no I/O and
    opens nothing. [make] opens the device, and returns [Error] when the
    device cannot be opened, such as when the hardware is missing. [Rig.open_]
    calls [make] only when no device of [name] is open on its machine. A success
    hands the handle to rig, which stops it ({!Driver.stop}) once the device is
    lost or closed. A caller that calls [make] itself owns the handle and must
    stop it. Each driver's opener states whether [make] opens another handle
    while an earlier one is not stopped. *)

type 'a io_opener = {
  name : string;  (** The device's name, without a machine suffix. *)
  io : (module Io with type t = 'a);  (** The device's io library. *)
  make : unit -> ('a, string) result;  (** Opens a new io device handle. *)
}
(** The type for openers of named io devices. [Rig.open_io] calls [make] and
    owns its handles as [Rig.open_] does for an {!opener}. A caller that calls
    [make] itself owns the handle and must call its [stop] ({!Io}). *)
