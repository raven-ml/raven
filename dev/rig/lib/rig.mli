(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, their memory, and the order of work across them.

    A {e device} is memory and, for most, a processor that runs work on it: the
    {!host}, a GPU that a {e driver} opened ({!Driver}), or a store of bytes the
    host reads and writes, which runs no work ({!Io}). A device's work is one
    sequence of {e submissions}, numbered [1], [2], … : the {e values} of its
    {e timeline}. Value [v] is {e reached} once every submission up to [v]
    completed. A {e point} ({!Point}) names a device and a value.

    Memory returns by the timeline. Each memory records the point of its last
    write and, per device, the point of its last use: its {e stamps}. Work that
    reads memory waits for its last write; work that writes it, and a reuse of
    the memory, wait for every use. A wait on the work of the device itself is
    the device's own order; a wait on another device's work is a {e foreign}
    wait, which the device's queue makes itself where its driver can, and the
    host makes before the submission otherwise.

    {v
      Buffer.t ──view, borrow──> memory ──stamps──> points
          │                                            ▲
          │ submit ~reads ~writes, make ~fixed         │ submit
          ▼                                            │
      Submission.t ─────────── one device ── facts.edge (C)
    v}

    A program opens devices ({!open_}), allocates {!Buffer}s, and copies between
    them ({!Buffer.copy}), which waits for the work that touched them. A library
    that runs compiled work on devices prepares {!Submission}s once and submits
    each run's buffers with them ({!submit}); it orders its own host access with
    {!Buffer.wait}. A vendor library links only [rig.edge] and matches
    {!Rig_edge.Driver}; a program passes it to {!open_} with the library's
    function that opens the hardware.

    {1:domains Domains}

    Every function may be called from any domain, at the same time as others. No
    function but {!open_} holds a lock while it waits: {!open_} holds its name's
    lock while the opener runs, so opens of one name run one at a time and opens
    of other names go on. A copy through staging memory ({!Buffer.copy}) holds
    part of it while it waits for devices, and a copy that finds it all held
    waits. A wait releases the domain lock while it blocks and returns to OCaml
    at least every 200 ms, so a pending [Sys.Break] from Ctrl-C raises there;
    an interrupted wait loses no device and assigns no value.

    {1:loss Loss}

    A device whose driver reports a fault, whose hand-over or commit fails, or
    whose word stays below a committed value past its driver's hang bound while
    a wait of this library watches it ([hang_ms] in {!Rig_edge.facts}), is
    {e lost}, once and for good ({!Lost}); so is a device {!close} ended, and
    every device of a process that {!fail}ed. Work that waits in its queue on a
    lost device's unreached values is lost with it. Every later use of the
    device, of its memory, and of other memory that waits for a point the device
    did not reach raises {!Lost}; memory whose points it reached is ordinary
    memory again, and other devices go on. A lost device's facts still answer:
    {!name}, {!arch}, {!budget} and {!capability} as before, {!submitted} the
    last value handed over, and {!signaled} the last value its word showed,
    which may stop moving. A lost device's memory returns only once its timeline
    word shows its last submitted value reached, which it may never do: such
    memory is kept for the life of the process. Opening the device's name again
    makes a new device.

    {1:reclaim Reclamation}

    Nothing frees a buffer by hand. Once a buffer and its views are unreachable,
    the collector hands its memory back to its device, even while the domain
    that made them is blocked. The device keeps it in a cache for reuse once
    every other device's last use of it is reached, and gives it back to its
    driver once its own work is done too. A {e drain} of a device takes back the
    memory of its buffers collected since, and runs the release of every hold
    that became due ({!Hold}). A device's memory and holds return as its word
    shows their points reached, which may be late by its driver's bound of
    values ({!submit}). {!Buffer.create}, {!Buffer.of_io} and
    {!Buffer.copy} drain the devices they use first. An allocation of more than
    the device's {!budget} raises {!Out_of_memory} at once and keeps the cache.
    Another that the budget or the driver refuses runs rounds of reclamation
    for the budget that refused (cached memory returned, work that holds memory
    back waited for, unreachable buffers collected) before it raises
    {!Out_of_memory}; a wait in a round raises {!Lost} for a lost device. A
    copy whose device's driver refuses to map the staging memory runs the same
    rounds for that device. Buffer memory paces the collector, so unreachable
    buffers hold a bounded share of the program's memory. *)

(** {1:devices Devices} *)

type t
(** The type for devices. A device stays the same value while it is open; an
    open of a device's name after its loss makes a new value, unequal to the
    lost one. *)

val host : t
(** [host] is this process's host, named ["CPU"]. Its memory is the process's
    heap; its work is the process's own code, which needs no submission. *)

val name : t -> string
(** [name d] is [d]'s name: the name its opener gave it, followed by
    ["@MACHINE"] for a device of another machine. It is for people; nothing
    parses it. *)

val host_of : t -> t
(** [host_of d] is the host of [d]'s machine: {!host} for this machine's
    devices, the device {!open_host} opened for another machine's. A host is its
    own. A loss of another machine's host loses none of its devices. *)

val arch : t -> string
(** [arch d] is the architecture of [d]'s processor, as its driver names it,
    such as ["gfx1100"], ["sm_89"] or ["Apple7"]; the machine's instruction set
    for a host, ["arm64"] or ["x86_64"]; and [""] for an {!Io} device. *)

val computes : t -> bool
(** [computes d] is [true] iff [d] runs work: a host, or a device a driver
    opened. It is [false] for an {!Io} device, whose memory is only read and
    written. *)

val runs_on_host : t -> bool
(** [runs_on_host d] is [true] iff [d]'s work is this process's code: [d] is
    {!host}, or a device {!memory_device} opened. The host computes on such a
    device's memory as on its own. *)

val shares_host_memory : t -> bool
(** [shares_host_memory d] is [true] iff this process addresses [d]'s memory as
    host memory: [d] and {!host} reach each other's memory,
    [reaches host d && reaches d host]. It holds for {!host}, a
    {!memory_device}, and a driver's device of this machine that runs no copy,
    such as Metal's; it is [false] for every device of another machine, that
    machine's host included. *)

val reaches : t -> t -> bool
(** [reaches d d'] is [true] iff [d]'s work addresses [d']'s own memory
    ({!Buffer.Device}) once [d] borrows it ({!Buffer.borrow}), as a compiler
    that places copies must know before any buffer exists. A device reaches its
    own memory. Of devices of one machine:
    - {!host} reaches the memory of a {!memory_device} and of a driver's device
      that runs no copy ({!queues}): that memory is the host's;
    - a driver's device reaches {!host}'s memory where it maps host memory
      (its [maps_host] fact, {!Rig_edge.facts}), a memory device's, and that
      of a device of its own
      driver that it maps ({!Rig_edge.Driver.peer}). Another machine's host is a
      driver's device of that machine, and reaches and is reached by this rule.

    Otherwise it is [false]: across machines, and between an {!Io} device and
    any other device. *)

val capability : t -> 'a Type.Id.t -> 'a option
(** [capability d k] is [Some c] if [d]'s driver declares its capability record
    under [k] ({!Rig_edge.capability}), [c] being the record it filled when
    [d] opened, and [None] otherwise. *)

(** The type for the work of a part. *)
type kind = Rig_edge.kind =
  | Words  (** 32-bit words placed on the queue: {!Submission.Words}. *)
  | Fill  (** A C function called on the queue: {!Submission.Fill}. *)
  | Copy  (** A copy between memory: {!Submission.Copy}. *)
  | Launch  (** A function run over a grid: {!Submission.Launch}. *)

type queue = Rig_edge.queue = {
  name : string;  (** The name parts give, such as ["COPY:0"]. *)
  runs : kind list;  (** The work its parts may be. *)
}
(** The type for a device's queues. *)

val queues : t -> queue list
(** [queues d] is [d]'s queues, in its driver's order, and what each runs; [[]]
    for {!host} and an {!Io} device. A part names its queue by name
    ({!Submission.part}). A queue named ["COPY:i"] runs [Copy], and [d]'s own
    copies go to the first one. *)

val equal : t -> t -> bool
(** [equal d d'] is [true] iff [d] and [d'] are the same device. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a device's {!name}. *)

(** {2:budgets Budgets} *)

val budget : t -> int
(** [budget d] is the most bytes [d] holds at once in the memory that counts in
    its budget ({!Buffer.memory}): live buffers, loaded images' code and its
    cache. It starts at [max_int] for {!host}, its driver's [budget] fact for a
    driver's device ({!Rig_edge.facts}) and {!Rig_edge.Io.budget} for an io
    device; {!set_budget} changes it. *)

val set_budget : t -> int -> unit
(** [set_budget d n] sets [d]'s budget to [n], returning cached memory
    ({!free_cache}) until [d] holds at most [n] bytes or its cache is empty.
    Live buffers are never released.

    Raises [Invalid_argument] if [n < 0]. *)

val free_cache : t -> unit
(** [free_cache d] returns [d]'s cached memory to its driver; for {!host}, the
    memory it keeps of collected buffers for reuse. *)

exception Out_of_memory of t * int
(** [Out_of_memory (d, n)] is raised when [d] cannot allocate [n] bytes, at once
    or after the rounds of reclamation {!reclaim} describes, [d] being the
    device asked, whichever budget refused ({!Buffer.Pinned}). *)

(** {2:lost Lost devices} *)

exception Lost of t * string
(** [Lost (d, why)] is raised by a use of the lost device [d]; by a use of [d]'s
    memory, which is what [d] allocated, through any buffer, and the borrows on
    [d] of other memory; and by a use of other memory that waits for a point of
    [d] that [d] did not reach. Memory whose points [d] reached is ordinary
    memory again. [why] is the driver's reason, ["closed"] for a device {!close}
    ended, or, for a device lost because its queue waited on another lost
    device, ["NAME lost"] with that device's name. It prints as
    ["NAME lost: why"]. *)

val lost : t -> string option
(** [lost d] is [Some why] if [d] is lost, [why] being what its {!Lost} carries,
    and [None] otherwise. It raises nothing and waits for nothing. *)

val close : t -> unit
(** [close d] ends [d]. It waits for the work submitted on [d] before the call,
    however long it runs, then ends [d] as a loss does, with the reason
    ["closed"], and returns once [d]'s driver stopped: an open of [d]'s name
    afterwards makes a new device. Then {!lost}[ d] is [Some "closed"], unless
    [d] was lost first, and uses of [d] and its memory raise {!Lost}. Memory of
    [d] that buffers still reach returns to its driver as they are collected.
    Work another domain submits on [d] during the close may be lost. A close is
    no failure of the process ({!failure}).

    On another machine's host ({!host_of}) it first closes each device of that
    machine, those opening included, whether or not the host is lost; opens of
    that machine's devices answer [Error] from then on ({!open_}). Otherwise, on
    a lost or closed [d] it only waits for [d]'s stop. An {!Io} device ends at
    once: its work is the caller's. A close a [Sys.Break] interrupted is
    finished by calling [close] again.

    Raises [Invalid_argument] if [d] is {!host}. *)

val fail : string -> unit
(** [fail why] fails the process: every device but {!host} that is not lost is
    lost with [why], as a fault loses a device, and every {!open_} and
    {!open_io} that starts afterwards answers [Error why] without calling its
    opener; one that races [fail] makes a device that is lost. Devices already
    lost keep their reason. It returns once each loss is recorded, without
    waiting for the stops. Only the first call acts, and the process stays
    failed. *)

val failure : unit -> string option
(** [failure ()] is the process's first failure, if any: the first loss of a
    device that no {!close} ended, as {!Lost} prints it, or [why] of {!fail} if
    that came first. It raises nothing and waits for nothing. *)

(** {1:timeline Timeline} *)

(** Points on timelines.

    A point is a device and a value of its timeline, held in one word: an
    immediate value that a store into an array or a record never boxes. *)
module Point : sig
  type device := t

  type t [@@immediate]
  (** The type for points. *)

  val device : t -> device
  (** [device p] is [p]'s device. *)

  val value : t -> int
  (** [value p] is [p]'s value, [1] or more. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a point as [NAME:v]. *)
end

val submitted : t -> int
(** [submitted d] is the last value assigned on [d]'s timeline, [0] before any:
    by {!submit}, or by a {!Buffer.copy} or {!Image.load} that [d] runs. It
    never decreases. *)

val signaled : t -> int
(** [signaled d] is the last value [d]'s timeline word showed reached. Work up
    to it completed, and its writes are visible to the caller. It first commits
    [d]'s work ({!submit}) unless another call holds [d]'s turn, so that called
    repeatedly it reaches every submitted value once its work completed. The
    commit may block as [d]'s hand-over does (its [may_block] fact,
    {!Rig_edge.facts}), and one that
    fails loses [d]. On a lost device it only reads the word.

    Raises {!Lost} if [d]'s word is read behind a transport and the read loses
    [d]. *)

val wait : t -> int -> unit
(** [wait d v] returns once [d] reached [v], first committing [d]'s work if
    [v] is not committed ({!submit}). For a driver whose host writes the word
    ([Host] {!Rig_edge.completion}), or whose word the host does not address,
    it blocks in the driver ({!Rig_edge.Driver.sleep}) from the first read of
    the word. For another, it spins on the word, then waits with the domain
    lock released, blocking in the driver between reads once the word stood
    still. It waits however long the work runs, unless [d]'s driver bounds
    hangs ({!section-loss}).

    Raises [Invalid_argument] if [v > submitted d], and {!Lost} if [d] is lost
    or is lost by the wait. *)

(** {1:buffers Buffers} *)

(** Buffers of device memory.

    A buffer is {!length} bytes in a range of one device's memory. What the
    bytes mean, elements of some type, is the caller's: this library moves,
    orders and returns bytes.

    A buffer is {e owned} when {!create} or {!of_string} made it, and
    {e borrowed} when it is over memory something else holds: a bigarray
    ({!of_bigarray}) or another device's memory ({!borrow}). A {!view} is as the
    buffer it views. *)
module Buffer : sig
  type device := t

  type t
  (** The type for buffers. *)

  (** The type for the memories of a device that {!create} allocates. *)
  type memory = Rig_edge.memory =
    | Device
        (** The device's own memory, which counts in the device's {!budget}. *)
    | Pinned
        (** Host memory that both the device's work and the host address,
            page-locked where the host pages: coherent, with no flush. It counts
            in the host's {!budget}, except on a device whose memory the host
            addresses, where it is the device's own memory and counts in the
            device's. *)
    | Mapped
        (** The device's own memory, which the host also addresses through a
            write-combined window, and which counts in the device's {!budget}. A
            host write is seen by work submitted after it; the host reads it
            slowly. Where the device has no such window, or the window or the
            budget cannot hold the buffer, it is [Pinned] memory, which keeps
            these promises; no cache is released for it. *)

  val create : ?memory:memory -> device -> int -> t
  (** [create d n] is an owned buffer of [n] bytes in [d]'s memory [memory]
      (defaults to [Device]), with unspecified contents. A buffer of no bytes
      allocates nothing. On a device whose memory the host addresses, every
      [memory] is the device's own. On {!host}, a buffer of 64 KiB or more (four
      pages, where pages are larger) starts on a page, so devices can {!borrow}
      it. On an {!Io} device it is memory the device's {!Rig_edge.Io.alloc}
      makes.

      Raises [Invalid_argument] if [n < 0] or [d] is an io device that makes no
      memory of its own; {!Out_of_memory}; and {!Lost} if [d] is lost. *)

  (** The type for accesses to memory. *)
  type access =
    | Read  (** Reading it. *)
    | Read_write  (** Reading and writing it. *)

  val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
  (** [of_io d k r ~access n] is a buffer over the [n] bytes of [r], a region of
      the io device [d] whose library declares [k] ({!Rig_edge.Io.region_key}),
      whose memory admits [access] ({!val-access}): [Read] for a region [d]
      holds for reading, such as a file opened read-only. It returns to [d]'s
      {!Rig_edge.Io.free} once unreachable. The memory is outside the claims
      ({!Claim}).

      Raises [Invalid_argument] if [d] is no io device, its library's key is not
      [k], or [n < 0]. *)

  val io : t -> 'r Type.Id.t -> 'r option
  (** [io b k] is the region [b]'s memory lies in, if it is io memory of a
      library whose key is [k], and [None] otherwise.

      Raises [Invalid_argument] if [b] is dead ({!Claim.consume}). *)

  val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
  (** [of_bigarray ba] is a borrowed buffer on {!host} over [ba]'s bytes,
      without a copy. It keeps [ba] reachable. The memory is outside the claims
      ({!Claim}). *)

  val of_string : string -> t
  (** [of_string s] is an owned buffer on {!host} holding [s]'s bytes, placed as
      {!create} places a host buffer. *)

  val borrow : device -> t -> t option
  (** [borrow d b] is [Some b'], a borrowed buffer on [d] over [b]'s memory,
      without a copy, of [b]'s length, or [None] where [d] cannot map it. It is
      [Some b] for [b] on [d]. [b'] keeps [b] reachable, and its stamps are
      [b]'s: work through [b'] is work on [b]'s memory.

      [d] maps host memory of its machine that starts on a page, where it maps
      host memory at all (its [maps_host] fact, {!Rig_edge.facts}), and memory
      of a device of its own driver that its driver maps
      ({!Rig_edge.Driver.map_peer}). Every borrow on [d] of one memory shares
      one mapping, made at the first borrow. It lasts while both the memory and
      [d] live, and is released by whichever ends first: with the memory, once
      [d]'s work submitted until then is done; or with [d], once [d] is lost or
      closed and its word shows its last submitted value, which a lost [d]'s
      may never do ({{!loss}Loss}). A borrow dropped and made again while both
      live maps nothing. Uses of a borrow on a lost [d] raise {!Lost}. A borrow
      of a borrow maps the memory the first one maps. A {!memory_device} maps
      any host memory. Host memory that does not start on
      a page, such as a host buffer of fewer than 64 KiB, borrows only on
      {!host} and memory devices. An io device's memory borrows through its
      pages ({!Rig_edge.Io.pages}), as host memory, where its device maps them;
      a device other than the host asks the io device to read the borrowed bytes
      ahead ({!Rig_edge.Io.prefetch}).

      Raises [Invalid_argument] if [b] is dead ({!Claim.consume}); {!Lost} if
      [d] is lost or is lost by the borrow, and for [b] as {!Lost} states; and
      [Sys_error] where asking an io device for [b]'s pages failed and may pass
      ({!Rig_edge.Io.pages}). *)

  val wait : t -> access -> unit
  (** [wait b access] returns once the work on [b]'s memory that an access of
      the host must follow is done: for [Read], the work of its last write; for
      [Read_write], the work of every use, its own device's included. It waits
      for the work on all of [b]'s memory, through any view, and for the work on
      the memory a borrow maps. Host memory and an {!Io} device's memory have
      work to wait for only where a device borrowed them: an io device's memory,
      through its pages ({!Rig_edge.Io.pages}).

      It waits for one device after another and holds no lock. Once every point
      is reached it reads one word per device and allocates nothing. Work
      submitted after it returns is the caller's to exclude, by a claim taken
      before the wait ({!Claim}) or a lock of its own.

      Raises [Invalid_argument] if [b] is dead, and {!Lost} as {!Lost} states.
  *)

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they are
      there. It orders itself as work that reads [src] and writes [dst]: it
      waits for [src]'s last write and for every use of [dst].

      Between memory the host addresses, the host copies. Otherwise a device
      with a copy queue copies, as work on its timeline: [dst]'s device when
      only [src] is host-addressable, [src]'s otherwise, directly between memory
      it addresses or maps, and through the host's {e staging memory} otherwise:
      host memory, made at the first copy that needs it and kept for the life of
      the process ({!domains}). A device of this machine that maps no host
      memory (its [maps_host] fact, {!Rig_edge.facts}) stages through
      [Pinned] memory of its own
      instead, made at its first copy that needs it and kept until it is lost. A
      device that runs no copy has memory the host addresses, which the host
      copies; a borrow on it of another device's memory copies by that device.
      An {!Io} device's memory, of any machine, is read and written by its
      {!Rig_edge.Io.read} and {!Rig_edge.Io.write}, through the staging memory
      when the host does not address the other side, except a copy into it from
      memory the host does not address, of a device with a copy queue that maps
      its pages ({!Rig_edge.Io.pages}), which that device runs through them.
      Memory of a driver's device of another machine copies only directly, as
      one copy on a device of that machine: with memory of its machine, by
      [src]'s device; with memory this process's host addresses, by the other
      machine's side's device, whose driver carries the bytes
      ({!Submission.Copy}). No staging memory reaches another machine.

      Staging memory that a device lost while it used it is replaced, so a loss
      reaches no other device's copies.

      A copy between an {!Io} device's memory and a borrow of it ({!borrow})
      is refused, whatever their bytes: the io device would read or write its
      memory through its own pages, which a system may never finish, such as
      a file written from its own mapping.

      Raises [Invalid_argument] if [src] and [dst] differ in size, overlap
      ({!overlaps}), or either is dead, one is an {!Io} device's memory and the
      other a borrow of it, or a view of one, [dst]'s memory is [Read]
      ({!val-access}), or one is memory of a driver's device of another
      machine and the device that would copy runs no copy, or the other is
      memory of this machine that this process's host does not address, of a
      third machine, or of its machine that [src]'s device does not reach
      ({!reaches}), or the copy would stage through the [Pinned] memory of a
      device that maps no host memory and the other side's device maps neither
      it nor host memory; {!Lost} if a device that runs the copy is lost or is
      lost by it, and for [src] and [dst] as {!Lost} states; {!Out_of_memory}
      if a host or a device cannot allocate its staging memory, or a device's
      driver refuses to map the host's after the rounds of
      {{!reclaim}reclamation}; and what an {!Io} device's read or write
      raises. *)

  val device : t -> device
  (** [device b] is the device [b] is on: [d] for a buffer that {!create},
      {!of_io} or {!borrow} made on [d], {!host} for one {!of_string} or
      {!of_bigarray} made, and its buffer's device for a {!view} and for
      {!Claim.consume}'s result. *)

  val length : t -> int
  (** [length b] is the number of [b]'s bytes. *)

  val is_borrowed : t -> bool
  (** [is_borrowed b] is [true] iff [b] is borrowed. *)

  val dead : t -> string option
  (** [dead b] is [Some why] if [b] is dead, [why] the reason its memory was
      consumed with ({!Claim.consume}), and [None] while [b] lives. A library
      that calls this one on its caller's buffer checks it first, to refuse a
      dead buffer under its own name. *)

  val access : t -> access
  (** [access b] is the accesses [b]'s memory admits, the same for each of its
      views and borrows: [Read_write] for memory {!create}, {!of_string} and
      {!of_bigarray} make, and what {!of_io} was given.

      [Read] memory is never written: a {!copy} into it raises, and so do a
      submission that writes it ({!Submission.make}, {!submit}) and a host claim
      for writing ([rig_buffer_claim] in [rig.h]), and it is never exclusive
      ({!Claim}). The host writes it only by breaking this, through {!bigarray},
      after which what reads of it see is unspecified. A library that writes its
      caller's buffer checks this first, to refuse a [Read] buffer under its own
      name. *)

  val view : t -> first:int -> length:int -> t
  (** [view b ~first ~length] is the [length] bytes of [b] from its byte [first]
      on, over [b]'s memory.

      Raises [Invalid_argument] if [b] is dead, [first] or [length] is negative,
      or the bytes do not lie inside [b]'s. *)

  val spans : t -> bool
  (** [spans b] is [true] iff [b]'s bytes are all of the memory it lies in,
      below its borrows: those of a buffer {!create}, {!of_string} or
      {!of_bigarray} made, and of a borrow of one, and not those of a {!view} of
      part of it. *)

  val overlaps : t -> t -> bool
  (** [overlaps b b'] is [true] iff [b] and [b'] share a byte of memory: through
      views of one memory, a borrow and the memory it maps, or two bigarrays
      over the same bytes. Buffers of no bytes overlap nothing. *)

  val bigarray :
    ('a, 'b) Bigarray.kind -> t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** [bigarray k b] is the bytes of the host buffer [b] read as elements of
      kind [k], without a copy: [length b / Bigarray.kind_size_in_bytes k] of
      them, in the host's byte order. Writing through it writes [b], whose
      memory must admit it ({!val-access}). It, and every array made from it,
      keeps [b]'s memory alive while reachable, memory a borrow on the host maps
      included: no buffer reuses it and its device does not free it until then.
      Access through it is the host's: {!wait} orders it after devices' work.
      From then on the memory is outside the claims for good, and is never
      exclusive again ({!Claim}).

      Raises [Invalid_argument] if [b] is dead or not on {!host}, [b]'s bytes
      are not a whole number of elements of [k] starting at a multiple of their
      size (of one component's for complex kinds), or [b]'s memory is held
      exclusive by claims that have not consumed it ({!Claim.consume}), and
      {!Lost} as {!Lost} states. *)

  val blit_from_string : string -> int -> t -> int -> int -> unit
  (** [blit_from_string s i b j n] copies the [n] bytes of [s] from [i] into the
      host buffer [b] from its byte [j], and returns once they are there. It
      waits as a {!wait} with [Read_write] does. It takes no claim: claims are
      the caller's, as for {!copy}.

      Raises [Invalid_argument] if the ranges are not valid, [b] is dead or not
      on {!host}, or its memory is [Read] ({!val-access}), and {!Lost} as
      {!Lost} states. *)

  val blit_to_bytes : t -> int -> bytes -> int -> int -> unit
  (** [blit_to_bytes b i s j n] copies the [n] bytes of the host buffer [b] from
      its byte [i] into [s] from [j]. It waits as a {!wait} with [Read] does.

      Raises [Invalid_argument] if the ranges are not valid, or [b] is dead or
      not on {!host}, and {!Lost} as {!Lost} states. *)

  (** {1:low Low level}

      For the libraries that submit work. C code reads a buffer's host address,
      its length and the reason it is dead with [rig_buffer_host],
      [rig_buffer_bytes] and [rig_buffer_why], claims its memory with
      [rig_buffer_claim] and [rig_buffer_release], and waits under the claim
      with [rig_buffer_wait], which runs {!wait}; all are declared in
      [rig.h]. *)

  val address : t -> int
  (** [address b] is the address of [b]'s first byte as [b]'s device's work
      addresses it: its memory's address ({!Rig_edge.location}) plus {!offset}
      for a driver's memory, the host address for {!host}'s. An empty buffer
      that {!create} made on a driver's device names no memory: its address is
      [0], as work of no bytes reads none. An address fits in the 62 bits of an
      [int]'s non-negative range.

      Raises [Invalid_argument] if [b] is dead, is an {!Io} device's memory, or
      is memory its driver names by handle only. *)

  val handle : t -> nativeint
  (** [handle b] is the driver's object for [b]'s memory ({!Rig_edge.location}),
      which [b] starts {!offset} bytes into, such as an [MTLBuffer]; [0n] for an
      empty buffer that {!create} made on a driver's device.

      Raises [Invalid_argument] if [b] is dead or is {!host}'s or an {!Io}
      device's memory, which no driver object names. *)

  val offset : t -> int
  (** [offset b] is the number of bytes of [b]'s memory before [b]'s first byte.
  *)
end

(** {1:claims Claims} *)

(** Claims on memory.

    Every view and borrow of one memory shares one count of claims. A reader
    claims the memory while the host reads it, and a compiled call that writes
    over memory it was handed holds it exclusive, so that no reader sees the
    write. Claims never wait: a claim that cannot be had raises, or for an
    exclusive one, is refused. A host reader claims, then waits
    ({!Buffer.wait}): a wait before the claim does not cover a write in place
    that a donation on another domain makes between them.

    Memory that something outside this library reaches is {e outside the claims}
    and never exclusive: memory {!Buffer.of_bigarray} and {!Buffer.of_io} make,
    and memory {!Buffer.bigarray} exported. [Read] memory ({!Buffer.val-access})
    is never exclusive either.

    A claimed memory may be {e consumed}: every buffer over it made before
    becomes {e dead}, and reaching a dead buffer's bytes raises
    [Invalid_argument] with the consumption's reason. Consuming releases
    nothing: the memory lives while a buffer reaches it.

    Device work takes no claim. A {!submit} requires its caller's claims on the
    buffers it is passed until it returns. *)
module Claim : sig
  val read : Buffer.t -> unit
  (** [read b] claims [b]'s memory for reading, beside other readers.

      Raises [Invalid_argument] if [b] is dead or its memory is held exclusive,
      and {!Lost} as {!Lost} states. *)

  val release : Buffer.t -> unit
  (** [release b] ends a {!read} of [b]'s memory that the caller made. It
      accepts a dead [b].

      Raises [Invalid_argument] and changes nothing if the memory has no read
      claim of a {!read}, a {!with_} or [rig_buffer_claim]: a reader outside the
      claims is not one. *)

  type t
  (** The type for the claims of a {!with_}. *)

  val with_ : read:Buffer.t list -> donate:Buffer.t list list -> (t -> 'a) -> 'a
  (** [with_ ~read ~donate f] claims the memory of [read] and [donate] for
      reading, then tries each value of [donate] (its buffers, one per device)
      exclusive, and is [f] of the claims. A value is exclusive if each of its
      buffers {!Buffer.spans} its memory and the memory has no other claim;
      otherwise its buffers stay read. Every claim is released when [f] returns
      or raises, and the claims [f] was given hold nothing from then on:
      {!exclusive} of them is [false] and {!consume} raises. A use of them on
      another domain that overlaps [f]'s return is the caller's to order: rig
      does not detect it, and such a use may answer as if [f] were still
      running.

      Raises [Invalid_argument] before [f], releasing what it claimed, if a
      buffer is dead, a memory is held exclusive, or a buffer of [donate]
      overlaps another of [read] or [donate], and {!Lost} likewise, for a buffer
      as {!Lost} states. *)

  val exclusive : t -> Buffer.t -> bool
  (** [exclusive c b] is [true] iff [c] holds [b]'s memory exclusive: the caller
      may write it in place, through the buffer {!consume} returns, and no
      reader sees the write. *)

  val consume : t -> why:string -> Buffer.t -> Buffer.t
  (** [consume c ~why b] consumes [b]'s memory with the reason [why]: [b] and
      every buffer over that memory made before are dead. It is a live buffer
      over the same memory, which the caller writes in place only if [c] holds
      it {!exclusive}.

      Raises [Invalid_argument] if [c]'s {!with_} returned or raised, [c] does
      not claim [b]'s memory, [b] is dead, or [b] does not {!Buffer.spans} its
      memory. *)
end

(** {1:holds Holds} *)

(** What submissions' work uses beyond memory.

    A hold retains what its release frees, such as the driver objects a graph
    or an indirect command buffer makes and the images they name, until every
    submission made with it ({!Submission.make}) is done. It holds no memory
    and orders no work: a submission orders its memory itself, its fixed
    memory included. *)
module Hold : sig
  type t
  (** The type for holds. *)

  val make : (unit -> unit) -> t
  (** [make release] is a hold: [release] runs once, after the hold is
      unreachable and every submission made with it is done on each of its
      devices, or, on a lost device, once that device's stop returned with its
      word at its last value. It never runs while a lost device's word stays
      below its last value, and never in a child of [fork] for a hold made
      before the fork. It runs in the next {{!reclaim}drain} of any device, such
      as a {!Buffer.copy}'s, or as a lost device's stop returns. It holds no
      lock of this library, must not call it, and must not raise: an exception
      it raises is raised again by the call whose drain ran it. It counts as a
      call in flight on each device of the hold that is not lost, so no
      {!Rig_edge.Driver.stop} of those devices runs beside it. *)
end

(** {1:images Images} *)

(** Code loaded on a device. *)
module Image : sig
  type device := t

  type t
  (** The type for loaded images: a binary in a device's format, whose functions
      that device's work runs. *)

  val load : device -> string -> (t, string) result
  (** [load d binary] loads [binary] on [d]: a code object for AMD, a cubin for
      NV, a CUDA module for CUDA, a metallib for Metal. Where [d]'s memory holds
      the code, [load] allocates it as {!Buffer.create} allocates [Device]
      memory, so the code counts in [d]'s {!budget}, then copies it there as a
      submission on [d], after [d]'s queued work, and waits for it. Each call
      loads anew. The image stays loaded while [t] is reachable, and is unloaded
      once it is not and the work [d] was handed until then is done; its memory
      then returns to [d].

      [Error why] if [d]'s driver rejects [binary], with its reason, which
      starts with [d]'s {!name}. [d] stays usable.

      Raises [Invalid_argument] if [d] loads no code ({!host} or an {!Io}
      device), {!Out_of_memory}, and {!Lost} if [d] is lost or is lost by the
      load. *)

  val device : t -> device
  (** [device i] is the device [i] is loaded on. *)

  val entry : t -> string -> int option
  (** [entry i f] is the name compiled code gives [i]'s function [f] (the [code]
      of {!Rig_edge.Driver.entry}), such as a kernel descriptor's address, a
      [CUfunction] or an [MTLComputePipelineState], or [None] if [i] has no
      function [f]. Work that runs [f] keeps [i] reachable until it is done,
      such as a {!Hold}'s release that holds it.

      Raises [Invalid_argument] if [i]'s device cannot run [f]
      ({!Rig_edge.Driver.entry}), and {!Lost} if [i]'s device is lost. *)
end

(** {1:submitting Submitting work} *)

(** Prepared submissions.

    A submission is work for one device: its {e parts}, each for one of the
    device's queues ({!queues}), and the parts each waits for within the
    submission. It is made once and run many times. What changes from one run to
    the next, the buffers its work reads and writes and the points it waits for,
    are arguments of {!submit}, which keeps none of them once it returns, with a
    {!Submission.Run}, the caller's storage for one submit at a time. A
    submission holds nothing of a run, so any number of threads submit it at
    once, each with its own run. Its prepared form and a run live outside the
    OCaml heap, so a submit allocates nothing.

    A submission keeps every buffer its parts name and its fixed memory
    reachable while it is reachable itself, so their memory is never freed
    between {!make} and a submit: a part never hands its driver memory that was
    given back. Once the submission is unreachable, that memory returns by its
    stamps, after the work of the last submit. *)
module Submission : sig
  type device := t

  type t
  (** The type for prepared submissions. *)

  type ref = { at : int; slot : int }
  (** The type for a launch's references to a run's buffers: the 8 parameter
      bytes at [at] hold a byte offset into the run's buffer [slot], counting
      [reads] then [writes] ({!submit}). *)

  (** The type for the work of a part. Every buffer it names is used by the
      work: a copy's [dst] is written, everything else read. *)
  type work =
    | Words of Buffer.t
        (** The 32-bit words of the buffer, placed on the queue's ring, such as
            a run copy's indirect-buffer or GPFIFO entry. The buffer is host
            memory. *)
    | Fill of {
        fill : nativeint;
        arg : Buffer.t;
        ring_units : int;
        segment_bytes : int;
      }
        (** The C function at [fill], called inside the driver's hand-over with
            the queue's context, the host address of [arg] and the value
            ([edge] in {!Rig_edge.facts}). A ring driver's fill writes at most
            [ring_units] entries and [segment_bytes] bytes of argument segment;
            a library driver's declares [0] of each. *)
    | Copy of { src : Buffer.t; dst : Buffer.t }
        (** A copy of [src]'s bytes into [dst], on a copy queue. On a driver's
            device of another machine, one of them may be memory this process's
            host addresses, whose bytes the device's driver carries. *)
    | Launch of {
        image : Image.t;
        kernel : string;
        params : int;
        refs : ref array;
      }
        (** [image]'s function [kernel] run once, over the grid of groups of
            threads that its block of the run sets ({!block}), with the [params]
            bytes of parameters of that block, at most 4096.

            In place of the 8 bytes at each ref's [at], the function reads that
            offset plus the address of the first byte of the run's buffer [slot]
            as the device's work addresses it ({!Buffer.address}). Through a
            ref, it reaches only bytes of that buffer, and writes them only if
            the buffer is one of the run's [writes]: rig orders the work by
            these facts and checks neither. *)

  type part = { queue : string; after : int array; work : work }
  (** The type for parts: [work] on [queue], one of the device's
      ({!queues}). A part runs after the parts of its submission before
      it on its queue, and after those whose indices [after] lists: [after]
      orders parts of different queues. *)

  val make :
    ?hold:Hold.t ->
    ?fixed:(Buffer.t * Buffer.access) list ->
    reads:int ->
    writes:int ->
    device ->
    part array ->
    t
  (** [make ~hold ~fixed ~reads ~writes d parts] is a submission of [parts] on
      [d] whose every run reads [reads] buffers and writes [writes] buffers
      ({!submit}), a buffer counted as often as it is passed. Every submit of
      the submission is one [hold]'s release waits for ({!Hold.make}), and the
      submission keeps [hold] reachable. [make] reads [parts], and the arrays in
      them, once: changing them afterwards changes nothing.

      [fixed] (defaults to none) names memory every run uses besides the run's
      buffers and the parts' own, such as the memory a [Words] replay's command
      buffers address, each with its access. Each is ordered by its access on
      every submit, as a run's buffer is. The submission keeps it reachable
      while the submission is. A memory may be fixed in any number of
      submissions.

      Raises [Invalid_argument] if [d] is {!host} or an {!Io} device, which run
      no submitted work, [reads] or [writes] is negative, an index of a part's
      [after] is negative or not below its own, a queue is not one of [d]'s, a
      part's buffer is dead, a {!Words} or {!Fill} buffer is not host memory, a
      {!Copy}'s buffers differ in size, are not [d]'s memory (on a driver's
      device of another machine, one of them may be memory this process's host
      addresses), or its [dst]'s memory is [Read] ({!Buffer.val-access}), or
      [d]'s driver runs no copies (it lists no copy queue, {!queues}), or a
      part's queue does not run its work, a buffer of [fixed] is dead, not on
      [d], or [Read_write] on [Read] memory, a {!Launch}'s [image] is not loaded
      on [d] or has no function [kernel], its [params] is negative or above
      4096, a ref's [at] is not a multiple of 8, its 8 bytes are not among the
      parameters or its [slot] is not below [reads + writes], or two refs share
      an [at]; as {!Image.entry} where [d] cannot run a launch's function; and
      {!Lost} if [d] is lost. Parts that never fit [d]'s queues, and launches
      whose blocks they refuse, are refused at {!submit}. *)

  type block = private int
  (** The type for where a launch's block lies in a run ({!Run}). *)

  val block : t -> int -> block
  (** [block s i] is the block of [s]'s part [i] in a run: its grid, its groups
      and its parameters. The blocks of [s]'s launches lie one after the other,
      in the order of its parts. [block s i] is meaningful only in a run that
      then submits [s]: another submission's blocks may lie elsewhere.

      Raises [Invalid_argument] if [s] has no part [i] or it is no {!Launch}. *)

  (** Runs. *)
  module Run : sig
    type t
    (** The type for runs: the storage of one submit at a time, which holds the
        blocks of the submission's launches, and what rig records while the
        submit runs, such as the points it follows, the regions its work names
        and its answer. *)

    val make : unit -> t
    (** [make ()] is a run with no blocks. It grows to the largest submission it
        serves and the blocks its setters store into, outside the OCaml heap,
        and never shrinks. *)

    (** {1:blocks Blocks}

        A block holds a launch's geometry and parameters. A run keeps each
        byte's last store across submits, of whichever submission: a launch
        reads what was stored last, so a caller stores every parameter its
        function reads before each submit. Bytes no setter ever stored read
        [0]. A store into a block the run does not hold yet grows the run to
        hold it. No setter allocates on the OCaml heap: each is an external
        taking untagged ints and unboxed floats, so no argument is boxed at a
        call, inlined or not.

        Every setter raises [Invalid_argument] if a submit is using the run,
        such as a signal handler's store during a submit's wait, and
        [Stdlib.Out_of_memory] if the host has no memory to grow the run. *)

    external groups :
      t ->
      (block[@untagged]) ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      unit = "caml_rig_run_groups_byte" "caml_rig_run_groups"
    (** [groups r b x y z] sets the grid of [b] to [x] by [y] by [z] groups.

        Raises [Invalid_argument] if one is negative or above [2{^32} - 1]. *)

    external threads :
      t ->
      (block[@untagged]) ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      unit = "caml_rig_run_threads_byte" "caml_rig_run_threads"
    (** [threads r b x y z] sets the groups of [b] to [x] by [y] by [z] threads.

        Raises [Invalid_argument] if one is negative or above [2{^32} - 1]. *)

    external shared : t -> (block[@untagged]) -> (int[@untagged]) -> unit
      = "caml_rig_run_shared_byte" "caml_rig_run_shared"
    (** [shared r b n] sets the dynamic shared memory of each group of [b] to
        [n] bytes.

        Raises [Invalid_argument] if [n] is negative or above [2{^32} - 1]. *)

    (** {2:params Parameters}

        [int32 r b i v] stores [v] at byte [i] of [b]'s parameters, in the
        host's byte order, as do the others for their sizes: [int32] the 32 low
        bits of [v], [int64] [v] sign-extended, [float32] [v] rounded to single
        precision, [float64] [v]. At a ref's [at], the 8 bytes are the offset
        into its buffer ({!ref}).

        Each raises [Invalid_argument] if [i] is negative or the bytes it stores
        end past [b]'s parameters. *)

    external int32 :
      t -> (block[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
      = "caml_rig_run_int32_byte" "caml_rig_run_int32"

    external int64 :
      t -> (block[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
      = "caml_rig_run_int64_byte" "caml_rig_run_int64"

    external float32 :
      t -> (block[@untagged]) -> (int[@untagged]) -> (float[@unboxed]) -> unit
      = "caml_rig_run_float32_byte" "caml_rig_run_float32"

    external float64 :
      t -> (block[@untagged]) -> (int[@untagged]) -> (float[@unboxed]) -> unit
      = "caml_rig_run_float64_byte" "caml_rig_run_float64"
  end
end

val submit :
  Submission.t ->
  run:Submission.Run.t ->
  reads:Buffer.t array ->
  writes:Buffer.t array ->
  waits:Point.t array ->
  Point.t
(** [submit s ~run ~reads ~writes ~waits] runs [s]'s work once, reading the
    buffers of [reads] and writing those of [writes] after the points of
    [waits], with [run] as its storage. It hands the work to [s]'s device [d] as
    the value [v] it assigns, and is the point [(d, v)]. A buffer may appear
    more than once, in either array. The buffers are on [d]: [d]'s work reaches
    other memory through a {!Buffer.borrow}, an {!Io} device's through its
    pages. [submit] reads each element of [reads] and [writes] once and keeps
    the buffer reachable until it has raised its stamps. While [submit] runs, no
    claim holds a buffer it is passed exclusive ({!Claim}): a caller ensures it
    by a claim of its own, or by memory nothing else reaches. [submit] does not
    check it. It:
    + Loads the points [s]'s work must follow: the last write of each buffer of
      [reads], of each buffer its parts read and of its fixed memory read, every
      use by another device of each buffer of [writes], of each copy's [dst] and
      of its fixed memory written, and the points of [waits].
      Each foreign point not yet reached is a wait in [d]'s queue if [d]'s
      queues wait for the producer's completion ({!Rig_edge.waits}) and [d] maps
      the producer's timeline word, decided once per pair of devices, up to the
      most waits [d]'s queues carry; otherwise [submit] waits for it now,
      holding no lock. Either way it first commits the work of the point's
      device, before it takes [d]'s turn.
    + Takes [d]'s {e turn}, the right to be the one call of [d]'s room check,
      hand-over or commit, and asks [d]'s driver for room ([edge] in
      {!Rig_edge.facts}). Once the parts fit, it assigns [v], one more than
      {!submitted}[ d], hands the work over, which encodes it on [d]'s queues,
      and raises the stamps of [reads], [writes], the parts' buffers, the fixed
      memory and the hold to [(d, v)]. While they do not fit, it commits [d]'s
      work, waits for [d]'s next value with the turn released, and tries again.
      No OCaml code runs between the assignment and the turn's release, so a
      value is handed over or [d] is lost.

    The work runs after [d]'s earlier work without another call: [v]'s work
    starts once the work of every earlier value of [d] completed, on every
    queue of [d]. [d]'s word shows [v] once [v] is {e committed} and its work
    completed. Every wait for a value of [d], on the host or in another
    device's queue, first commits [d]'s work, so no wait waits for work that is
    not committed. [d]'s driver also commits on its own, at least once every [k]
    values, [k] a bound of its own ([edge] in {!Rig_edge.facts}): while work is
    submitted the word shows each value at most [k] values late, and values
    submitted since the last commit show after the next wait for [d] or the
    next [k] values.

    [submit] reads the blocks of [run] ({!Submission.block}) in the hand-over
    alone, which may come after it released the domain lock to wait for room or
    for a producer. The work runs with the blocks as they were then: a store
    into [run] after [submit] returns changes none of it. The caller keeps other
    threads from storing into [run] until [submit] returns.

    It allocates nothing unless it waits.

    Raises [Invalid_argument] if another submit is using [run] ([submit] takes
    it at entry and gives it back when it returns or raises), if [reads] or
    [writes] holds another number of buffers than {!Submission.make} declared, a
    buffer of [reads], [writes], a part or [s]'s fixed memory is dead, a buffer
    of [reads] or [writes] is not on [d], the memory of a buffer of [writes] is
    [Read] ({!Buffer.val-access}), a launch's ref names one whose memory has no
    address ({!Buffer.address}), [run] does not hold the block of [s]'s last
    launch, which no setter stored into, or the parts never fit [d]'s empty
    queues, name one its driver does not run, or hold a launch whose block [d]'s
    driver refuses: a grid or a group of no size along an axis, or groups,
    threads per group or shared memory beyond [d]'s or the function's limits;
    and {!Lost} if [d] is lost, [d]'s hand-over fails, or a producer [d]'s queue
    waits on is lost before the hand-over, and for the buffers and the points
    [s] follows as {!Lost} states. A device lost after [v] was handed over
    raises {!Lost}, with [v]'s stamps naming it. *)

(** {1:profiles Profiles} *)

(** Profiles of devices' work.

    A profile of every device is taken while a function runs ({!take}). Profiles
    nest and overlap, in any domains: each holds the events recorded while it is
    taken. Every time is on the host clock ({!now}). When no profile is taken,
    recording reads one atomic word and allocates nothing. *)
module Profile : sig
  type device := t

  (** The type for profile events. Times are nanoseconds of the host clock. *)
  type event =
    | Span of {
        device : device;
        lane : string;  (** Its track within the device. *)
        name : string;
        start : int;
        stop : int;
      }
        (** Work that ran on a lane of a device. The host's lanes are its
            domains, ["domain 0"], ["domain 1"], …. *)
    | Allocation of { device : device; time : int; allocated : int }
        (** The bytes of memory [device] allocated from [time] on. *)
    | Load of { image : Image.t; binary : string; time : int }
        (** An image loaded. *)
    | Counters of {
        device : device;
        name : string;
        start : int;
        stop : int;
        counters : (string * int array) list;
            (** Each counter asked for and its count during the run, one per
                unit of the hardware that counts it. *)
      }  (** The counters of a run of a function. *)
    | Trace of {
        device : device;
        name : string;
        start : int;
        stop : int;
        part : int;  (** The part of the device that wrote it. *)
        data : string;  (** The trace as the device wrote it. *)
      }  (** The thread trace of a part of a device during a run. *)
    | Overwritten of { device : device; time : int; runs : int }
        (** Runs whose counters and traces the device overwrote unread. *)
    | Copy of {
        src : device;
        dst : device;
        bytes : int;
        start : int;
        stop : int;
      }
        (** A transfer of [bytes] from [src]'s memory to [dst]'s, by a
            {!Buffer.copy}, from when it was asked to when the host saw it done.
            A copy through staging memory records each piece into and out of it,
            with {!host} as one side, and no event of its own. *)

  val take :
    ?counters:string list -> ?trace:bool -> (unit -> 'a) -> 'a * event list
  (** [take f] is [f ()] and the events recorded while it ran, in time order
      and, at equal times, longest first, then in the order they were recorded.
      It asks the libraries that encode work to count [counters] (defaults to
      none) and, with [trace] (defaults to [false]), to trace ({!counters},
      {!traced}). Before it returns it waits for the points whose events are
      still to be read ({!after}); those of a device lost meanwhile are left
      out. If [f] raises, the events are dropped and the exception is raised
      again with its backtrace.

      Raises [Invalid_argument] if [counters] names a counter twice. *)

  val enabled : unit -> bool
  (** [enabled ()] is [true] iff some profile is being taken. *)

  val counters : unit -> string list
  (** [counters ()] is the counters the profiles being taken ask for, each once,
      in the order they started. A library that keeps encoded work keeps it for
      each value of [counters ()]. *)

  val traced : unit -> bool
  (** [traced ()] is [true] iff a profile being taken asks for traces. *)

  val now : unit -> int
  (** [now ()] is the host clock: nanoseconds of the system's monotonic clock,
      from an unspecified start. *)

  val span : string -> (unit -> 'a) -> 'a
  (** [span name f] is [f ()], recorded as a span named [name] on the calling
      domain's lane of {!host}, from the call until [f] returns or raises, in
      each profile taken when the call starts. *)

  val after : Point.t -> (unit -> event list) -> unit
  (** [after p f] records the events [f ()] in each profile being taken, once
      [p] is reached: [f] reads what [p]'s work wrote, such as its times or
      counters. [f] runs in the first wait that finds [p] reached, among the
      waits for [p]'s device ({!wait}, {!Buffer.wait}, {!Buffer.copy} and
      {!take}'s), before that wait returns: memory a {!Buffer.wait} returns for
      is read before its caller rewrites it. [f] holds no lock of this library,
      must not call it and must not raise: an exception it raises is raised
      again by the wait that ran it. It does nothing unless {!enabled}. *)

  val record : Point.t -> lane:string -> name:string -> Buffer.t -> unit
  (** [record p ~lane ~name stamps] is [after p] of a {!Span} of [p]'s device
      named [name] on [lane], whose start and stop are the unsigned 64-bit words
      at bytes 8 and 24 of the host memory [stamps], in the host's byte order
      and on the host clock: [p]'s work, or its driver, writes them, as
      {!timestamp} does.

      Raises [Invalid_argument] if [stamps] is not 32 bytes of host memory
      starting at a multiple of 8. *)

  val timestamp : nativeint
  (** [timestamp] is the address of the C function

      {v void rig_timestamp(void *word); v}

      which stores {!now} into the 64-bit word at [word] with one aligned atomic
      store, in the platform's C calling convention. It takes no lock, so a
      device library may call it from a completion path. *)

  val output_chrome_trace : out_channel -> event list -> unit
  (** [output_chrome_trace oc events] writes [events] to [oc] in Chrome's trace
      event format, JSON, which Perfetto and [chrome://tracing] load: a process
      per device, a thread per lane, a complete event per span, per copy (on its
      [src]'s process) and per run's counters (each summed over its units), a
      counter [memory] per allocation change, and instant events for loads,
      traces and overwritten runs. Times are microseconds from the earliest
      event. [oc] is neither flushed nor closed. *)
end

(** {1:drivers Drivers}

    For the vendor libraries that open devices. A driver links only
    [rig.edge], whose {!Rig_edge} states the contract, and matches
    {!Rig_edge.Driver}. *)

module type Driver = Rig_edge.Driver
(** The type for drivers. *)

module type Io = Rig_edge.Io
(** The type for io devices' libraries. *)

(** {1:opening Opening} *)

val open_ :
  (module Driver with type t = 'a) ->
  ?machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (t, string) result
(** [open_ (module D) ~machine ~name make] is the open device named [name] on
    [machine] (defaults to this one), the machine whose hardware [make] opens. A
    machine's name names one machine for the life of the process: a library that
    reaches machines gives each one it makes a name of its own, so a second
    connection to one address is another machine, with devices of its own. If no
    device of that name is open there, [make ()] opens it, under the name's
    lock, so one name on one machine has one live device; its [Error] is the
    result, and an exception it raises is raised again, the name left unopened.
    Opens of other names go on meanwhile. A closed or lost device's name opens
    again once its driver's {!Rig_edge.Driver.stop} returned.

    The result is [Error why] if the name's device is lost and its stop has not
    returned, if the process failed ({!fail}), if [machine] is another machine
    whose host is lost or closed, [why] as that host's {!Lost} prints, or if the
    process opened 65,535 devices already: device indices are never reused.

    Raises [Invalid_argument] if the open device of that name is another
    driver's, [machine] is another machine whose host was never opened
    ({!open_host}), or the driver's facts break the contract: their [edge] is
    [0n] or its state's first member is [NULL], a queue named ["COPY:i"] runs
    no [Copy], or [hang_ms] is [Some n] with [n < 1] ({!Rig_edge.facts}). *)

val open_host :
  (module Driver with type t = 'a) ->
  machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (t, string) result
(** [open_host (module D) ~machine ~name make] is {!open_} of [machine]'s host
    ({!host_of}): a driver's device of [machine] that runs the submissions and
    loads the code its driver takes. Devices of [machine] open once it is open.

    Raises [Invalid_argument] as {!open_}, or if [machine] has a host of another
    name, open and not lost, or still opening. *)

val open_io :
  (module Io with type t = 'a) ->
  ?machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (t, string) result
(** [open_io (module I) ~machine ~name make] is {!open_} for an io device.

    Raises [Invalid_argument] as {!open_}. *)

val memory_device : string -> (t, string) result
(** [memory_device name] is {!open_} of the device named [name] whose memory is
    the host's and whose work this process runs, with a timeline of its own: it
    runs a submission's copies and fills before its submit returns. It stands
    for a device with a timeline in tests of what names several devices. *)
