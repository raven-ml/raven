(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Calls of the metallib's kernels: one submission, or several in order, its
    parts.

    A family plans a kernel call that nx.metal found nothing to refuse in as one
    {e call}: launches of the metallib's kernels, run in order, and the
    {e slots} they address, the buffers a submit passes. A call's launches are
    cut into {e parts}, each one submission ({!field-parts}): one part, unless
    the call passes what one submission holds ({!part_budget},
    {!part_launches}). A plan makes no buffer, touches no device and holds no
    array: a slot names a call's operand ({!Operands}), the device's workspace
    or tickets, a buffer of a size the door makes once it accepts the call and
    frees once the call's last part is submitted, or a constant the door finds
    or makes once it accepts the call, so a plan the door abandons leaves
    nothing behind, and calls of equal facts may share one.

    A launch's parameters are 64-bit words, which its kernel reads at the
    indices {!Kernels} names, as the fields of a struct at buffer 0. A {e ref}
    is a word that holds a byte offset into a slot's buffer, which rig turns
    into that byte's GPU address when the work runs, and orders the work by.

    Parts whose launches have the same kernels and refs, whatever their words'
    values, take one submission ({!key}), so a device keeps one submission for
    each, which a later part submits with its launches' values ({!args}). *)

(** {1:slots Slots} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for bytes outside the OCaml heap. *)

(** The type for the buffers a call addresses. *)
type slot =
  | Operand of int
      (** The buffer of the call's operand of this number ({!Operands}). *)
  | Workspace
      (** The device's workspace: bytes of no value at the call's start, which
          no other call writes while the call runs. *)
  | Tickets
      (** The device's tickets: 32-bit words that are zero at the start of each
          part, and that every launch leaves zero. *)
  | Fresh of int
      (** A buffer of this many bytes, of no value at the call's start, which
          the door makes once it accepts the call and frees once it has
          submitted the call's last part. *)
  | Constant of { key : string; length : int; bytes : unit -> bigbytes }
      (** A buffer of [length] bytes holding [bytes ()], which no launch writes,
          and which are the same for every call that names [key]. Once it
          accepts the call, the door takes the device's buffer of [key], or
          calls [bytes] once, uploads its result into a new buffer and waits for
          the upload. [bytes] may be called from any domain, and from several at
          once for one key. The door raises [Invalid_argument] where the buffer
          of [key] or [bytes ()] is not of [length] bytes, so a key two tables
          share by mistake raises where a kernel would read past a buffer. The
          device keeps it for later calls, with at most {!kept_constant_count}
          constants taking at most {!kept_constants} bytes together, and
          otherwise the door frees it once it has submitted the call's last
          part. *)
  | Uploaded of string
      (** A buffer holding these bytes, which no launch writes, and which are
          the call's alone: the door makes it and uploads them once it accepts
          the call, and frees it once it has submitted the call's last part. It
          takes none of the device's kept constants. *)

val kept_workspace : int
(** [kept_workspace] is 64 MiB, the most bytes of {!Workspace} a call takes. A
    call that needs more takes a {!Fresh} buffer instead. *)

val kept_tickets : int
(** [kept_tickets] is 4 MiB, the most bytes of {!Tickets} a call takes. A call
    that needs more takes a {!Fresh} buffer, which its first launch zeroes. *)

val kept_constants : int
(** [kept_constants] is 64 MiB, the most bytes of {!Constant}s a device keeps. A
    constant the device no longer keeps returns once collected
    ({!Rig.section-reclaim}), since a call on another domain may have just taken
    it. *)

val kept_constant_count : int
(** [kept_constant_count] is 16, the most {!Constant}s a device keeps, so that a
    call finds its own among few. *)

(** {1:launches Launches} *)

type ref = { word : int; slot : int; write : bool }
(** The type for refs: the word that holds a byte offset into the buffer of
    [slot], which the launch writes iff [write]. *)

val most_refs : int
(** [most_refs] is 13, the most refs a launch takes: the loop's
    {!Kernels.loop_buffers} buffers and its tail, so that the memory a kept
    launch holds is bounded. A family whose refs grow with its program, the loop
    alone, splits its launches under it, so no call reaches {!launch}'s refusal.
*)

(** {1:budget Time budget}

    Metal ends a command buffer whose work it judges too long, which loses the
    device, and the evidence names a launch as the work it times. So each
    launch's estimated time on the GPU stays within {!budget}: {!launch} refuses
    a launch whose {!seconds} exceed it, and a family splits longer work into
    several launches.

    A launch states its {!cost} in counts its family can make; {!seconds} alone
    turns them into time, on the family's slowest GPU, the M1's of 7 cores. The
    estimate is meant to err high: it adds the times of the launch's traffic,
    its work and its longest chain, which the GPU overlaps, and each rate is one
    a launch reaches at best. The rates below are measured, on the M1 Max, where
    they say so. A family's weights, the counts it gives an operation, are its
    estimates where it does not state a measurement. *)

type cost = {
  bytes : int;
      (** The launch's device-memory traffic, read and written: a whole {!line}
          for each access whose line no neighbouring thread shares. *)
  ops : int;
      (** The launch's work, of all its threads together, in [float32]
          operations: one a lane and a cycle. A family weighs its other
          operations, a 64-bit division or a barrier, by the cycles it measured
          them to take, and counts a thread that has no work in a group that has
          some as working, since it holds its lane. *)
  chain : int;
      (** The longest sequence of dependent steps one group runs, which no other
          group shortens, in {e links}: a dependent [float32] operation is a
          link, and a dependent read of device memory {!read_links}. *)
}
(** The type for a launch's counts. *)

val line : int
(** [line] is 128, the bytes of the GPU's cache line. *)

val bandwidth : float
(** [bandwidth] is 58.0e9 bytes a second, 0.85 of the M1's 68.25 GB/s: the M1
    Max reaches 0.93 of its own in a copy, and more counted in whole lines from
    a source whose reads share none. *)

val cores : int
(** [cores] is 7, the M1's GPU cores, each of which runs a group whole. *)

val lanes : int
(** [lanes] is 128, the lanes of a core. *)

val nominal_clock : float
(** [nominal_clock] is 1.278e9, the M1 GPU's cycles a second. *)

val op_rate : float
(** [op_rate] is [0.91 nominal_clock], a lane's [float32] operations a second at
    best: 128 independent [fma]s a step, written out, reach 0.91 of one a lane
    and a cycle in a full grid of the M1 Max, and the M1 is taken to reach the
    same fraction. Other operations may take longer. *)

val lane_rate : float
(** [lane_rate] is 25.0e6 links a second, a lane's in a chain: the M1 Max takes
    35.5 ns a link of a dependent [fma] in a lone thread. *)

val read_links : int
(** [read_links] is 16, the links of a dependent read of device memory: the M1
    Max takes 472 ns a read in a lone thread following a cycle through 64 MiB.
*)

val divide_ops : int
(** [divide_ops] is 16, the [float32] operations of a thread's quotient of a
    number below [2{^32}] by the host's divisor (device.h's [divide]) and its
    remainder: in a full grid of the M1 Max, rounds of them take 12.8 to 13.8
    operations each beyond the loop's. *)

val wide_divide_ops : int
(** [wide_divide_ops] is 96, {!divide_ops} for a 64-bit number by device.h's
    [nx_divide]: 93.7 to 94.7 operations a round. *)

val native_divide_ops : int
(** [native_divide_ops] is 4096, {!divide_ops} for a 64-bit number past [2{^32}]
    by the GPU's own [/] and [%], which it emulates: 3494 operations a round at
    a divisor of 7, and 3892 at 3. *)

val dispatch : float
(** [dispatch] is about 2.06e-6, the seconds a launch takes whatever its work:
    the M1 Max's 1.37 us at most a dispatch of an empty kernel, in a chain of
    dependent dispatches, at the M1's clock and 1.48 times. *)

val budget : float
(** [budget] is 0.05, the most seconds {!seconds} gives a launch. *)

val part_budget : float
(** [part_budget] is 10.0, the most seconds {!seconds} gives the launches of one
    part of a call, which one submission runs. A call past it runs as several
    parts. The evidence points to a launch as what Metal times: a command buffer
    of 120 s of 40 ms launches has run on an M1 Max. *)

val commit_after : float
(** [commit_after] is [part_budget /. 256.], about 39 ms. Once it has submitted
    a part whose launches {!seconds} estimates past it, of a call of one part or
    several, the door commits the device's work ({!Rig.Point.reached}).
    rig_metal commits a command buffer once 256 values have joined it, so a
    command buffer holds at most [2 *. part_budget] of the calls of one domain:
    255 values under [commit_after] each, then one part. A command buffer then
    holds at most one such part of each domain submitting at the time and less
    than [part_budget] of smaller work. Where the commit finds the device's turn
    held for 200 ms, the buffer is committed later: by the holder's commit, a
    completion, a submit finding fewer than three queued, or 256 values, and
    until then a domain's next part may join it. Only an encode of about 10{^ 5}
    launches, or a submit waiting on a full queue, holds the turn that long; the
    parameter arena holds 2-3 such encodes at once. *)

val part_launches : int
(** [part_launches] is 100,000, the most launches one part holds, so that a
    submission's encoding and the blocks that hold its values ({!Rig.submit})
    stay bounded. *)

val allowed : unit -> float
(** [allowed ()] is {!budget}, or less in tests ({!bound_budget}). *)

val seconds :
  threads:int -> groups:int * int * int -> ?resident:int -> cost -> float
(** [seconds ~threads ~groups ~resident c] estimates the time of a launch over
    [groups] of [threads] threads, of which a core holds [resident] at once
    (defaults to [1]), and cost [c]: {!dispatch}; plus [c.bytes] at
    {!bandwidth}; plus [c.ops] over the lanes the launch can fill, the [threads]
    of its groups up to {!lanes} a group and {!cores} groups, at {!op_rate};
    plus [c.chain] at {!lane_rate} for each wave of [resident * cores] groups.
*)

val estimate :
  bandwidth:float ->
  cores:int ->
  op_rate:float ->
  threads:int ->
  groups:int * int * int ->
  ?resident:int ->
  cost ->
  float
(** [estimate ~bandwidth ~cores ~op_rate ~threads ~groups ~resident c] is
    {!seconds} less {!dispatch} on a GPU of [bandwidth] bytes a second and
    [cores] cores of {!lanes} lanes of [op_rate] [float32] operations a second,
    for the bench, which checks a launch's time on the M1 Max against it plus
    the fixed cost it measures there. [seconds] is it at the M1's plus
    {!dispatch}. *)

val holds : threads:int -> int -> int
(** [holds ~threads shared] is the most groups of [threads] threads and [shared]
    threadgroup memory bytes a core holds at once by its 3072 threads and 64 KiB
    of threadgroup memory, a group's rounded up to 16 bytes. *)

val resident : granted:(Kernels.t -> int) -> Kernels.t -> shared:int -> int
(** [resident ~granted k ~shared] is the groups of [k] a core holds at once when
    each takes [shared] bytes of threadgroup memory: {!holds} at [k]'s threads,
    bounded by its registers. [granted k] is the threads Metal lets a group of
    [k]'s pipeline have, which its registers bound: a group runs on one core, so
    the core holds that many threads of it at once. It is at least [1] where
    [granted k] is at least {!Kernels.threads}[ k], as resolving a pipeline
    checks, and [shared] is at most {!group_memory}. It is [1] for a kernel
    whose threads are declared its most, as the factorisations' and the spectral
    routines' are, since Metal grants it exactly those: their estimates count
    one group a core at a time, which errs high, by up to {!holds} times for a
    step its chain bounds. *)

val most : fits:(int -> bool) -> int -> int option
(** [most ~fits groups] is the most groups up to [groups] that [fits], by
    bisection, for a [fits] that holds of every count below one it holds of. It
    is [None] if [fits 1] does not hold. *)

(** {1:calls Calls} *)

type launch = {
  kernel : Kernels.t;
  constants : int array;
      (** The values of its kernel's function constants, a specialised step's
          ({!Kernels.family_fft_steps}), by index; none for every other kernel.
          A kernel and its constants are a {e key}. *)
  groups : int * int * int;  (** The groups along x, y and z. *)
  threads : int * int * int;  (** A group's threads along x, y and z. *)
  shared : int;  (** The threadgroup memory bytes of a group. *)
  resident : int;  (** The groups a core holds at once. *)
  cost : cost;
  words : Bytes.t;
      (** The parameters, {!Kernels.words}[ kernel] little-endian words. *)
  refs : ref array;
}
(** The type for launches. *)

type part = private {
  run : launch array;
      (** The part's launches, in order, at least one: a run of its call's. *)
  binds : int array;
      (** The call's slot of each of the part's slots, which its refs' slots
          number in the order its launches first name them. A call of one part
          numbers them as its own: [binds.(i)] is [i]. *)
  local : int array;
      (** The part's slot of each of the call's slots, [-1] for one its launches
          name none of. *)
  kept : bool;
      (** Whether a device keeps the part's submission for later parts of its
          key ({!key}). *)
  commits : bool;
      (** Whether its launches' estimate passes {!commit_after}, after which the
          door commits the device's work. *)
}
(** The type for parts: the launches one submission runs. *)

type t = private {
  launches : launch array;  (** In order, at least one. *)
  slots : slot array;  (** The slots refs name, by index. *)
  workspace : int;  (** The bytes of {!Workspace} the call takes. *)
  tickets : int;  (** The bytes of {!Tickets} the call takes. *)
  parts : part array;
      (** The call's launches, cut in order into parts, at least one
          ({!finish}). *)
  zeroing : int;
      (** The launches first that zero tickets of the call's own ({!finish}). *)
  tickets_slot : int;  (** The tickets' slot, [-1] for none. *)
}
(** The type for calls. *)

(** {1:building Building} *)

type builder
(** The type for calls being planned. *)

val make : unit -> builder
(** [make ()] is a call of no launch and no slot. *)

val slot : builder -> int -> int
(** [slot b k] is the index of a slot of [b] that holds the buffer of the call's
    operand [k]: one slot for each [k]. A family names a buffer by the first
    operand of it ({!Operands.operand}'s [buffer]), so that a call binds each
    buffer to one slot.

    Raises [Invalid_argument] if [k] is negative. *)

val take : builder -> int -> int
(** [take b n] is the byte offset into [b]'s workspace slot of a piece of [n]
    bytes, the first on 256 bytes past the pieces [b] holds. *)

val mark : builder -> int
(** [mark b] is the end of the pieces [b] holds. *)

val release : builder -> int -> unit
(** [release b m] gives back the pieces taken since [mark b] was [m]: the
    launches planned so far run before any later one, so a later launch may use
    them again. *)

val workspace : builder -> int
(** [workspace b] is the index of [b]'s workspace slot, whose pieces {!take}
    gives. *)

val tickets : builder -> int -> int
(** [tickets b n] is the byte offset into [b]'s tickets slot ({!tickets_slot})
    of [n] ticket words for the launch planned next. Each is zero when that
    launch starts, since every launch leaves the words it takes zero, so every
    launch takes them from the slot's start. *)

val tickets_slot : builder -> int
(** [tickets_slot b] is the index of [b]'s tickets slot. *)

val constant : builder -> key:string -> length:int -> (unit -> bigbytes) -> int
(** [constant b ~key ~length bytes] is the index of a slot of [b] that holds the
    {!Constant} [key] of [length] bytes [bytes ()]: one slot for each key. A key
    names its bytes alone, across families: a family's keys start with its name,
    as ["copy "].

    Raises [Invalid_argument] if [length] is negative, or if [b] names [key]
    with another length. *)

val uploaded : builder -> string -> int
(** [uploaded b s] is the index of a slot of [b] that holds the {!Uploaded}
    bytes [s]: one slot for each distinct [s]. *)

val words : Kernels.t -> Bytes.t
(** [words k] is zero parameters for a launch of [k]. *)

val set : Bytes.t -> int -> int -> unit
(** [set w i v] stores [v] as word [i] of [w]. *)

val group_memory : int
(** [group_memory] is 32 KiB, the threadgroup memory a group of the Apple7
    family holds. No kernel declares threadgroup memory of its own, so a launch
    may take all of it. *)

val most_groups : int
(** [most_groups] is [2{^32} - 1], the most groups of a launch along an axis: a
    kernel reads its group's index as 32 bits. *)

val launch :
  builder ->
  Kernels.t ->
  groups:int * int * int ->
  ?shared:int ->
  ?resident:int ->
  ?threads:int ->
  ?constants:int array ->
  cost:cost ->
  Bytes.t ->
  ref array ->
  unit
(** [launch b k ~groups ~shared ~resident ~threads ~constants ~cost w refs] adds
    a launch of [k] specialised by [constants] (defaults to none) over [groups]
    of [threads] threads along x (defaults to {!Kernels.threads}[ k]) with
    [shared] threadgroup memory bytes (defaults to [0]), of which a core holds
    [resident] groups at once (defaults to [1]), at most what the kernel's
    pipeline lets a core hold, which the family reads from Metal, and cost
    [cost], with words [w] and refs [refs], to [b], after its other launches.

    Raises [Invalid_argument] if an axis of [groups] is not positive or above
    {!most_groups}, if [shared] is negative or above {!group_memory}, if
    [constants] is empty for a specialised step or not empty for another kernel,
    if [threads] is not positive or above {!Kernels.threads}[ k], if [resident]
    is not positive or more groups than a core's 3072 threads or 64 KiB of
    threadgroup memory hold, a group's rounded up to 16 bytes, if a count of
    [cost] is negative, if the launch's {!seconds} exceed {!allowed}[ ()], if
    [w] is not {!Kernels.words}[ k] words, if a ref's word is not among them, or
    if [refs] holds more than {!most_refs}. *)

val zero : builder -> slot:int -> at:int -> int -> unit
(** [zero b ~slot ~at n] adds the launches that zero [n] bytes of the slot
    [slot] from byte [at], a 32-bit word a thread, each within {!allowed}[ ()],
    to [b], after its other launches.

    Raises [Invalid_argument] if [at] or [n] is negative or not a multiple of 4.
*)

val divisor : int -> int
(** [divisor d] is the word by which a kernel divides a number below [2{^32}] by
    [d] (device.h's [divide]): a multiplier in its low 32 bits and a shift
    above, by Granlund and Montgomery's division by an invariant integer; [0]
    where [d] is not positive or at least [2{^32}], which no kernel reads. *)

val finish : ?most:int -> builder -> t
(** [finish ~most b] is [b]'s call. Its tickets slot is {!Tickets} where it took
    at most {!kept_tickets} bytes, and otherwise a {!Fresh} buffer that launches
    it puts first zero, each within {!allowed}[ ()].

    Its launches, those put first included, are cut in order into parts: each
    part takes the launches after the one before, as many as keep it within
    {!part_launches} launches and {!part_allowed}[ ()] seconds together, and,
    after the first, within the first's bytes of parameters, each launch's
    rounded up to 256 bytes, and within the first's bytes of values, [32 + 16 w]
    bytes for each launch of [w] words, so that a later part needs no more of
    the device's memory for its values than the first. A part takes at least one
    launch. At {!part_budget} and {!part_launches} the first part of a call of
    several holds at least 199 launches, more bytes than any one launch, so no
    later part passes it.

    Each part is keyed by its launches and kept where it has at most [most]
    launches (defaults to every count), those put first excepted, so that the
    submissions of calls whose launches grow with their extents stay few.

    Its workspace slot is {!Workspace} where its pieces took at most
    {!kept_workspace} bytes at once and it is of one part; a call that needs
    more, and a call of more than one part, takes its workspace as a {!Fresh}
    buffer of their peak, so that no other call writes it between two parts.

    The call holds at most [184 + 8 w + 40 r] bytes of host memory for each
    launch of [w] words and [r] refs, [32] more where its threads are not its
    kernel's, [128 + 16 s] for each part and [128 + 24 s] for the call, of [s]
    slots, beside its slots' own bytes: about 47 MB for {!part_launches}
    launches of 16 words and 4 refs.

    Raises [Invalid_argument] if [b] has no launch, if a slot of [b] is in no
    launch's refs, since {!key} holds refs alone and the call would bind more
    buffers than its key's submission takes, if a ref names no slot of [b], or
    if a ref writes a {!Constant}. *)

val finish_some : ?most:int -> builder -> t option
(** [finish_some ~most b] is [Some (finish ~most b)], or [None] if [b] has no
    launch. *)

(**/**)

val bound_fresh : workspace:int -> tickets:int -> unit
(** [bound_fresh ~workspace ~tickets] makes {!finish} take a {!Fresh} buffer
    past [workspace] bytes of workspace and [tickets] bytes of tickets, at most
    {!kept_workspace} and {!kept_tickets}, which it takes until then. For tests.

    Raises [Invalid_argument] if a bound is negative or above its kept size. *)

type traced = {
  kernel : Kernels.t;
  constants : int array;
  groups : int * int * int;
  resident : int;
  tickets : bool;
  words : Bytes.t;
}
(** The type for a launch {!traced} saw: its kernel and constants, its groups,
    the groups a core holds at once by its plan, whether it takes tickets, and
    its parameters as {!launch} took them, which nothing writes. *)

val traced : (unit -> 'a) -> 'a * traced list
(** [traced f] is [f ()] and the launches the calling domain planned or replayed
    ({!replay}) while [f] ran, in order. For tests, which check the path a call
    takes. *)

val unseen : (unit -> 'a) -> 'a
(** [unseen f] is [f ()], whose launches {!traced} does not see: those of a plan
    that no call runs. *)

val replay : t -> unit
(** [replay c] records [c]'s launches for {!traced} as {!launch} did when [c]
    was planned, those {!finish} put first excepted, so that a call that takes a
    plan made before traces as the call that planned it. *)

val epoch : unit -> int
(** [epoch ()] counts the changes to the bounds a plan reads: {!bound_fresh},
    {!bound_budget}, {!bound_parts} and {!bound_kept}, and nx_metal.mli's
    [bound_direct] and [bound_plans]. A plan is valid only at the epoch it was
    made at. *)

val renew : unit -> unit
(** [renew ()] counts a change to a bound a plan reads. *)

val bound_budget : float -> unit
(** [bound_budget s] makes {!allowed} [s], so that tests split work into
    launches with small arrays; {!budget} restores it. It applies to calls
    planned after it returns. For tests.

    Raises [Invalid_argument] if [s] is not positive or above {!budget}. *)

val bound_parts : launches:int -> seconds:float -> unit
(** [bound_parts ~launches ~seconds] makes {!finish} cut parts of at most
    [launches] launches and [seconds] seconds together, in place of
    {!part_launches} and {!part_budget}, so that tests reach calls of several
    parts with small arrays. Under such bounds a later part of one launch may
    hold more bytes than the first. It applies to calls planned after it
    returns. For tests.

    Raises [Invalid_argument] if [launches] is not in \[1; {!part_launches}\] or
    [seconds] is not positive or above {!part_budget}. *)

val part_allowed : unit -> float
(** [part_allowed ()] is {!part_budget}, or less in tests ({!bound_parts}). *)

val bound_kept : launches:int -> unit
(** [bound_kept ~launches] makes {!finish} keep only calls of at most [launches]
    launches, as well as at most its [most]; [max_int], the initial bound,
    leaves [most] alone. For tests.

    Raises [Invalid_argument] if [launches] is negative. *)

(**/**)

(** {1:submissions Submissions} *)

val key : part -> int array
(** [key p] names [p]'s submission: equal for parts whose launches have the same
    keys and refs, by the part's slots, in the same order, whatever their words'
    values. *)

val matches : int array -> part -> bool
(** [matches k p] is [key p = k], and allocates nothing. *)

val parts : (launch -> Rig.Image.func) -> part -> Rig.Submission.part array
(** [parts f p] is the parts of [p]'s submission, one for each launch, [f l]
    being the function of launch [l]'s key. A launch of a specialised step is
    labelled by its key ({!label}), so that a profile names the step. *)

val label : Kernels.t -> int array -> string
(** [label k cs] names the key of [k] and [cs]: [k]'s name, then each constant
    in decimal, separated by spaces. *)

val args : part -> Rig.Submission.args array
(** [args p] is the values of [p]'s launches, in order, as a submit of a
    submission of [p]'s {!parts} passes them: each launch's groups, its threads,
    its threadgroup memory bytes, and its words, little-endian, as rig reads
    them. *)
