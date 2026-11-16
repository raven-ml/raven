(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Calls of the code object's kernels: one submission each.

    A family plans a kernel call that nx.amd found nothing to refuse in as
    one {e call}: launches of the code object's kernels, run in order in
    one submission, and the {e slots} they address, the buffers a submit
    passes. A plan makes no buffer and touches no device: a slot names an
    array's buffer, the device's workspace or tickets, a buffer of a size
    the door makes once it accepts the call and frees once the call is
    submitted, or a constant the door finds or makes once it accepts the
    call, so a plan the door abandons leaves nothing behind.

    A launch's parameters are 64-bit words, which its kernel reads at the
    indices {!Kernels} names. A {e ref} is a word that holds a byte offset
    into a slot's buffer, which rig turns into that byte's address when
    the work runs, and orders the work by.

    Calls whose launches have the same kernels and refs, whatever their
    words' values, take runs of one submission ({!key}), so a device keeps
    one submission for each, which a later call writes its words into. *)

(** {1:slots Slots} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for bytes outside the OCaml heap. *)

(** The type for the buffers a call addresses. *)
type slot =
  | Array of Rig.Buffer.t  (** An operand's or a destination's buffer. *)
  | Workspace
      (** The device's workspace: bytes of no value at the call's start,
          which no other call writes while the call runs. *)
  | Tickets
      (** The device's tickets: 32-bit words that are zero at the call's
          start, and that every launch leaves zero. *)
  | Fresh of int
      (** A buffer of this many bytes, of no value at the call's start,
          which the door makes once it accepts the call and frees once it
          has submitted it. *)
  | Constant of { key : string; length : int; bytes : unit -> bigbytes }
      (** A buffer of [length] bytes holding [bytes ()], which no launch
          writes, and which are the same for every call that names [key].
          Once it accepts the call, the door takes the device's buffer of
          [key], or calls [bytes] once, uploads its result into a new
          buffer and waits for the upload. [bytes] may be called from any
          domain, and from several at once for one key. The door raises
          [Invalid_argument] where the buffer of [key] or [bytes ()] is
          not of [length] bytes, so a key two tables share by mistake
          raises where a kernel would read past a buffer.
          The device keeps it for later calls, with at most
          {!kept_constant_count} constants taking at most {!kept_constants}
          bytes together, and otherwise the door frees it once it has
          submitted the call. *)

val kept_workspace : int
(** [kept_workspace] is 64 MiB, the most bytes of {!Workspace} a call
    takes. A call that needs more takes a {!Fresh} buffer instead. *)

val kept_tickets : int
(** [kept_tickets] is 4 MiB, the most bytes of {!Tickets} a call takes. A
    call that needs more takes a {!Fresh} buffer, which its first launch
    zeroes. *)

val kept_constants : int
(** [kept_constants] is 64 MiB, the most bytes of {!Constant}s a device
    keeps. A constant the device no longer keeps returns once collected
    ({!Rig.section-reclaim}), since a call on another domain may have just
    taken it. *)

val kept_constant_count : int
(** [kept_constant_count] is 16, the most {!Constant}s a device keeps,
    so that a call finds its own among few. *)

val kept_submissions : int
(** [kept_submissions] is 64 MiB, the most bytes of submissions a device
    keeps for later calls, counted as {!t}'s [bytes] are. *)

val kept_count : int
(** [kept_count] is 64, the most submissions a device keeps whose calls
    launch one kernel first, so that a call finds its own among few. *)

val name_bytes : int
(** [name_bytes] is 30, the most bytes of a kernel's name ({!Kernels.name})
    that {!t}'s [bytes] count for rig's copy of it in a launch. *)

(** {1:launches Launches} *)

type ref = { word : int; slot : int; write : bool }
(** The type for refs: the word that holds a byte offset into the buffer of
    [slot], which the launch writes iff [write]. *)

type launch = {
  kernel : Kernels.t;
  groups : int * int * int;  (** The groups along x, y and z. *)
  shared : int;  (** The dynamic LDS bytes of a group. *)
  words : Bytes.t;
      (** The parameters, {!Kernels.words}[ kernel] little-endian words. *)
  used : int;
      (** The count of the words' first ones that the kernel reads, which
          {!write} alone stores. *)
  refs : ref array;
}
(** The type for launches. A group has {!Kernels.threads}[ kernel]
    threads along x. *)

type t = private {
  launches : launch array;  (** In order, at least one. *)
  slots : slot array;  (** The slots refs name, by index. *)
  workspace : int;  (** The bytes of {!Workspace} the call takes. *)
  tickets : int;  (** The bytes of {!Tickets} the call takes. *)
  bytes : int;
      (** The bytes a device counts for the call's submission: 1 KiB, and
          [32 (9 + ⌈w / 2⌉ + r)] for each launch of a kernel of [w] words
          ({!Kernels.words}) and of [r] refs. *)
  kept : bool;
      (** Whether a device keeps the call's submission for later calls of
          its key ({!key}): whether [bytes] are at most
          {!kept_most}[ ()]. *)
}
(** The type for calls. *)

(** {1:building Building} *)

type builder
(** The type for calls being planned. *)

val make : unit -> builder
(** [make ()] is a call of no launch and no slot. *)

val slot : builder -> Rig.Buffer.t -> int
(** [slot b x] is the index of a slot of [b] that holds [x]: one slot for
    each buffer, by physical equality. *)

val take : builder -> int -> int
(** [take b n] is the byte offset into [b]'s workspace slot of a piece of
    [n] bytes, the first on 256 bytes past the pieces [b] holds. *)

val mark : builder -> int
(** [mark b] is the end of the pieces [b] holds. *)

val release : builder -> int -> unit
(** [release b m] gives back the pieces taken since [mark b] was [m]: the
    launches planned so far run before any later one, so a later launch
    may use them again. *)

val workspace : builder -> int
(** [workspace b] is the index of [b]'s workspace slot, whose pieces
    {!take} gives. *)

val tickets : builder -> int -> int
(** [tickets b n] is the byte offset into [b]'s tickets slot ({!tickets_slot})
    of [n] ticket words for the launch planned next. Each is zero when that
    launch starts, since every launch leaves the words it takes zero, so
    every launch takes them from the slot's start. *)

val tickets_slot : builder -> int
(** [tickets_slot b] is the index of [b]'s tickets slot. *)

val constant :
  builder -> key:string -> length:int -> (unit -> bigbytes) -> int
(** [constant b ~key ~length bytes] is the index of a slot of [b] that
    holds the {!Constant} [key] of [length] bytes [bytes ()]: one slot for
    each key. A key names its bytes alone, across families: a family's
    keys start with its name, as ["fft "].

    Raises [Invalid_argument] if [length] is negative, or if [b] names
    [key] with another length. *)

val words : Kernels.t -> Bytes.t
(** [words k] is zero parameters for a launch of [k]. *)

val set : Bytes.t -> int -> int -> unit
(** [set w i v] stores [v] as word [i] of [w]. *)

val small : int -> bool
(** [small n] is [true] iff an index space of [n] indices is small,
    [n < 2³²]: device.h's [split] then divides its indices in 32 bits by
    its axes' divisor words. *)

val axis : Bytes.t -> int -> int -> unit
(** [axis w i n] stores the head of the record of an axis of extent [n]
    from word [i] of [w] ({!Kernels.axis_n}): [n], then its divisor word
    ({!divisor}), [0] where [n] is [0] or at least [2³²], which a small
    space never divides by. The operands' strides follow, from
    {!Kernels.axis_strides}.

    Raises [Invalid_argument] if [n] is negative. *)

val layout : Bytes.t -> int -> Nx_array.Layout.t -> int
(** [layout w i l] stores [l] as the layout words from word [i] of [w]
    ({!Kernels.layout_rank}): its rank, its offset, whether it is
    {!small}, then each axis's record ({!axis}) of its extent and stride,
    and is the index of the word past them. That index is the [used] of
    {!launch} only where the layout is the last words its kernel
    reads. *)

val stride_groups : int
(** [stride_groups] is 1024, the most groups of a grid whose work-items
    stride ({!grid}). Each group reads the launch's words across PCIe, and
    past the groups the GPU holds at once, 1024 groups of 64 work-items or
    more on the R9700, a striding grid gains no bandwidth. *)

val in_band : Kernels.t -> int -> bool
(** [in_band k g] is [true] iff [g] groups of [k] lie in the {e slow
    band}: [k] takes at most 96 VGPRs ({!Kernels.vgprs}), so that a SIMD
    holds 16 of its waves of 32 work-items, and its [g] groups, dealt one
    to each of 4 shader engines, fill the busiest engine's 512 wave slots,
    or twice that, to within 4 waves. On the R9700 such a launch is
    slower, as one of 253 to 256 or 509 to 512 groups of 256 work-items
    is, or of 1017 to 1024 groups of 64; the bands of group sizes other
    than 64, 128 and 256 are extrapolated. The band also slows
    a grid that counts its tiles, which may choose its tile around it. *)

val grid : ?per:int -> Kernels.t -> int -> int
(** [grid ~per k n] is the groups along x of a launch of [k] whose
    work-items stride over [n] units, [per] a group (defaults to
    {!Kernels.threads}[ k], one a work-item): enough for every unit, at
    most {!stride_groups}, at least one, and never in the slow band
    ({!in_band}). Where the count [n] needs is in the band, it is the
    first count past the band, or the last before it where the first is
    above {!stride_groups}. A kernel whose grid counts its tiles does not
    stride and sizes its own.

    The count depends on [k]'s VGPRs as well as on [n], so no float
    association may depend on it.

    Raises [Invalid_argument] if [per] is not positive. *)

val divisor : int -> int64
(** [divisor d] is the word by which device code divides an unsigned
    32-bit integer by [d] without a division instruction, which RDNA4
    lacks: its multiplier [m] in the low 32 bits and its shift [s] above,
    so that [n / d] is [(⌊n m / 2³²⌋ + n) / 2^s] (device.h's [divide]).

    Raises [Invalid_argument] unless [1 <= d < 2³²]. *)

val shared_most : Kernels.t -> int
(** [shared_most k] is the most dynamic LDS bytes a group of a launch of
    [k] may take: the 64 KiB a group of RDNA4 holds, less [k]'s static LDS
    ({!Kernels.lds}). *)

val launch :
  builder ->
  Kernels.t ->
  groups:int * int * int ->
  ?shared:int ->
  ?used:int ->
  Bytes.t ->
  ref array ->
  unit
(** [launch b k ~groups ~shared ~used w refs] adds a launch of [k] over
    [groups] with [shared] dynamic LDS bytes (defaults to [0]), words [w]
    of which [k] reads the first [used] (defaults to
    {!Kernels.words}[ k]), and refs [refs] to [b], after its other
    launches. A call writes only the words read, so its cost on the host
    follows [used].

    Raises [Invalid_argument] if [shared] is negative or above
    {!shared_most}[ k], since rig would launch a group asking more LDS
    than the GPU has, if [used] is negative or above
    {!Kernels.words}[ k], or if a ref's word is not among the first
    [used]. *)

val zero : builder -> slot:int -> at:int -> int -> unit
(** [zero b ~slot ~at n] adds to [b] a launch of {!Kernels.Zero} that
    zeroes the [n] bytes of slot [slot] from byte [at], after its other
    launches.

    Raises [Invalid_argument] if [n] is negative or not a multiple of
    4. *)

val finish : builder -> t
(** [finish b] is [b]'s call. Its workspace slot is {!Workspace} where
    its pieces took at most {!kept_workspace} bytes at once, and a
    {!Fresh} buffer of their peak otherwise; its tickets slot is
    {!Tickets} where it took at most {!kept_tickets} bytes, and otherwise
    a {!Fresh} buffer that a launch it puts first zeroes.

    Its slots are those of [b] that a launch's ref names, in the order [b]
    named them, each ref renumbered to its slot's index among them: {!key}
    holds refs alone, so a call binds the buffers its key's submission
    takes. A workspace or tickets slot no ref names takes no bytes.

    Raises [Invalid_argument] if [b] has no launch, if a ref names no slot
    of [b], or if a ref writes a {!Constant}. *)

val finish_some : builder -> t option
(** [finish_some b] is [Some (finish b)], or [None] if [b] has no
    launch. *)

(**/**)

val bound_fresh : workspace:int -> tickets:int -> unit
(** [bound_fresh ~workspace ~tickets] makes {!finish} take a {!Fresh}
    buffer past [workspace] bytes of workspace and [tickets] bytes of
    tickets, at most {!kept_workspace} and {!kept_tickets}, which it takes
    until then. For tests.

    Raises [Invalid_argument] if a bound is negative or above its kept
    size. *)

val past_band : threads:int -> int -> int
(** [past_band ~threads g] is [g] stepped out of the slow band as {!grid}
    steps the groups of a kernel of [threads] work-items a group and at
    most 96 VGPRs. For tests.

    Raises [Invalid_argument] if [threads] is not positive. *)

val bound_grid : groups:int -> unit
(** [bound_grid ~groups] makes {!grid} give at most [groups] groups, so
    that tests launch a striding kernel at a count of their choosing;
    [max_int], the initial bound, restores {!grid}'s rule. It applies to
    calls planned after it returns. For tests.

    Raises [Invalid_argument] if [groups] is not positive. *)

type traced = {
  kernel : Kernels.t;
  groups : int * int * int;
  tickets : bool;
  shared : int;
}
(** The type for a launch {!traced} saw: its kernel, its groups, whether it
    takes tickets, and the bytes of LDS it takes beside its kernel's
    own. *)

val traced : (unit -> 'a) -> 'a * traced list
(** [traced f] is [f ()] and the launches the calling domain planned while
    [f] ran, in order. For tests, which check the path a call takes. *)

val bound_kept : int -> unit
(** [bound_kept n] makes a device keep at most [n] bytes of submissions,
    in place of {!kept_submissions}, and {!finish} keep only calls of at
    most [n] bytes. For tests.

    Raises [Invalid_argument] if [n] is negative or above
    {!kept_submissions}. *)

val kept_most : unit -> int
(** [kept_most ()] is the most bytes of submissions a device keeps:
    {!kept_submissions} but under {!bound_kept}. *)

(**/**)

(** {1:submissions Submissions} *)

val key : t -> int array
(** [key c] names [c]'s submission: equal for calls whose launches have the
    same kernels and refs, in the same order, whatever their words'
    values. *)

val matches : int array -> t -> bool
(** [matches k c] is [key c = k], and allocates nothing. *)

val parts : Rig.Image.func array -> t -> Rig.Submission.part array
(** [parts fs c] is the parts of [c]'s submission, [fs] holding each kernel's
    function at its {!Kernels.index}. *)

val write : t -> Rig.Submission.Run.t -> Rig.Submission.t -> unit
(** [write c r s] stores each launch's geometry and the words it reads
    into [r]'s blocks of [s], a submission of [c]'s {!parts}. The other
    words of a block keep what an earlier call stored. *)
