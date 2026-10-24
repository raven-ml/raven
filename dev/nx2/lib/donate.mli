(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Donation: handles over a value's memory that one operation reads, and their
    consumption by the eager engine.

    A donor is offered under a key of its own, which it keeps for life and
    every handle of every donation of it holds. A handle has no state of its
    own: it lives while its donor does. Only computation consumes handles: an
    operation the engine ({!Exec}) runs, and a function of several
    operations. The engine checks an operation's operands, claims its handles,
    which kills their donors, writes the operation's first result into a
    handle's memory where {!reuse} allows, and ends the handles once it has
    read them ({!consumed}). An operation that raises before it writes or
    consumes their memory gives them back ({!unclaim}); one that raises once
    it consumed a handle's memory for its result leaves its donors dead: rig's
    consumption is one-way. A function that reads its operands in several
    operations claims them once, before the first ({!consume}), and reads
    them through {!read}.

    An interpretation reads a handle, or a claimed value, as its arrays
    ({!lent}) and kills nothing. Claims follow no order: racing consumers on
    several domains for one donor is a misuse, after which any of them may
    have raised and any of the donors they claimed may be dead. *)

(** The type for a donated operand and its position among an operation's
    operands. *)
type handle = Handle : int * ('v, 's, 'd) Value.t -> handle

val donate : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [donate ~by x] is a handle over [x]'s memory that one operation reads
    ({!Value.Donated}): [x] lives until an eager operation consumes the
    handle, then dies with it, naming the operation, unless the operation
    gives it back ({!unclaim}); the operation reads [x] as it was when called.
    An operation that reads the handle only at its first result's own index (a
    map's plain load, a scatter's target, a contraction's [init], an
    assembly's first piece over the whole result) writes that result into the
    memory, where rig holds it exclusive (no other value reads it), it has the
    result's dtype and C-contiguous layout, and the operation reads none of it
    at other indices through another operand; the memory is consumed either
    way where no other value reads it. A movement passes a new handle of [x] on instead and kills
    nothing: the handle it read and the one it passed on die with [x]. A
    constant or a traced [x] is [x].

    Raises [Invalid_argument] naming [by] if [x] is dead. *)

val live : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [live x] is a handle's arrays as a live value; any other value as it
    is. *)

val is_handle : ('v, 's, 'd) Value.t -> bool
(** [is_handle x] is [true] iff [x] is a handle ({!Value.Donated}), live or
    not. *)

val lent : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [lent x] is the arrays of a handle, or of a value {!read} gave, as a live
    value, for an interpretation, which may keep it: their memory is shared
    ({!Rig.Claim.share}), so no later donation writes there. A handle's donor
    lives on. Any other value is itself. *)

val lends : ('v, 's, 'd) Value.t -> bool
(** [lends x] is whether {!lent}[ x] is not [x]. *)

val consume : by:string -> 'd Value.any list -> handle list
(** [consume ~by xs] is the handles among [xs], claimed for the consumer [by]
    of the operands [xs], which reads them through {!read}. It raises before
    it claims any, as {!handles} does, then naming [by] for an operand that
    died, operand [i] being [xs]' element [i]. *)

val consume_list : by:string -> 'd Value.any list -> 'd Value.any list
(** [consume_list ~by xs] is [xs] as they were when [by], which is no
    operation, consumed them ({!consume}): each handle and each of their donors
    among [xs] {!read} as a live value. *)

(** {1:reads A consumer's reads} *)

type reads
(** The type for the reads of a consumer of several operations. *)

val reads : handle list -> reads
(** [reads hs] reads for the consumer of the claimed handles [hs], none read
    yet. *)

val read : reads -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [read r x] is {!called}[ hs x], [hs] [r]'s handles, each handle and donor
    of [hs] as a new {!Value.Claimed} value: an operation reads it as live and
    claims nothing, {!donate} gives it back as it is, and an interpretation
    receives it {!lent}. A second read of one donor shares its memory
    ({!Rig.Claim.share}): the consumer may give back both values, which no
    donation then writes under the other. Domains may read through one [r]
    at once. *)

val release : reads -> unit
(** [release r] makes [r]'s reads live, once their consumer returned or
    raised. *)

(** {1:engine The engine's steps} *)

val handles : by:string -> 'd Value.any list -> handle list
(** [handles ~by xs] is the handles among [xs]. Raises [Invalid_argument]
    naming [Nx.donate] and [by] for two handles of one donor or over one
    memory: the same handle twice, two donations of one value, or one handle
    and a value a movement passed on. *)

val claim : by:string -> handle list -> unit
(** [claim ~by hs] claims [hs], from {!handles}, for their consumer [by]: each
    donor, and with it each of its handles, dies naming [by]. Raises
    [Invalid_argument] naming [by] and the earlier consumer for a donor that
    died first, on any domain; the donors claimed before it stay dead. *)

val called : handle list -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [called hs x] is [x] as it was when the consumer of the claimed handles
    [hs] was called: a handle as its arrays, a donor of one of [hs] as a live
    value, and any other value as it is. *)

val unhandled : handle list -> 'r Value.prim -> 'r Value.prim
(** [unhandled hs op] is [op] with each operand {!called}[ hs]. *)

val unclaim : handle list -> unit
(** [unclaim hs] gives back the claimed handles [hs], for a consumer that
    raised before it wrote or consumed their memory: each donor, and with it
    each of its handles, lives again, offered under its key. *)

val holds :
  Nx_array.Dtype.any -> Nx_array.Layout.t -> ('v, 's) Nx_array.t -> bool
(** [holds dt l a] is whether [a] may hold a result of [dt] and layout [l] in
    its memory: of that dtype and layout, over all of its memory. *)

val take : by:string -> Rig.Buffer.t -> Rig.Buffer.t option
(** [take ~by b] is the live buffer over [b]'s memory, consumed for [by],
    where rig holds it exclusive: no other value reads it. [None] otherwise,
    [b] untouched. *)

val reuse :
  by:string ->
  'r Value.prim ->
  handle list ->
  (int * Rig.Buffer.t * Rig.Buffer.t) option
(** [reuse ~by op hs] is the claimed handle among [hs] whose memory takes
    [op]'s first result, if one may: on one device, at a position
    {!Prim.own_index} gives, {!holds} the result, beside operands at its
    placement none of which [op] reads at other indices from that memory, and
    which rig holds exclusive. Its memory is then consumed ({!take}), and the
    answer is its position, its buffer and the live buffer over its memory. *)

val consumed : by:string -> reused:Rig.Buffer.t list -> handle list -> unit
(** [consumed ~by ~reused hs] ends the claimed handles [hs] once [by] read
    them: memory no other value reads, every shard of a value or none, and
    not among [reused], is consumed and freed, so that it returns to its
    device once the work on it is done. Shards that share memory with one
    another, as borrows of one allocation do, are read by each other and
    consumed by none. *)
