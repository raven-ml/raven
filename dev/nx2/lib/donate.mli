(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Donation: handles over a value's memory that one operation reads, and their
    consumption.

    A donor is offered under a key of its own, which it keeps for life and
    every handle of every donation of it holds. A handle has no state of its
    own: it lives while its donor does. A consumer claims its handles' donors
    in key order, without a lock: of consumers racing for the same handles, at
    least one succeeds, and each loser gives back what it claimed and raises
    naming the consumer it lost a donor to. The eager engine ({!Exec}) claims an
    operation's handles before it reads them, writes the operation's first
    result into a handle's memory where {!reuse} allows, and ends the handles
    once it has read them ({!consumed}). An operation that raises before it
    writes or consumes their memory gives them back ({!unclaim}); one that
    raises once it consumed a handle's memory for its result leaves its donors
    dead: rig's consumption is one-way. *)

val donate : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [donate ~by x] is a handle over [x]'s memory that one operation reads
    ({!Value.Donated}): [x] lives until that operation is called, then dies with
    the handle, naming the operation, unless the operation gives it back
    ({!unclaim}); the operation reads [x] as it was when called. An operation
    that reads the handle only at its first result's own index (a map's plain
    load, a scatter's target, a contraction's [init], an assembly's first piece
    over the whole result) writes that result into the memory, where rig holds
    it exclusive (no other value reads it), it has the result's dtype and
    C-contiguous layout, and the operation reads none of it at other indices
    through another operand; the memory is consumed either way where no other
    value reads it. A movement passes a new handle of [x] on instead and kills
    nothing: the handle it read and the one it passed on die with [x]. A
    constant or a traced [x] is [x].

    Raises [Invalid_argument] naming [by] if [x] is dead. *)

val live : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [live x] is a handle's arrays as a live value, for an interpretation that
    reads them; any other value as it is. *)

val is_handle : ('v, 's, 'd) Value.t -> bool
(** [is_handle x] is [true] iff [x] is a handle ({!Value.Donated}), live or
    not. *)

val consume : by:string -> 'r Value.prim -> 'r Value.prim
(** [consume ~by op] checks [op]'s operands and kills the donor of each handle
    among them, naming [by], as {!Exec.run} does before it reads them, a
    movement's too. It is [op] as it was when called: each handle as its
    arrays, and each of their donors that is an operand itself as a live value.
    Raises [Invalid_argument] naming [Nx.donate] for two handles of one donor,
    then naming [by] for an operand that died. *)

val resolve :
  by:string ->
  'd Value.any list ->
  (('v 's. ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t) -> 'a) ->
  'a
(** [resolve ~by xs k] is [k read], the call [by] consuming the donated
    handles among its operands [xs] once, for a call that reads an operand
    in more than one operation. [read] gives each handle's arrays, and each
    of their donors among [xs], as live values; any other value as it is.
    {!consume}'s checks run before [k], operand [i] being [xs]' element [i].
    The donors die, naming [by], once [k] returns; if [k] raises, they and
    their handles live on. [k]'s operations reuse none of their memory,
    which is freed once no value holds it.

    While [k] runs, a hold on the memory of the handles and of their donors
    among [xs] ({!Rig.Claim.hold}) shares it, so that a consumer that claims
    one of the handles meanwhile, on any domain, computes into fresh memory
    and overwrites nothing [k] reads; the claim after [k] then raises naming
    that consumer. Where a movement copied a handle, the donor's memory lies
    apart from the handle's, and both are held. The hold ends when [k]
    returns or raises. *)

val consume_list : by:string -> 'd Value.any list -> 'd Value.any list
(** [consume_list ~by xs] is {!consume} of a consumer [by] of the operands
    [xs], which is no operation: [xs] as they were when it was called, the
    donors of their handles dead. *)

(** {1:engine The engine's steps} *)

(** The type for a donated operand and its position among an operation's
    operands. *)
type handle = Handle : int * ('v, 's, 'd) Value.t -> handle

val handles : by:string -> 'd Value.any list -> handle list
(** [handles ~by xs] is the handles among [xs]. Raises [Invalid_argument]
    naming [Nx.donate] and [by] for two handles of one donor or over one
    memory: the same handle twice, two donations of one value, or one handle
    and a value a movement passed on. *)

val claim : by:string -> handle list -> unit
(** [claim ~by hs] claims [hs], from {!handles}, for their consumer [by], in
    their donors' key order: each donor, and with it each of its handles, dies
    naming [by]. Raises [Invalid_argument] naming [by] and the earlier
    consumer for a donor that died first, on any domain, having given back the
    handles it claimed. *)

val unhandled : handle list -> 'r Value.prim -> 'r Value.prim
(** [unhandled hs op] is [op] as it was when called, read after {!claim}
    claimed its handles [hs]: each handle as its arrays, and each donor of
    [hs] among [op]'s operands as a live value. *)

val unclaim : handle list -> unit
(** [unclaim hs] gives back the claimed handles [hs], for a consumer that
    raised before it wrote or consumed their memory: each donor, and with it
    each of its handles, lives again, offered under its key. *)

val claim_handle : by:string -> int -> ('v, 's, 'd) Value.t -> unit
(** [claim_handle ~by i x] is {!claim} of the one handle [x], operand [i];
    nothing where [x] is no handle. *)

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
