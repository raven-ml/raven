(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Donation: handles over a donor's memory, and the one protocol by which an
    operation or a function of several operations consumes them.

    A {e donor} is the [Array] or [Shards] value that {!donate} gives up. A
    {e handle} is a value over a donor's memory that one consumer reads
    ({!Value.Handle}).

    A {e consumer} is the function the deaths name: an eager operation, which
    [Eval.run] computes, or a function of several operations. Each consumes
    the handles among its operands through {!with_consumer}, by these steps:
    + {!with_consumer} checks that each operand lives. An operation checks
      its rule and its route before it calls {!with_consumer}, so that one
      they refuse kills nothing. A function of several operations checks
      nothing more, since its operations run in an order it fixes.
    + It {e claims} the handles, which kills their donors for good, naming
      the consumer.
    + The consumer reads each handle as its arrays ({!called}, {!arrays}). A
      donor beside one stays dead, and the consumer reads it by its arrays
      alone.
    + {!reuse} gives a result the memory of a claimed handle that nothing
      else reads.
    + The claim {e ends} when the consumer returns or raises, which frees the
      memory no other value reads and no result holds.

    A donor dies once, at the first claim of a handle of it ({!Value.death}),
    and stays dead whatever its consumer does after. Any other reader, a
    function of several operations that consumes nothing or an
    interpretation, reads a handle as its arrays, shared ({!lent}), and kills
    nothing.

    A consumer claims its handles in operand order. Racing consumers on
    several domains for one donor is a misuse, after which any of them may
    have raised and any of the donors they claimed may be dead.

    This module lies above {!Prim}; [Eval], [Nx] and the modules [Nx] is
    made of call it. *)

(** {1:handles Handles} *)

val donate : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [donate ~by x] is a handle over [x]'s memory that one consumer reads
    ({!Value.Handle}): [x] lives until a consumer claims a handle of it, then
    dies naming the consumer, which reads [x] as it was when called. An
    operation may write its first result into the memory ({!reuse}). A
    movement passes a new handle of [x] on instead and kills nothing: the
    handle it read and the one it passed on die with [x]. A handle or a
    traced [x] is [x].

    Raises [Invalid_argument] naming [by] if [x] is dead. *)

val live : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [live x] is a handle's arrays as a live value; any other value as it
    is. *)

val is_handle : ('v, 's, 'd) Value.t -> bool
(** [is_handle x] is [true] iff [x] is a handle ({!Value.Handle}), live or
    not. *)

val lent : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [lent x] is the arrays of a handle as a live value, which its reader may
    keep or return: their memory is shared ({!Rig.Claim.share}), so no later
    donation writes there, and the handle's donor lives on. Any other value is
    itself.

    Raises [Invalid_argument] as {!Rig.Claim.share} does if a consumer took
    the memory meanwhile. *)

val read : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [read ~by x] is the operand [x] as [by], a function of several operations
    that consumes nothing, reads it: {!lent}[ x]. Raises [Invalid_argument]
    naming [by] if [x] is dead. *)

(** {1:consuming Consuming}

    A claim reads its consumer's operands by their positions in the list
    {!with_consumer} takes. Every read through a claim raises
    [Invalid_argument] naming its consumer once the claim has ended.

    When a claim ends, the memory of each handle it claimed that no other
    value reads, every live shard of a value or none, and that no {!reuse}
    took, is consumed and freed, so that it returns to its device once the
    work on it is done; its buffers stay dead naming the consumer. Memory
    that several handles lie over, as two handles of one donor may, is
    consumed once. Shards that share memory with one another, as borrows of
    one allocation do, are read by each other and consumed by none. Memory a
    {!reuse} took is the result's where the consumer returns. Where it
    raises, no result holds that memory, and it is freed too, unless a later
    consumption killed it. *)

type 'd claim
(** The type for one consumer's claim on the handles among its operands, of
    brand ['d]. *)

val with_consumer : by:string -> 'd Value.any list -> ('d claim -> 'a) -> 'a
(** [with_consumer ~by xs f] checks that each of [xs] lives, claims every
    handle among them for their consumer [by], in their order, each killing
    its donor naming [by], and is [f c] for that claim [c], which ends when
    [f] returns or raises. Two handles of one donor (the same handle twice,
    two donations of one value, a handle and one a movement passed on) are
    read as the donor beside its handle is, and the donor dies once, at the
    first. [f] must not keep an array it reads by [c], or a value over one,
    past its return, but those {!take} gave where it returns.

    Raises [Invalid_argument] naming [by] and the earlier consumer for a dead
    value, claiming nothing, and for a donor that died to another domain's
    consumer while it claimed, leaving dead the donors it claimed before.
    Raises what [f] raises. *)

val called : 'd claim -> 'r Value.prim -> 'r Value.prim
(** [called c op] is [op] with each handle as its arrays, a live value
    ({!live}), and any other operand as it is, a dead donor beside a handle
    included. [op]'s operands must be those [c] claimed. *)

val arrays : 'd claim -> int -> Nx_array.any array
(** [arrays c i] is operand [i]'s arrays, one per device, shared with nothing.

    Raises [Invalid_argument] naming [c]'s consumer for a traced operand and
    a position no operand has. *)

val reuse :
  'd claim ->
  own:int list ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  (Rig.Buffer.t * ('v, 's) Nx_array.t) option
(** [reuse c ~own dt l] is the memory of the first handle [c] claimed at a
    position of [own] that may hold a result of [dt] laid out as [l]: on one
    device, its array of [dt] and [l] over all of its memory, each operand at
    a position outside [own] apart from that memory, each at a position of
    [own] reading it as the handle's array does, the memory no other value
    reads (rig holds it exclusive), and no earlier [reuse] of [c] having taken
    it. The memory is then consumed for [c]'s consumer, and the answer is the
    handle's buffer, dead from then on, and an array of [dt] and [l] over the
    memory, the result's. [None] otherwise.

    [own] must hold only positions the consumer reads at the result's own
    index, each before it writes the result there, as the positions
    {!Prim.own_index} gives an operation.

    Raises [Invalid_argument] naming [c]'s consumer for a position of [own] no
    operand has. *)

val take :
  'd claim ->
  int ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  ('v, 's) Nx_array.t option
(** [take c i dt l] is the array {!reuse}[ c ~own:[i] dt l] gives: operand
    [i]'s memory for a result of [dt] laid out as [l], through which alone
    the consumer then reads the operand, each element before it writes over
    it. [None] where {!reuse} is.

    Raises [Invalid_argument] as {!arrays} does. *)
