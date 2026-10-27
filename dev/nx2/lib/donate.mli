(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Donation: handles over a donor's memory, and the one protocol by which an
    operation or a function of several operations consumes them.

    A {e donor} is the [Array] or [Shards] value that {!donate} gives up. A
    {e handle} is a value over a donor's memory that one consumer reads
    ({!Value.Handle}). Every handle of every donation of a donor holds it, and
    two handles of one donor are told by [==] on it. A handle lives while its
    donor does. A donor dies once, when a compare-and-set changes its word
    from [Live] to [Died].

    A {e consumer} is the function the deaths name: an eager operation, which
    [Eval.run] computes, or a function of several operations, which
    {!with_consumer} runs. Both consume the handles among their operands by
    the same steps:
    + The consumer checks its operands. An operation checks its rule and its
      route too, so that one they refuse kills nothing. A function of several
      operations checks only that each operand lives, since its operations
      run in an order it fixes.
    + {!claim} claims the handles, which kills their donors for good, naming
      the consumer.
    + The consumer reads each handle as its arrays and each donor beside one
      as live ({!called}, {!arrays}).
    + {!reuse} gives a result the memory of a claimed handle that nothing
      else reads.
    + {!consumed} ends the claim when the consumer returns or raises, which
      frees the memory no other value reads and no result took.

    A donor stays dead whatever the consumer does after the claim. Any other
    reader, a function of several operations that consumes nothing or an
    interpretation, reads a handle as its arrays, shared ({!lent}), and kills
    nothing.

    A consumer claims its handles in operand order. Racing consumers on
    several domains for one donor is a misuse, after which any of them may
    have raised and any of the donors they claimed may be dead.

    This module calls {!Prim}. [Eval], [Nx] and the modules [Nx] is made of
    call it. *)

(** {1:handles Handles} *)

val donate : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [donate ~by x] is a handle over [x]'s memory that one consumer reads
    ({!Value.Handle}): [x] lives until a consumer claims a handle of it, then
    dies naming the consumer, which reads [x] as it was when called. An
    operation may write its first result into the memory ({!reuse}). A
    movement passes a new handle of [x] on instead and kills nothing: the
    handle it read and the one it passed on die with [x]. A handle, a constant
    or a traced [x] is [x].

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

    The steps of the protocol above, which an operation and a function of
    several operations take alike. A claim reads its consumer's operands by
    their positions in the list {!claim} takes. *)

type 'd claim
(** The type for one consumer's claim on the handles among its operands, of
    brand ['d]. *)

val claim : by:string -> 'd Value.any list -> 'd claim
(** [claim ~by xs] claims every handle among [xs] for their consumer [by], in
    their order, and each kills its donor naming [by], for good. Two handles
    of one donor (the same handle twice, two donations of one value, a handle
    and one a movement passed on) are read as the donor beside its handle is,
    and the donor dies once, at the first.

    Raises [Invalid_argument] naming [by] and the earlier consumer for a donor
    that died first, on any domain; the donors claimed before it stay dead. *)

val called : 'd claim -> 'r Value.prim -> 'r Value.prim
(** [called c op] is [op] with each operand as [c]'s consumer reads it: a
    handle as its arrays, a donor of one of [c]'s handles as a live value, and
    any other value as it is. [op]'s operands must be those [c] claimed. *)

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
    operand has, and once {!consumed} returned. *)

val consumed : 'd claim -> unit
(** [consumed c] ends [c] once its consumer returned or raised: the memory of
    each handle [c] claimed that no other value reads, every live shard of a
    value or none, and that no {!reuse} took, is consumed and freed, so that
    it returns to its device once the work on it is done; its buffers stay
    dead naming the consumer. Memory that several handles lie over, as two
    handles of one donor may, is consumed once. Shards that share memory with
    one another, as borrows of one allocation do, are read by each other and
    consumed by none. Memory a {!reuse} took is the result's. A consumer that
    raises before it makes its result must free that memory; [Eval] does, and
    {!with_consumer} leaves what {!take} gave [f] to [f]. *)

(** {1:consumers Functions of several operations} *)

val with_consumer : by:string -> 'd Value.any list -> ('d claim -> 'a) -> 'a
(** [with_consumer ~by xs f] checks that each of [xs] lives, then is [f c] for
    [c], the claim of [by] on the handles among them ({!claim}), and ends [c]
    ({!consumed}) when [f] returns or raises. [f] must not keep an array it
    reads by [c], or a value over one, past its return, but those {!take}
    gave.

    Raises [Invalid_argument] naming [by] for a dead value, claiming nothing,
    as {!claim} does, and what [f] raises. *)

val arrays : 'd claim -> int -> Nx_array.any array
(** [arrays c i] is operand [i]'s arrays, one per device, shared with nothing.

    Raises [Invalid_argument] naming [c]'s consumer for a constant or a traced
    operand, a position no operand has, and once {!consumed} returned. *)

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
