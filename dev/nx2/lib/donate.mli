(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Donation: handles over a value's memory that one operation reads, and their
    consumption by the eager engine.

    A donor is offered under a key of its own, which it keeps for life and
    every handle of every donation of it holds. A handle has no state of its
    own: it lives while its donor does. Only an operation the engine ({!Exec})
    runs consumes handles. The engine checks the operation's operands, claims
    its handles, which kills their donors for good, writes the operation's
    first result into a handle's memory where {!reuse} allows, and ends the
    handles once it has read them ({!consumed}). A donor stays dead whatever
    the kernels answer after the claim.

    A function of several operations, and an interpretation, read a handle as
    its arrays, shared ({!lent}), and kill nothing. Claims follow no order:
    racing consumers on several domains for one donor is a misuse, after which
    any of them may have raised and any of the donors they claimed may be
    dead. *)

(** The type for a donated operand and its position among an operation's
    operands. *)
type handle = Handle : int * ('v, 's, 'd) Value.t -> handle

val donate : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [donate ~by x] is a handle over [x]'s memory that one operation reads
    ({!Value.Donated}): [x] lives until an eager operation claims the handle,
    then dies with it, naming the operation; the operation reads [x] as it was
    when called. An operation that reads the handle only at its first result's
    own index (a map's plain load, a scatter's target, a contraction's [init],
    an assembly's first piece over the whole result) writes that result into
    the memory, where rig holds it exclusive (no other value reads it), it has
    the result's dtype and C-contiguous layout, and the operation reads none
    of it at other indices through another operand; the memory is consumed
    either way where no other value reads it. A movement passes a new handle
    of [x] on instead and kills nothing: the handle it read and the one it
    passed on die with [x]. A constant or a traced [x] is [x].

    Raises [Invalid_argument] naming [by] if [x] is dead. *)

val live : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [live x] is a handle's arrays as a live value; any other value as it
    is. *)

val is_handle : ('v, 's, 'd) Value.t -> bool
(** [is_handle x] is [true] iff [x] is a handle ({!Value.Donated}), live or
    not. *)

val lent : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [lent x] is the arrays of a handle as a live value, which its reader may
    keep or return: their memory is shared ({!Rig.Claim.share}), so no later
    donation writes there, and the handle's donor lives on. Any other value is
    itself.

    Raises [Invalid_argument] as {!Rig.Claim.share} does if a consumer took
    the memory meanwhile. *)

val read : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [read ~by x] is the operand [x] as [by], a function of several
    operations, reads it: {!lent}[ x]. Raises [Invalid_argument] naming [by]
    if [x] is dead. *)

(** {1:engine The engine's steps} *)

val handles : 'd Value.any list -> handle list
(** [handles xs] is the handles among [xs], the first of each donor's. A
    second handle of one donor, the same handle twice, two donations of one
    value or a handle a movement passed on, is read as the donor beside its
    handle is: [claim] kills the donor once and [consumed] consumes its memory
    once. *)

val claim : by:string -> handle list -> unit
(** [claim ~by hs] claims [hs], from {!handles}, for their consumer [by]: each
    donor, and with it each of its handles, dies naming [by], for good. Raises
    [Invalid_argument] naming [by] and the earlier consumer for a donor that
    died first, on any domain; the donors claimed before it stay dead. *)

val unhandled : handle list -> 'r Value.prim -> 'r Value.prim
(** [unhandled hs op] is [op] with each operand as the consumer of the handles
    [hs] reads it, before it claims them: a handle as its arrays, a donor of
    one of [hs] as a live value, and any other value as it is. *)

val reuse :
  by:string ->
  'r Value.prim ->
  handle list ->
  (int * Rig.Buffer.t * Rig.Buffer.t) option
(** [reuse ~by op hs] is the claimed handle among [hs] whose memory takes
    [op]'s first result, if one may: on one device, at a position
    {!Prim.own_index} gives, of the result's dtype and layout over all of its
    memory, beside operands at its placement none of which [op] reads at
    other indices from that memory, and which rig holds exclusive. Its memory is then consumed for [by], and the
    answer is its position, its buffer and the live buffer over its memory. *)

val consumed : by:string -> reused:Rig.Buffer.t list -> handle list -> unit
(** [consumed ~by ~reused hs] ends the claimed handles [hs] once [by] read
    them: memory no other value reads, every shard of a value or none, and
    not among [reused], is consumed and freed, so that it returns to its
    device once the work on it is done; its buffers stay dead naming [by]. Shards that share memory with one
    another, as borrows of one allocation do, are read by each other and
    consumed by none. *)
