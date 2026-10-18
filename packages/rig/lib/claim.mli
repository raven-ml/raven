(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Claims: the implementation of {!Rig.Claim}.

    Each lifetime of an allocation has one claim, in its {!Def.alloc}, which
    every memory record over it shares: a claim word
    ({!Memory.section-claims}), a generation, and the reason of the last
    consumption. A read claim adds
    {!Memory.one_claim} to the word by compare-and-set, and is refused while
    the word is negative, that is while the memory is held exclusive.

    {!with_} takes a read claim on every buffer it is given. Then, for each
    donated buffer that spans its memory, it turns the word, if it holds that
    one claim and no bit, into {!Memory.exclusive}. {!consume} raises the
    generation from its buffer's own, which kills every buffer made before,
    then gives an exclusive word back to the one read claim, so the buffer
    it returns is claimed as any other. When {!with_}'s function returns or
    raises, each word still exclusive goes back to one read claim, and every
    read claim is released.

    {!consume} makes its check and its effects in one C step. The end of the
    claims waits for a step under way, so a consumption that races the end
    completes before any release, or raises. Of two consumptions of one
    buffer, the step lets one raise the generation, and the other raises.

    A claim is taken before the buffer is checked live. A consumption on
    another domain raises the generation before it gives the word back, so a
    compare-and-set that reads the word given back, or any later one, is
    followed by a check that sees the new generation. Every path that claims
    checks under its claim, since a check before the claim would pass a
    buffer that a consumption kills between the two. A claim refused by
    memory held exclusive that its with_ has since released is tried again.

    A hold ({!hold}) adds {!Memory.one_hold} to the word unless it is
    exclusive. A consumption of memory not held exclusive sets the word's
    lock bit if no hold holds the memory, raises the generation, and clears
    the bit; a hold waits while the bit is set and checks its buffer live
    once it holds. So a consumption either finds the hold and raises, or
    the hold finds the buffer dead. A release ({!release}) clears each
    hold's mark in its record and takes {!Memory.one_hold} off the word in
    one C step, so each hold ends once, whichever release, on any domain,
    or drain after the hold's collection comes first.

    {!with_} takes its claims in one C call, the last step before its
    function's handler. Both exits begin with [finish], which ends the
    claims by an atomic addition, waits in C for a consumption under way,
    and releases the claims in one C call; the compiler checks that it holds
    no poll point.
    In native code no poll point lies between the last claim and the
    handler, or between [f]'s return or raise and the release, so an
    asynchronous exception, such as [Sys.Break] or a signal handler's,
    leaves no claim behind.

    {!Submission.submit} takes read claims of its own, with
    {!Memory.claim_add}, on the buffers of a run, and releases them once the
    driver's [submit] has returned. A submission holds its fixed memory
    ({!hold}) for its life. A copy, blit or fill ({!Copy}) holds read claims
    on its buffers until it returns ({!reading2}, {!wait}). *)

val read_bytes :
  Def.buffer ->
  ('a, 'b) Bigarray.kind ->
  (('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'c) ->
  'c
(** [read_bytes b k f] is {!Rig.Buffer.read}[ b k f]. Its claim ends in C as
    [f] returns or raises, with no poll point before it. *)

val write_bytes :
  Def.buffer ->
  ('a, 'b) Bigarray.kind ->
  (('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'c) ->
  'c
(** [write_bytes b k f] is {!Rig.Buffer.write}[ b k f]. *)

val wait : string -> Def.access -> Def.buffer -> unit
(** [wait fn access b] takes a read claim on [b]'s memory for a call of [fn],
    then checks [b] under it and waits as {!Rig.Buffer.wait}[ b access] does.
    The caller ends the claim with {!Memory.claim_drop}. It allocates nothing
    when the work the access follows is done.

    Raises as {!reading} does, and {!Rig.Lost} as {!Rig.Buffer.wait} does,
    holding no claim. *)

val reading : string -> Def.buffer -> (unit -> 'a) -> 'a
(** [reading fn b f] is [f ()] under a read claim on [b]'s memory for a call of
    [fn]. It checks [b] live under the claim, as {!with_} does, and the claim
    ends in C as [f] returns or raises.

    Raises [Invalid_argument] naming [fn] if the memory is held exclusive or
    [b] is dead. *)

val reading2 :
  string ->
  Def.buffer ->
  Def.buffer ->
  (Def.buffer -> Def.buffer -> 'a -> 'b) ->
  'a ->
  'b
(** [reading2 fn a b f x] is [f a b x] under read claims on [a]'s and
    [b]'s memory, as {!reading} takes them. It allocates nothing when [f] is
    a top-level function.

    Raises as {!reading} does, for either buffer. *)

val share : Def.buffer -> unit
(** [share b] is {!Rig.Claim.share}[ b]. Memory already outside the claims
    costs two loads and no call. *)

type t
(** The type for the claims of one {!with_}: its buffers, and whether its
    function returned or raised, after which the claims hold nothing. *)

val with_ : read:Def.buffer list -> donate:Def.buffer list -> (t -> 'a) -> 'a
(** [with_ ~read ~donate f] is {!Rig.Claim.with_}[ ~read ~donate f]. *)

val exclusive : t -> Def.buffer -> bool
(** [exclusive c b] is {!Rig.Claim.exclusive}[ c b]. *)

val consume : t -> why:string -> Def.buffer -> Def.buffer
(** [consume c ~why b] is {!Rig.Claim.consume}[ c ~why b]. *)

val free : Def.buffer -> bool
(** [free b] is {!Rig.Buffer.free}[ b]. One C step takes a read claim,
    holds the memory exclusive if that claim is its only one, and consumes
    it, so a claim that follows finds [b] dead. A borrow maps under a read
    claim, so no free returns memory that a borrow maps meanwhile. *)

val consumable : t -> Def.buffer -> bool
(** [consumable c b] is {!Rig.Claim.consumable}[ c b]. *)

type hold
(** The type for holds: a token whose collection releases them
    ({!Memory.hold_token}), and their record. *)

val held : string -> Def.buffer list -> hold
(** [held fn bs] is {!Rig.Claim.hold}[ bs] for a call of [fn], which its
    errors name. *)

val hold : Def.buffer list -> hold
(** [hold bs] is {!held}[ "Claim.hold" bs]. *)

val release : hold -> unit
(** [release h] is {!Rig.Claim.release}[ h]. *)
