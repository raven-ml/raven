(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Claims: reads, takes and keeps of an allocation's memory, which
    {!Rig.section-claims} states.

    Each lifetime of an allocation has one claim, in its {!Def.alloc}, which
    every memory record over it shares: a claim word
    ({!Memory.section-claims}), a generation, and the reason of the last
    take or free. A read claim adds {!Memory.one_claim} to the word by
    compare-and-set, and a keep's hold {!Memory.one_hold}.

    {!take} makes its checks and its effects in one C step. It turns the
    word of each buffer's memory from 0, no claim, hold or bit, into -1,
    checks each buffer live and its work no lost device's, then raises
    each generation by 2 from its buffer's own, which kills every buffer
    made before, and gives each word back 0. Where a word holds a claim,
    hold or bit, it gives back the words it turned and answers [None]. Two
    buffers over one memory find the word the step turned already, so they
    answer [None]. A claim or hold that meets -1 spins until the step ends.
    A step that meets another's -1 gives back the words it turned, spins
    until that word is given back, and tries again: it waits only holding
    no word, so every step ends whatever the spinners hold. A claim is
    taken before its buffer is checked live: its compare-and-set reads the
    word the step gave back, or a later one, so its check sees the new
    generation.

    A keep ({!val-keep}) adds {!Memory.one_hold} to each word, then checks its
    buffer live, so either a take finds the hold and answers [None], or the
    keep finds the buffer dead and raises. A release ({!release}) clears
    each hold's mark in its record and takes {!Memory.one_hold} off the word
    in one C step, so each hold ends once, whichever release, on any
    domain, or drain after the keep's collection comes first.

    {!Submission.submit} takes read claims of its own, with
    {!Memory.claim_add}, on the buffers of a run, and releases them once the
    driver's [submit] has returned. A submission keeps its fixed memory
    ({!kept}) for its life. A copy, blit or fill ({!Copy}) holds read claims
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
(** [write_bytes b k f] is {!Rig.Buffer.write}[ b k f]. [f] writes the
    window of [Mapped] memory ({!Buffer.windowed}), which the owner's driver
    flushes once [f] returns or raises, before the claim ends, the stores and
    the flush one counted call ({!Memory.stored}). *)

val wait : string -> Def.access -> Def.buffer -> unit
(** [wait fn access b] takes a read claim on [b]'s memory for a call of [fn],
    then checks [b] under it and waits as {!Rig.Buffer.wait}[ b access] does.
    The caller ends the claim with {!Memory.claim_drop}. It allocates nothing
    when the work the access follows is done.

    Raises as {!reading} does, and {!Rig.Lost} as {!Rig.Buffer.wait} does,
    holding no claim. *)

val stored : string -> Def.buffer -> ('a -> 'b) -> 'a -> 'b
(** [stored fn b f x] is [f x], the host's stores into [b]'s window under a
    read claim that it takes as {!wait}[ fn Read_write b] does, and the
    owner's flush after them as one counted call ({!Memory.stored}), then the
    claim's end, also where either raises. Nothing allocates between the
    claim and the handler that ends it, so an asynchronous exception keeps
    no claim: the caller builds [f] before the call. *)

val reading : string -> Def.buffer -> (unit -> 'a) -> 'a
(** [reading fn b f] is [f ()] under a read claim on [b]'s memory for a call of
    [fn]. It checks [b] live under the claim, and the claim ends in C as [f]
    returns or raises.

    Raises [Invalid_argument] naming [fn] if [b] is dead. *)

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
(** [share b] is {!Rig.Buffer.share}[ b]. Memory already outside the claims
    costs two loads and no call. *)

val take : why:string -> Def.buffer list -> Def.buffer list option
(** [take ~why bs] is {!Rig.Buffer.take}[ ~why bs]. *)

val free : ?why:string -> Def.buffer -> bool
(** [free ~why b] is {!Rig.Buffer.free}[ ~why b]. Its C step is {!take}'s
    for one buffer, without the check for a loss, and waits out another take
    or free of the memory under way. A borrow maps under a read claim, so no
    free returns memory that a borrow maps meanwhile. *)

val unborrow : ?why:string -> Def.buffer -> bool
(** [unborrow ~why b] is {!Rig.Buffer.unborrow}[ ~why b]. A claim or a hold
    through a borrow counts on its lease as well as on its memory's word
    ({!Memory.claim_add}, [rig_buffer_claim]), and one C step moves the
    lease's count from no claims or holds to its end, so a claim through the
    borrow that follows finds its buffers dead. The memory's word is not
    touched. *)

type keep
(** The type for keeps: a token whose collection releases their holds
    ({!Memory.hold_token}), and their record. *)

val kept : string -> Def.buffer array -> keep
(** [kept fn bs] is {!Rig.Buffer.val-keep}[ bs] for a call of [fn], which its
    errors name. *)

val keep : Def.buffer list -> keep
(** [keep bs] is {!kept}[ "Buffer.keep" bs]. *)

val release : keep -> unit
(** [release k] is {!Rig.Buffer.release}[ k]. *)
