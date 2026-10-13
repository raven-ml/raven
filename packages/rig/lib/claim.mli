(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Claims: the implementation of {!Rig.Claim}.

    Each allocation has one claim ({!Def.claim}), which every memory record
    over it shares: a claim word ({!Memory.section-claims}), a generation,
    and the reason of the last consumption. A read claim adds
    {!Memory.one_claim} to the word by compare-and-set, and is refused while
    the word is negative, that is while the memory is held exclusive.

    {!with_} takes a read claim on every buffer it is given. Then, for each
    donated buffer that spans its memory, it turns the word, if it holds that
    one claim and no bit, into {!Memory.exclusive}. {!consume} raises the
    generation, which kills every buffer made before, then moves an exclusive
    word to {!Memory.consumed}. When {!with_}'s function returns or raises,
    each word still negative goes back to one read claim, outside the claims
    if the consumer exported the memory, and every read claim is released.

    A claim is taken before the buffer is checked live. A consumption on
    another domain has raised the generation before it releases its claims,
    so a compare-and-set that reads the word after that release is followed
    by a check that sees the new generation. A claim refused by a hold that
    has since ended is tried again.

    {!with_} takes its claims in one C call, the last step before its
    function's handler, and ends them in one C call, the first step of both
    exits. In native code no poll point lies between the last claim and the
    handler, or between [f]'s return or raise and the release, so an
    asynchronous exception, such as [Sys.Break] or a signal handler's,
    leaves no claim behind.

    {!Submission.submit} takes read claims of its own, in C, on the buffers
    of a run and the fixed memory, and releases them once the driver's
    [submit] has returned. A transfer through staging memory ({!Copy})
    holds read claims on its buffers until it returns ({!hold}). *)

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

val hold : string -> Def.buffer -> (unit -> 'a) -> 'a
(** [hold fn b f] is [f ()] under a read claim on [b]'s memory for a call of
    [fn]. It checks [b] live under the claim, as {!with_} does, and the claim
    ends in C as [f] returns or raises.

    Raises [Invalid_argument] naming [fn] if the memory is held exclusive or
    [b] is dead. *)

val hold2 :
  string ->
  Def.buffer ->
  Def.buffer ->
  (Def.buffer -> Def.buffer -> 'a -> 'b) ->
  'a ->
  'b
(** [hold2 fn a b f x] is [f a b x] under read claims on [a]'s and [b]'s
    memory, as {!hold} takes them. It allocates nothing when [f] is a
    top-level function.

    Raises as {!hold} does, for either buffer. *)

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

val take : why:string -> Def.buffer -> Def.buffer option
(** [take ~why b] is {!Rig.Claim.take}[ ~why b]. *)
