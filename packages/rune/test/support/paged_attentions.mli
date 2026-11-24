(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Paged {!Nx.attention} under {!Rune.jit} against the dense call over the
    same rows.

    A case draws pools of pages, a table naming them in a shuffled order
    and the query rows on the host, in float32 rounded to its dtype. The
    dense keys and values are the rows the table reads, zeros at a key
    whose entry is out of range, which the dense call's mask hides instead.
    Both calls compile on a set of devices; the paged one's [o] and [lse]
    have the dense one's bits (rune.mli's Attention). *)

(** The type for masks, each over the query's position, [q + len - tq]. *)
type mask =
  | Full  (** {!Nx.causal}. *)
  | Window  (** Causal over the last 24 keys, a constant width. *)
  | Reach
      (** Causal over the last 24 keys, a captured width: the mask reads a
          capture. *)

(** The type for the table's entries out of range, beside entries wholly
    past a row's length, which take [-1] unless [holes] is [Whole]. *)
type holes =
  | Whole  (** None. *)
  | Scattered  (** Every third entry, [-1] or the pools' page count. *)
  | Tile
      (** [-1] for keys [64] to [127], where [tk] reaches them: a tile of
          keys the mask shows and the table hides. *)

type t = {
  dtype : Nx.Dtype.any;  (** bfloat16, float16 or float32. *)
  lead : int;
  hq : int;
  hk : int;
  tq : int;
  page : int;
  max_pages : int;
  dk : int;
  dv : int;
  lengths : int array option;  (** One a lead index, within [[0, tk]]. *)
  mask : mask;
  holes : holes;
  nans : bool;
      (** NaN at the keys past each row's length and in the pages no entry
          names, which no row reads. *)
  seed : int;
}
(** The type for cases. [tk] is [max_pages · page]. *)

val pp : Format.formatter -> t -> unit

val gen : t Windtrap.Gen.t
(** [gen] draws bfloat16, float16 and float32, one or two lead indices, heads of
    [8/8], [8/2] and [8/1], a decode step or a prefill of up to 100 rows, pages
    of 1 and 16 keys over up to 320 keys, each mask and each kind of holes,
    lengths of [0], [tk] and between, and NaN where no row reads. *)

val examples : t list
(** [examples] reaches every cover of {!law}: decode steps and prefills of each
    mask and kind of holes, at pages of 1 and 16 keys, in bfloat16 and float32;
    and bfloat16 prefills of 256 rows of 128 query heads over one key head, keys
    of 272 dimensions and values of 64 and of 56, whose tiles' copies are within
    bytes of a workgroup's memory on CUDA. *)

val choices : t list
(** [choices] is float32 cases over which {!gen}'s choices that {!examples}
    leaves out are each made: heads of [8/8] and [8/1], [tq] of [7] and [64],
    [16], [64] and [100] keys, [dk] of [16] and [72], and [dv] of [8]. *)

val host_examples : t list
(** [host_examples] is {!examples} of fewer than 128 query heads, in float32,
    every cover but bfloat16's, then {!choices}. *)

val redrawn : t list -> t Windtrap.Gen.t
(** [redrawn cases] draws a case of the form of one of [cases], its seed and,
    where it has them, its lengths drawn anew: [0], [tk] and between. A compiled
    call of a drawn case reads its values from the arguments and captures of
    that form's program. *)

val law_call :
  paged:bool ->
  (module Nx.Devices with type d = 'd) ->
  t ->
  (float array * float array, 'd) Batch.call
(** [law_call ~paged s c] is {!law}'s paged call of [c] if [paged] and its dense
    call otherwise, compiled on [s]'s devices: its [o] and [lse] read as
    float64, which holds each of their values' bits. *)

val law_check :
  t ->
  (unit -> (float array * float array) * (float array * float array)) ->
  unit
(** [law_check c compiled] is {!law} of [c], [compiled ()] being what
    {!law_call} reads of the paged call, then of the dense call, each compiled
    in a program without the other. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that [c]'s paged call compiled on [s]'s devices gives the
    dense call's [o] and [lse] bit for bit, each compiled in a program of its
    own, not evaluated. *)
