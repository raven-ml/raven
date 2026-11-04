(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {!Nx.attention} under {!Rune.jit} against a float64 reference.

    A case draws query rows, keys and values on the host, in float32 rounded
    to its dtype, places them on a set of devices and compiles the call
    there. The reference is the host's eager call of the same rounded values
    in float64, and [M], each element's [Σ p |v|], the same call over [|v|].
    The compiled [o] and [lse] hold nx.mli's bound read with rune.mli's
    constants: the compiled [Nx.logsumexp]'s [ε], and, for a narrow float,
    the weights rounded once to [v]'s dtype. *)

(** The type for masks. *)
type mask =
  | Unmasked
  | Causal
  | Window  (** Causal over the last 8 keys. *)
  | Documents  (** Captured document ids of queries and keys, sorted. *)
  | Blocks  (** A captured table of visible blocks of 4 by 4 pairs. *)
  | Opaque
      (** Causal through floats, which a tile's ranges cannot class: each
          tile of the diagonal votes. *)

type t = {
  dtype : Nx.Dtype.any;  (** float16, bfloat16, float32 or float64. *)
  lead : int;
  hq : int;
  hk : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  page : int option;  (** Paged by pages of this many keys. *)
  holes : bool;  (** Some table entries out of range. *)
  lengths : int array option;  (** One a lead index, within [[0, tk]]. *)
  mask : mask;
  alibi : bool;  (** A score program reading captured slopes. *)
  rising : bool;  (** Scores that rise along the keys. *)
  positive : bool;  (** Values all positive. *)
  nans : bool;
      (** NaN in the values and keys past each row's length, which no row
          reads. *)
  seed : int;
}
(** The type for cases. *)

val pp : Format.formatter -> t -> unit

val gen : Nx.Dtype.any list -> t Windtrap.Gen.t
(** [gen dtypes] draws cases of [dtypes]: one to three lead indices, heads
    of [1/1], [8/8], [8/2] and [8/1], [tq] and [tk] around the tiles' edges
    ([0], [1], [15] to [17], [31] to [33], [63] to [65]), each mask, pages
    of 16 keys and of fewer with entries out of range, lengths of [0], [1]
    and [tk], and the adversarial rows: positive values, rising scores. *)

val examples : Nx.Dtype.any list -> t list
(** [examples dts] is cases each draw may miss, of [dts] in turn: a decode
    step of 64 query heads over 8 key heads and 1000 keys, which splits its
    keys, a prefill of 300 rows under each mask, and zero-size shapes. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks [c] compiled on [s]'s devices, and that the call is
    compiled, not evaluated. *)

val grad_law : (module Nx.Devices) -> vmap:bool -> t -> unit
(** [grad_law s ~vmap c] checks the gradient in the query, keys and values of
    a loss over [c]'s [o] and [lse], compiled on [s]'s devices, against the
    host's eager float64 gradient within the bound rune.mli's numerics
    compose, and that the call is compiled. Where [vmap], the compiled
    gradient is that of each lead index, mapped by {!Rune.vmap}.

    Raises [Invalid_argument] if [vmap] and [c] has pages or lengths. *)
