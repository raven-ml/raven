(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stateless RNG keys and key splitting.

    Keys are deterministic integers that can be split to derive independent
    subkeys. *)

(** {1:keys Keys} *)

type key = int
(** The type for RNG keys. *)

val key : int -> key
(** [key seed] is a normalized 31-bit non-negative key derived from [seed]. *)

val hash_int : int -> int
(** [hash_int x] is a deterministic integer hash in the 31-bit non-negative
    range. *)

val split : ?n:int -> key -> key array
(** [split ?n k] deterministically derives [n] subkeys from [k].

    [n] defaults to [2]. *)

val fold_in : key -> int -> key
(** [fold_in k data] mixes [data] into [k] and returns the derived key. *)

val to_int : key -> int
(** [to_int k] is [k] as an integer. *)

(** {1:generator Stateful generator} *)

module Generator : sig
  type t
  (** The type for mutable key generators. *)

  val create : ?key:key -> unit -> t
  (** [create ?key ()] is a generator initialized with [key].

      If [key] is omitted, a random seed is drawn from [Random.bits]. *)

  val next : t -> key
  (** [next g] returns a fresh subkey and advances [g]'s internal key. *)

  val current_key : t -> key
  (** [current_key g] is [g]'s current internal key. *)
end
