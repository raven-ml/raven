(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integer buffers.

    A growable run of integers, cleared and refilled rather than reallocated. A
    document's encoding is four of them: the token ids, and the bounds and marks
    of the pretoken spans they came from.

    A buffer is written by one domain at a time. *)

type t
(** The type for integer buffers. *)

val create : ?capacity:int -> unit -> t
(** [create ()] is an empty buffer. [capacity] defaults to [64] integers. *)

val clear : t -> unit
(** [clear t] drops every integer of [t], keeping the buffer. *)

val length : t -> int
(** [length t] is the number of integers in [t]. *)

val get : t -> int -> int
(** [get t i] is the [i]th integer. Unchecked: [i] must be less than [length t].
*)

val add : t -> int -> unit
(** [add t n] appends [n] to [t]. *)

val truncate : t -> int -> unit
(** [truncate t n] drops the integers of [t] past the first [n]. Unchecked: [n]
    must be at most [length t]. *)

val add4 : t -> int -> int -> int -> int -> count:int -> unit
(** [add4 t a b c d ~count] appends the first [count] of [a], [b], [c] and [d].
    All four are stored whatever [count] is, so a caller holding them in
    registers pays no branch and the stores past [count] are dead. Unchecked:
    [count] must be at most [4]. *)

val to_array : t -> int array
(** [to_array t] is a fresh array of the integers of [t]. *)

val blit_to_int32 :
  t ->
  pos:int ->
  len:int ->
  (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  at:int ->
  unit
(** [blit_to_int32 t ~pos ~len dst ~at] writes the [len] integers of [t] from
    [pos] into [dst] from [at], narrowed to 32 bits. Unchecked: [pos + len] must
    be at most [length t] and [at + len] at most the length of [dst]. *)

(** {1:internals Internals}

    The raw view a kernel driver appends through. *)

val buffer : t -> int array
(** [buffer t] is [t]'s storage: the first [length t] entries are the contents.
    Invalidated by whatever grows [t]. *)

val capacity : t -> int
(** [capacity t] is the number of integers {!buffer} holds. *)

val reserve : t -> int -> unit
(** [reserve t extra] ensures [t] has room for [extra] more integers. *)

val set_length : t -> int -> unit
(** [set_length t n] makes [t] hold its first [n] entries. Unchecked: [n] must
    be at most {!capacity} and the entries below it must have been written. *)
