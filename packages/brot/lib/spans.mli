(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pretoken spans.

    A fixed-capacity buffer of byte ranges into an input string. A buffer owns
    no text: ranges index the string passed to {!Pre_tokenizer.fill}. Ranges are
    non-empty, non-overlapping and ascending, but need not tile the input — a
    pre-tokenizer may drop delimiters.

    A writer keeps the count of a run of spans in a local and publishes it with
    {!set_count} once the run is written, so that a hot loop pays two stores per
    span and nothing else.

    A buffer is written by one domain at a time. *)

type t
(** The type for span buffers. *)

val create : capacity:int -> t
(** [create ~capacity] is an empty buffer holding up to [capacity] spans. It
    occupies [8 * capacity] bytes. *)

val capacity : t -> int
(** [capacity t] is the number of spans [t] can hold. *)

val count : t -> int
(** [count t] is the number of spans in [t]. *)

val clear : t -> unit
(** [clear t] drops all spans of [t]. *)

val start : t -> int -> int
(** [start t k] is the first byte of span [k]. Unchecked: [k] must be less than
    [count t]. *)

val stop : t -> int -> int
(** [stop t k] is one past the last byte of span [k]. Unchecked: [k] must be
    less than [count t]. *)

val write : t -> int -> int -> int -> unit
(** [write t k s e] stores the range \[[s];[e]) as span [k]. Unchecked: [k] must
    be less than [capacity t], and [s] and [e] must fit in 32 bits. Spans become
    visible to {!count}, {!start} and {!stop} only through {!set_count}. *)

val set_count : t -> int -> unit
(** [set_count t n] makes [t] hold its first [n] spans. Unchecked: [n] must be
    at most [capacity t] and spans [0] to [n-1] must have been {!write}n. *)

(** {1:internals Internals} *)

val buffer : t -> Bytes.t
(** [buffer t] is [t]'s storage: span [k] is the eight bytes at [8 * k], the
    start in the low 32 bits and the stop in the high 32, native-endian. The C
    kernels write spans through it; {!set_count} publishes them. *)
