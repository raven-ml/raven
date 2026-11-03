(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Running operations of a prefix: nx.mli promises that a running operation's
    result at an index is a function of the elements up to it alone, so that
    the compiled running operation of an axis's first [p] elements is the
    first [p] elements of the compiled running operation of the axis, bit for
    bit. *)

(** The type for running operations. *)
type running = Cumsum | Cumprod | Cummax | Cummin

type t =
  | Case : {
      dt : (float, 's) Nx.dtype;
      op : running;
      n : int;  (** The axis's length. *)
      p : int;  (** The prefix's length, at most [n]. *)
      axis : int;  (** [0] of a [[n; 2]] operand, or [1] of a [[2; n]] one. *)
      values : float array;  (** The [2 n] elements, in C order. *)
    }
      -> t  (** The type for cases. *)

val pp : Format.formatter -> t -> unit

val gen : t Windtrap.Gen.t
(** [gen] draws every float dtype and running operation, axes of each length
    either side of 16, 32 and 256, where the compiler's scan changes scheme,
    prefixes of those lengths and of any length, and elements across 17
    binades, near 1 for a product, with zeros of both signs, infinities and
    NaNs. *)

val examples : t list
(** [examples] holds a case of each length {!gen} draws, each dtype and each
    running operation; the empty prefix; float32 and float64 sums whose prefix
    of 32 or fewer, or past 32, is of a longer axis, of elements across 40
    binades; and sums, maxima and minima of zeros of both signs. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that the compiled running operation of [c]'s prefix on
    [s]'s first device has the bits of the prefix of the compiled running
    operation of the whole axis, every NaN read as one ({!Witness.codes}), with
    no call evaluated. It covers each length class, the empty prefix, a prefix
    of 32 or fewer of an axis past 32, a prefix past 32 of a longer axis, each
    running operation and each axis. *)
