(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integer arithmetic at its type's edges: nx.mli and nx_kernel's [Prog]
    promise that [Add], [Sub], [Mul], [Neg], [Abs] and a left shift on
    integers wrap modulo [2{^ bits}], that a quotient by [0] is [0] and a
    remainder by [0] the dividend, and that the least value's quotient by [-1]
    is itself and its remainder [0]. A compiled kernel gives eager's values
    there, and so do comparisons of those values, which a compiler that took
    overflow or a division by [0] for impossible would fold: [x + y > x] to
    [y > 0], [abs x < 0] to [false]. *)

(** The type for operations. [Neg], [Abs] and [Lshift n], {!Nx.lshift} by
    [n], read the first operand; [Div] is {!Nx.div} and [Mod] {!Nx.mod_}. *)
type op = Add | Sub | Mul | Neg | Abs | Lshift of int | Div | Mod

type t =
  | Case : {
      dt : ('v, 's) Nx.dtype;  (** An integer dtype. *)
      op : op;
      xs : int64 array;
      ys : int64 array;  (** As long as [xs]. *)
    }
      -> t
      (** The type for cases: [op] of [xs] and [ys], each element taken
          modulo [dt]'s width. *)

val pp : Format.formatter -> t -> unit

val gen : t Windtrap.Gen.t
(** [gen] draws every integer dtype and operation, a shift by any count up to
    the dtype's width, with 64 operand pairs, each operand one of its dtype's
    extremes, their neighbours, a value near [0], near half the greatest or
    anywhere. *)

val examples : t list
(** [examples] holds, for each integer dtype and operation, shifts by [1] and
    by one short of the width among them, every pair of the dtype's least and
    greatest values, their neighbours, [-2] to [2] and the values either side
    of half the greatest. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that on [s]'s first device the compiled [op x y], [op x y
    > x] and [op x y < 0] have eager's values, with no call evaluated. It
    covers each operation and each signed dtype, a case whose exact result
    leaves the dtype's range, and a division by [0]. *)
