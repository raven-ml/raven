(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pieces of a contraction's outputs, each a box whose operands' casts fit a
    buffer of a given size (nx_metal.mli's Contraction).

    The axes are the outputs' ([dst]'s): the batch axes, then [a]'s free axes,
    then [b]'s. A box takes indices along each, and the casts of a box are its
    operands' slices: a {e line} of [a] is a row with its terms, of [b] a column
    with its terms, and a box holds a line of [a] for each index of its axes
    along which [a] varies, and so for [b]. *)

type axis = {
  extent : int;
  a : bool;  (** Whether [a]'s cast varies along it. *)
  b : bool;  (** Whether [b]'s cast varies along it. *)
  step : int;
      (** A box's range of more than one index along it starts and ends at a
          multiple of [step], or at the extent: the blocks of a scale along it.
      *)
}
(** The type for the outputs' axes. *)

type box = { lo : int array; len : int array }
(** The type for boxes: along axis [i], indices from [lo.(i)] to below
    [lo.(i) + len.(i)], [len.(i)] at least [1]. *)

val bytes : axis array -> a:int -> b:int -> box -> int * int
(** [bytes axes ~a ~b x] is the bytes of [x]'s casts of [a] and of [b], [a] and
    [b] the bytes of a line's cast, [0] for an operand not cast. *)

type t = {
  boxes : box array;  (** In order, partitioning the outputs. *)
  a_bytes : int;  (** The most bytes of a box's cast of [a]. *)
  b_bytes : int;  (** The most bytes of a box's cast of [b]. *)
}
(** The type for pieces. *)

val cut : w:int -> batch:int -> rows:int -> axis array -> a:int -> b:int -> t
(** [cut ~w ~batch ~rows axes ~a ~b] is the pieces of outputs along [axes], of
    which the first [batch] are the batch's and the next [rows] [a]'s free axes,
    for casts of [a] and [b] lines of [a] and [b] bytes:
    + one box, every output, where its casts take at most [w] bytes;
    + else boxes of whole batch elements, the most whose casts take at most [w];
    + else, for each batch element: where an operand's cast takes at most
      [w / 2], or none, it is one box of that operand, and the other's axes are
      cut within what [w] leaves; else each operand's axes are cut within
      [w / 2], those of the operand of the larger cast outside the other's.

    An operand's axes are cut lexicographically: a box fixes one index of the
    outer axes, takes a range of one axis and the inner axes whole, the range a
    multiple of 64 where it is of the innermost axis and has 64 indices or more.
    A box is at least one line of each operand, which may pass [w]. Each extent
    is positive. *)

val refine : most:int -> axis array -> box -> box list
(** [refine ~most axes x] is [x] cut into boxes, in order, partitioning it, each
    of at most [most] outputs: [x] where it is; else [x]'s axes are cut
    lexicographically as {!cut} cuts an operand's, starting from [x]'s ranges,
    down to one output a box. A range of more than one index along an axis of
    [step] starts on a multiple of it where [x]'s does. *)
