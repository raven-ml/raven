(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quantised weights.

    A quantised weight is packed codes and their scales, in the byte layout a
    checkpoint stores, so a weight loaded from a mapped file is used without a
    copy. Its logical shape is [[| ...; n; k |]], [n] outputs by [k] inputs, as
    files store linear layers, with blocks of values running along [k] inside
    each row. {!dequant} is its meaning and {!apply} its one product.

    {!apply} and {!dequant} perform {!Effect.E_quant}. Run eagerly, where no
    handler takes the effect, they decode one bounded chunk at a time, whatever
    the weight's size.

    A quantised weight has no gradient: build it once and capture it.

    {[
    let gate_up =
      Nx_quant.mxfp4 ~scales
        (Nx.reshape [| experts; outputs; inputs / 2 |] blocks)
    in
    Nx_quant.apply ~ids gate_up x
    ]} *)

(** {1:weights Weights} *)

(** The type for quantised weights. Match on a weight to read its parts; only
    the constructors build one. Block-scaled FP8, designed in RFC 0004, lands
    later as a second format. *)
type t = private
  | Mxfp4 of {
      codes : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 2 |]]: two e2m1 codes per byte, the low nibble
              first. A code is a sign bit over a magnitude among 0, 0.5, 1, 1.5,
              2, 3, 4 and 6. *)
      scales : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 32 |]]: one e8m0 scale per 32 values. *)
    }
      (** MXFP4 (OCP Microscaling Formats, 2023). A value is its code's
          magnitude, signed, times [2 ^ (s - 127)] for its group's scale byte
          [s]; the byte [255] makes its whole group NaN. *)

val mxfp4 : scales:(int, Nx.uint8_elt) Nx.t -> (int, Nx.uint8_elt) Nx.t -> t
(** [mxfp4 ~scales codes] is the MXFP4 weight with [codes] and [scales]. Only
    their shapes are read: no byte of either is.

    Raises [Invalid_argument] naming the part if [codes] does not have shape
    [[| ...; n; k / 2 |]] with [k] a multiple of 32, or if [scales] does not
    have shape [[| ...; n; k / 32 |]]. *)

val shape : t -> int array
(** [shape w] is [w]'s logical shape, [[| ...; n; k |]]. *)

val place : Nx.Placement.t -> t -> t
(** [place p w] is [w] with every part placed with [p] ({!Nx.place}). A leading
    axis or [n] splits wherever {!Nx.place} can split it; [k] splits only at a
    32-value group, so that each shard holds whole groups and their scales.

    Raises [Invalid_argument] naming the part and the axis if [p] splits [k]
    across a group, or as {!Nx.place} does for either part. *)

(** {1:products Products} *)

val dequant : (float, 'b) Nx.dtype -> t -> (float, 'b) Nx.t
(** [dequant dt w] is the values of [w] at [dt], of shape [shape w].

    Values are exact at float32 and bfloat16 barring overflow: MXFP4 codes of
    magnitude 4 or more at scale byte 253, and of 2 or more at 254, are
    infinite. Other dtypes round each value once.

    Eagerly the values are decoded a chunk of 2{^ 22} values at a time into the
    result: beside it, only a few chunk-sized temporaries exist at once, so the
    peak does not grow with the weight. *)

val apply :
  ?ids:(int32, Nx.int32_elt) Nx.t -> t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply ?ids w x] is
    [Nx.matmul x (Nx.matrix_transpose (dequant Nx.float32 w))] at [x]'s dtype,
    with {!Nx.matmul}'s shapes: [w] is [[| ...; n; k |]], [x] is
    [[| ...; m; k |]] or [[| k |]], and batch axes broadcast. Products
    accumulate at float32 and the result is rounded once to [x]'s dtype.

    With [ids], [w] is [[| b...; e; n; k |]], [e] experts behind [b] leading
    axes, and [ids] is [[| b...; s... |]], its [b] leading axes broadcasting
    against [w]'s. The product is then over [w'] of shape
    [[| b...; s...; n; k |]], whose matrix at [(b, s)] is [w]'s expert
    [ids.(b, s)] of lane [b]. A weight with no leading axes is gathered by [ids]
    whole: [apply ~ids w x] with [w] of shape [[| e; n; k |]], [ids] of shape
    [[| t; 4 |]] and [x] of shape [[| t; 1; 1; k |]] is [[| t; 4; 1; n |]].

    An id outside \[[0], [e]), [-1] included, selects no expert: its position of
    the result is exactly zero, whatever [x] holds there, and no weight is read
    for it. Ids may repeat. An empty [ids] or [x] gives an empty result.

    Eagerly each expert that [ids] selects, or each matrix of [w] without [ids],
    is decoded a chunk of 2{^ 22} values at a time: beside the result and [x],
    only a few chunk-sized temporaries exist at once, so the peak does not grow
    with the weight.

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis is not [k],
    if the batch axes do not broadcast, or, with [ids], if [w] has no expert
    axis or [ids] lacks [w]'s leading axes. *)

(** {1:effect The effect} *)

(** The effect that {!apply} and {!dequant} perform. A handler that transforms
    or compiles programs matches on it; where none does, the functions run
    eagerly. *)
module Effect : sig
  (** The type for an operation on a weight, whose result has type
      [(float, 'b) Nx.t]. *)
  type (_, _) op =
    | Apply : {
        ids : (int32, Nx.int32_elt) Nx.t option;
        x : (float, 'b) Nx.t;
        transpose : bool;
            (** [true] multiplies by the weight rather than by its transpose,
                [[| ...; m; n |]] to [[| ...; m; k |]]. {!apply} performs
                [false]. *)
      }
        -> (float, 'b) op  (** {!apply}. *)
    | Dequant : (float, 'b) Nx.dtype -> (float, 'b) op  (** {!dequant}. *)

  type _ Stdlib.Effect.t +=
    | E_quant : { w : t; op : ('a, 'b) op } -> ('a, 'b) Nx.t Stdlib.Effect.t
          (** The operation [op] on the weight [w]. *)

  val perform : t -> ('a, 'b) op -> ('a, 'b) Nx.t
  (** [perform w op] performs [E_quant { w; op }] and, where no handler takes
      it, runs [op] eagerly. {!apply} and {!dequant} are [perform]; a handler
      re-performs an operation with it in its enclosing context.

      Raises [Invalid_argument] as {!apply} does, before performing, if the
      shapes of an [Apply] do not agree. *)
end

(** {1:structure Structure}

    A weight is a structure without a parameter ({!Nx.Ptree.S} with
    [type _ t = t]): its parts are tensors of a fixed type, so casts and
    {!Nx.Ptree.Payload} operations keep them. *)

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> t -> t
(** [walk c w] walks [w] at [c]'s path: it reports the case ["mxfp4"], then
    walks [codes] at [codes] and [scales] at [scales], each with
    {!Nx.Ptree.Walk.tensor}. It checks the parts it rebuilds as {!mxfp4} does,
    and reads neither bytes nor placement, so it runs under every transformation
    and compiled trace. A model's own [walk] walks a quantised field with it:
    [field c "gate_up" Nx_quant.walk w].

    Raises [Invalid_argument] as {!mxfp4} does, naming [Nx_quant.walk], if a
    walk returns parts of other shapes. *)

val ptree : t Nx.Ptree.t
(** [ptree] is a weight as a structure at one type, walked by {!walk}: its
    visits are [the root: case "mxfp4"], [codes: a leaf] and [scales: a leaf].
    Compiled programs therefore key on the format, and [Nx.Ptree.map ptree f w]
    maps its parts. *)
