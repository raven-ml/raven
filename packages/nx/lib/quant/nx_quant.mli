(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quantised weights.

    A quantised weight is a float array of shape [[| ...; n; k |]], [n] outputs
    by [k] inputs, stored in a block format: blocks of values along [k], each as
    codes and a shared scale, in the bytes a file holds. Each constructor takes
    those bytes, so a mapped file is used without a copy. {!dequant} is its
    meaning, {!apply} its product, {!take} a gather of its rows.

    A GGUF file's block formats are read from the [uint8] tensors
    [Nx_io.load_gguf] gives, after matching on the tensor's stored type:
    [Nx_quant.q8_0 t] for a [Q8_0] tensor [t], whose shape is the weight's with
    its last axis in bytes.

    {!take} gathers before decoding, so a caller reads a large table's rows
    without decoding the table. Routing positions to a stack of weights is
    {!Nx.map_segments}:

    {[
    let gate_up =
      Nx_quant.mxfp4 ~scales
        (Nx.reshape [| experts; outputs; inputs / 2 |] blocks)
    in
    (* ids : [| tokens; k |], x : [| tokens; inputs |] *)
    Nx.map_segments ~segments:experts ids
      (fun e rows ->
        Nx_quant.apply (Nx_quant.take ~axis:0 ~indices:e gate_up) rows)
      (Nx.unsqueeze ~axes:[ -2 ] x)
    ]} *)

(** {1:weights Weights} *)

(** The type for quantised weights. Match on a weight to read its parts; only
    the constructors build one.

    A GGUF format stores each block of [k]'s values as one run of bytes, in the
    layout of ggml's [block_q8_0], [block_q4_K] and [block_q6_K]
    (ggml-common.h). Its float16 fields are little-endian. In every format a
    block of zero bytes decodes to zeros, [-0.] in Q6_K. *)
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
  | Q8_0 of {
      blocks : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 32 * 34 |]]: per 32 values, a float16 scale [d]
              then 32 int8 quants. *)
    }  (** Q8_0 (GGUF). A value is [d * q] for its quant [q]. *)
  | Q4_K of {
      blocks : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 256 * 144 |]]: per 256 values, float16 [d] and
              [dmin], 12 bytes packing a 6-bit scale and a 6-bit min for each of
              8 sub-blocks of 32 values, and 128 bytes of 4-bit quants. *)
    }
      (** Q4_K (GGUF). A value is [d * sc * q - dmin * m] for its quant [q] and
          its sub-block's scale [sc] and min [m]. *)
  | Q6_K of {
      blocks : (int, Nx.uint8_elt) Nx.t;
          (** [[| ...; n; k / 256 * 210 |]]: per 256 values, 128 bytes of the
              quants' low 4 bits, 64 of their high 2 bits, an int8 scale for
              each of 16 sub-blocks of 16 values, and a float16 [d]. *)
    }
      (** Q6_K (GGUF). A value is [d * sc * (q - 32)] for its 6-bit quant [q]
          and its sub-block's scale [sc]. *)

val mxfp4 : scales:(int, Nx.uint8_elt) Nx.t -> (int, Nx.uint8_elt) Nx.t -> t
(** [mxfp4 ~scales codes] is the MXFP4 weight with [codes] and [scales]. Only
    their shapes and placements are read: no byte of either is.

    Raises [Invalid_argument] naming the part if [codes] does not have shape
    [[| ...; n; k / 2 |]] with [k] a multiple of 32, or if [scales] does not
    have shape [[| ...; n; k / 32 |]], and naming the axis if [codes] is split
    over devices inside a group of 32 values. *)

val q8_0 : (int, Nx.uint8_elt) Nx.t -> t
(** [q8_0 blocks] is the Q8_0 weight stored as [blocks]. Only its shape and
    placement are read.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; n; k / 32 * 34 |]] with [k] a multiple of 32, or, naming the axis,
    if [blocks] is split over devices inside a block. *)

val q4_k : (int, Nx.uint8_elt) Nx.t -> t
(** [q4_k blocks] is the Q4_K weight stored as [blocks]. Only its shape and
    placement are read.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; n; k / 256 * 144 |]] with [k] a multiple of 256, or, naming the
    axis, if [blocks] is split over devices inside a block. *)

val q6_k : (int, Nx.uint8_elt) Nx.t -> t
(** [q6_k blocks] is the Q6_K weight stored as [blocks]. Only its shape and
    placement are read.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; n; k / 256 * 210 |]] with [k] a multiple of 256, or, naming the
    axis, if [blocks] is split over devices inside a block. *)

val shape : t -> int array
(** [shape w] is [w]'s logical shape, [[| ...; n; k |]]. *)

(** {1:products Products} *)

val dequant : (float, 'b) Nx.dtype -> t -> (float, 'b) Nx.t
(** [dequant dt w] is the values of [w] at [dt], of shape [shape w], each
    rounded once to [dt].

    A value of [w] is a float32 value. An MXFP4 value is exact barring overflow:
    codes of magnitude 4 or more at scale byte 253, and of 2 or more at 254, are
    infinite at every dtype. Float32, bfloat16 and float64 hold every other
    value exactly. A GGUF value is computed as ggml's float32 dequantisation
    computes it, its products left to right, and equals it bit for bit: a Q8_0
    value and a Q4_K product are exact, a Q4_K value rounds once at its
    subtraction and a Q6_K value once at its last product. A block whose float16
    scale is infinite or NaN gives infinite or NaN values.

    MXFP4 at bfloat16 decodes by keeping the high half of each value's float32
    bits: every MXFP4 value is exact at bfloat16, so nothing is rounded. *)

val apply : t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply w x] is [w] applied to [x] as a linear map from [k] inputs to [n]
    outputs: with [a] the wider of float32 and [x]'s dtype,

    {[
    Nx.cast (Nx.dtype x)
      (Nx.matmul (Nx.cast a x) (Nx.matrix_transpose (dequant a w)))
    ]}

    [w] is [[| ...; n; k |]], [x] is [[| ...; m; k |]] or [[| k |]], batch axes
    broadcast as {!Nx.matmul}'s, and the result is [[| ...; m; n |]], or
    [[| ...; n |]] for a vector.

    When every value of [w]'s format is exact at [x]'s dtype, as MXFP4's are at
    bfloat16, this is
    [Nx.matmul x (Nx.matrix_transpose (dequant (Nx.dtype x) w))].

    Raises [Invalid_argument] if [x] is a scalar, if [x]'s last axis is not [k],
    or if the batch axes do not broadcast. *)

val take : axis:int -> indices:Nx.int64_t -> t -> t
(** [take ~axis ~indices w] is the rows or matrices of [w] at [indices] along
    [axis], gathered from its packed parts with no value decoded, [indices] read
    as {!Nx.take} reads them. At every index in range,
    [dequant dt (take ~axis ~indices w)] is
    [Nx.take ~axis ~indices (dequant dt w)]. An index out of range gathers zero
    bytes, which decode to zeros.

    Raises [Invalid_argument] if [axis] is [w]'s last axis, negative or not,
    since a gather along it would cut blocks, if [axis] is out of bounds, or as
    {!Nx.take} does. *)

(** {1:structure Structure} *)

val ptree : t Nx.Ptree.t
(** [ptree] is a weight as a structure: its case, ["mxfp4"], ["q8_0"], ["q4_k"]
    or ["q6_k"], then each part as a tensor at its field's name, [codes] then
    [scales] or [blocks], rebuilt with the case's checks, which refuse a part
    split over devices inside a block. Its parts are tensors of a fixed type, so
    casts and {!Nx.Ptree.Payload} operations keep them. Weights of two formats
    are different structures; [Nx.Ptree.place ptree] places a weight and
    [Nx.Ptree.Walk.structure ptree] walks one inside a model.

    A walk that returns parts of other shapes or splits raises
    [Invalid_argument] as the constructors do, naming [Nx_quant.ptree]. *)
